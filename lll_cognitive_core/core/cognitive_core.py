import time
import threading
from typing import Dict, Optional, Any
from lll_simple_ai_shared import (
    UnderstoodData,
    RecallResultsModels,
    BehaviorPlan,
    MemoryQueryType,
    EpisodicMemoriesModels,
    EpisodicMemoriesGenerateModels,
)

import queue
import logging

from .cache_memory_manager import CacheMemoryManager
from .data_structures import *
from .plugin_interfaces import (
    AssociativeRecallPlugin,
    BehaviorGenerationPlugin,
    MemoryManagerPlugin,
)


class CognitiveCore:
    """
    认知核心
    负责协调所有AI插件，维护记忆系统，生成智能行为
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 运行时记忆
        self.working_memory = WorkingMemory()

        self.episodic_memory_manager = CacheMemoryManager()  # 活跃情景记忆缓存

        # 插件初始化
        self.plugins = {
            "event_understanding": None,
            "associative_recall": None,
            "behavior_generation": None,
            "memory_extraction": None,
            "memory_manager": None,
        }

        self.last_deep_consolidation = time.time()
        self.last_light_consolidation = time.time()

        # 超过多少条历史记忆就使用专门的回想任务处理
        self.episodic_memories_direct_threshold = config.get(
            "episodic_memories_direct_threshold", 5
        )

        # 事件处理系统
        self.event_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None

        # 统计信息
        self.stats = {
            "events_processed": 0,
            "memory_consolidations": 0,
            "average_processing_time": 0.0,
            "last_consolidation_time": time.time(),
        }

        self.logger = logging.getLogger("CognitiveCore")

    def register_plugin(self, plugin_type: str, plugin_instance):
        """注册自定义插件"""
        if plugin_type in self.plugins:
            self.plugins[plugin_type] = plugin_instance
            self.logger.info(f"注册插件: {plugin_type}")
        else:
            self.logger.error(f"未知插件类型: {plugin_type}")

    def get_plugin(self, plugin_type: str):
        """获取插件实例（回退到默认实现）"""
        return self.plugins.get(plugin_type)

    def start(self):
        if self.is_running:
            return

        """启动认知核心"""
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()
        self.logger.info("CognitiveCore 启动成功")

    def stop(self):
        """停止认知核心"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.logger.info("CognitiveCore 已停止")

    def receive_event(self, raw_event: Dict[str, str]):
        """接收来自AI模块的原始事件"""
        try:
            event_with_context = UnderstandEventData(
                type=raw_event.get("type", ""),
                data=raw_event.get("data", ""),
                source=raw_event.get("source", ""),
                timestamp=time.time(),
            )
            self.event_queue.put(event_with_context)
        except Exception as e:
            self.logger.error(f"接收事件失败: {e}")

    def _processing_loop(self):
        """主处理循环"""
        while self.is_running:
            try:
                # 处理事件队列
                self._process_events()

                # 检查记忆整理条件
                self._check_memory_consolidation()

                # 更新系统状态
                self._update_system_state()

                time.sleep(0.02)  # 避免CPU过度占用

            except Exception as e:
                self.logger.error(f"处理循环错误: {e}")
                time.sleep(0.1)

    def _process_events(self):
        """处理事件队列"""
        processed_count = 0
        start_time = time.time()

        while (
            not self.event_queue.empty() and processed_count < 10
        ):  # 每轮最多处理10个事件
            try:
                event_data: UnderstandEventData = self.event_queue.get_nowait()
                self._process_single_event(event_data)
                processed_count += 1
                self.stats["events_processed"] += 1

            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"处理单个事件失败: {e}")

        # 更新处理时间统计
        if processed_count > 0:
            processing_time = time.time() - start_time
            avg_time = processing_time / processed_count
            self.stats["average_processing_time"] = (
                self.stats["average_processing_time"] * 0.9 + avg_time * 0.1
            )

    def _process_single_event(self, event_data: UnderstandEventData):
        """处理单个事件"""
        start_time = time.time()

        try:
            # 事件理解
            understood_data: UnderstoodData = self._understand_event(event_data)
            if not understood_data:
                return

            # 更新工作记忆
            self._update_working_memory(event_data, understood_data)

            # 行为生成和执行
            self._generate_and_execute_behavior()

            processing_time = time.time() - start_time
            self.logger.debug(
                f"事件处理完成: {understood_data.event_type}, 耗时: {processing_time:.3f}s"
            )

        except Exception as e:
            self.logger.error(f"处理事件失败: {e}")

    def _understand_event(
        self, event_data: UnderstandEventData
    ) -> Optional[Dict[str, Any]]:
        """事件理解阶段"""
        plugin = self.get_plugin("event_understanding")
        if not plugin:
            return None

        input_data = UnderstandEventInput(
            understand_event=event_data,
            recent_events=self.working_memory.recent_events,
            active_goals=self.working_memory.active_goals,
        )

        try:
            result = plugin.understand_event(input_data)
            return result
        except Exception as e:
            self.logger.error(f"事件理解插件错误: {e}")
            return None

    def _update_working_memory(
        self, event_data: UnderstandEventData, understood_data: UnderstoodData
    ):
        """更新工作记忆"""
        # 创建认知事件
        cognitive_event = CognitiveEvent(
            event_id=f"event_{int(time.time() * 1000)}",
            timestamp=time.time(),
            source=event_data.source,
            modality_type=event_data.type,
            raw_data=event_data,
            understood_data=understood_data,
            importance_score=understood_data.importance_score or 0,
        )

        # 添加到最近事件
        self.working_memory.recent_events.append(cognitive_event)
        # if len(self.working_memory.recent_events) > 50:
        #     self.working_memory.recent_events.pop(0)

        # 更新当前情境
        situation = understood_data.get("current_situation", None)
        if situation is not None:
            self.working_memory.current_situation = situation

        # 更新认知负荷
        self._update_cognitive_load()

        # 更新时间戳
        self.working_memory.last_update_time = time.time()
        self.working_memory.active_duration = time.time() - self.stats.get(
            "session_start_time", time.time()
        )

    def _generate_and_execute_behavior(self, understood_data: UnderstoodData):
        """生成和执行行为"""
        plugin: BehaviorGenerationPlugin = self.get_plugin("behavior_generation")

        if not plugin:
            return

        try:
            episodic_memories: List[EpisodicMemoriesModels] = []
            memory_manager: MemoryManagerPlugin = self.get_plugin("memory_manager")

            if memory_manager and understood_data.memory_query_plan:
                if (
                    understood_data.memory_query_plan.query_type
                    == MemoryQueryType.LONG_TERM_FRESH
                ):
                    # 从文件获取
                    episodic_memories: List[EpisodicMemoriesModels] = (
                        memory_manager.query_episodic_memories(
                            date_range=understood_data.memory_query_plan.time_range,
                            keywords=understood_data.memory_query_plan.query_triggers,
                        )
                    )
                    # 保存到缓存
                    self.episodic_memory_manager.save_episodic_memories(
                        episodic_memories
                    )
                elif (
                    understood_data.memory_query_plan.query_type
                    == MemoryQueryType.LONG_TERM_CACHED
                ):
                    # 从缓存获取
                    episodic_memories: List[EpisodicMemoriesModels] = (
                        self.episodic_memory_manager.query_episodic_memories(
                            date_range=understood_data.memory_query_plan.time_range,
                            keywords=understood_data.memory_query_plan.query_triggers,
                        )
                    )

            # 获取联想回忆结果
            episodic_memories_text: str | None = None
            if len(episodic_memories) > self.episodic_memories_direct_threshold:
                result = self._associative_recall(episodic_memories)
                if result:
                    episodic_memories_text = result.recalled_episode
                    if result.current_situation:
                        self.working_memory.current_situation = result.current_situation

            cognitive_state = GenerateBehaviorInput(
                current_situation=self.working_memory.current_situation,
                recent_events=self.working_memory.recent_events,
                episodic_memories=episodic_memories,
                active_goals=self.working_memory.active_goals,
                episodic_memories_text=episodic_memories_text,
                social_norms=[],
            )

            behavior_plan: BehaviorPlan = plugin.generate_behavior(cognitive_state)

            # 更新情境
            if behavior_plan.current_situation:
                self.working_memory.current_situation = behavior_plan.current_situation

            self._execute_behavior_plan(behavior_plan)
        except Exception as e:
            self.logger.error(f"行为生成插件错误: {e}")

    def _associative_recall(
        self, episodic_memories: List["EpisodicMemoriesModels"]
    ) -> RecallResultsModels | None:
        """联想回忆"""
        plugin: AssociativeRecallPlugin = self.get_plugin("associative_recall")

        if not plugin:
            return None

        recall_request = AssociativeRecallInput(
            current_situation=self.working_memory.current_situation,
            recent_events=self.working_memory.recent_events,
            episodic_memories=episodic_memories,
            active_goals=self.working_memory.active_goals,
        )

        try:
            return plugin.associative_recall(recall_request)
        except Exception as e:
            self.logger.error(f"联想回忆插件错误: {e}")
            return None

    def _execute_behavior_plan(self, behavior_plan: BehaviorPlan):
        """执行行为计划"""
        if not behavior_plan or "plan" not in behavior_plan:
            return

        # 这里应该通过Orchestrator发送到对应的AI模块
        for action in behavior_plan.plan:
            self.logger.info(f"执行行为: {action}")
            self._update_working_memory(
                UnderstandEventData(
                    type=action.type,
                    data=action.data,
                    source="me",
                    timestamp=time.time(),
                ),
                UnderstoodData(
                    event_type="my_action",
                    confidence=1.0,
                    response_priority="medium",
                    expected_response="none",
                    main_content=action.data,
                    event_entity="me",
                    key_entities=[],
                    importance_score=50,
                ),
            )
            # TODO: 通过HTTP发送到Orchestrator

    def _update_cognitive_load(self):
        """更新认知负荷"""
        # 基于事件数量、目标复杂度等计算认知负荷
        event_count = len(self.working_memory.recent_events)
        goal_complexity = len(self.working_memory.active_goals)
        memory_usage = len(self.working_memory.recent_events) / 50.0

        self.working_memory.cognitive_load = min(
            1.0, event_count * 0.02 + goal_complexity * 0.1 + memory_usage * 0.3
        )

    def _check_memory_consolidation(self):
        """检查记忆整理条件"""
        pass

    def _consolidate_memories(self, consolidation_type: str):
        """执行记忆整理"""
        # 获取记忆提取插件
        extraction_plugin = self.get_plugin("memory_extraction")

        if not extraction_plugin or not extraction_plugin:
            return

        try:
            # 记忆提取阶段
            extraction_data = ExtractMemoriesInput(
                current_situation=self.working_memory.current_situation,
                recent_events=self.working_memory.recent_events,
                active_goals=self.working_memory.active_goals,
            )

            extraction_result: List[EpisodicMemoriesGenerateModels] = (
                extraction_plugin.extract_memories(extraction_data)
            )

            event_map: Dict[str, CognitiveEvent] = {}
            for event in self.working_memory.recent_events:
                event_map[event.event_id] = event

            result: List[EpisodicMemoriesModels] = []

            for generate_model in extraction_result:
                generate_model: EpisodicMemoriesGenerateModels
                # 根据 id 查找对应的 CognitiveEvent
                cognitive_event = event_map.get(generate_model.id)

                # 如果找不到对应的 event_id，则舍弃该数据
                if cognitive_event is None:
                    continue

                # 从 CognitiveEvent 中提取所需字段
                # 将 timestamp 从 float 转换为 datetime
                event_timestamp = datetime.fromtimestamp(cognitive_event.timestamp)

                # 创建 EpisodicMemoriesModels 对象
                episodic_model = EpisodicMemoriesModels(
                    id=generate_model.id,
                    content=generate_model.content,
                    importance=generate_model.importance,
                    keywords=generate_model.keywords,
                    associations=generate_model.associations,
                    timestamp=event_timestamp,
                    entities=cognitive_event.understood_data.key_entities,
                    source=cognitive_event.source,
                )

                result.append(episodic_model)

            memory_manager: MemoryManagerPlugin = self.get_plugin("memory_manager")

            if memory_manager:
                # 保存到文件
                memory_manager.save_episodic_memories(result)

            # TODO: 清理self.episodic_memory_manager

            self._apply_consolidation_result(consolidation_type)

            # 更新整理时间
            if consolidation_type == "deep":
                self.last_deep_consolidation = time.time()
            else:
                self.last_light_consolidation = time.time()

            self.stats["memory_consolidations"] += 1
            self.logger.info(f"{consolidation_type}记忆整理完成")

        except Exception as e:
            self.logger.error(f"记忆整理错误: {e}")

    def _apply_consolidation_result(self, mode: str):
        """应用记忆整理结果 - 多模式支持"""
        if mode == "light":
            self._light_consolidation()
        elif mode == "deep":
            self._deep_consolidation()

    def _light_consolidation(self):
        """浅度整理"""

        # 轻度清理工作记忆
        self.working_memory.recent_events = self.working_memory.recent_events[-25:]

    def _deep_consolidation(self):
        """深度整理 - 包含存储逻辑"""

        # 深度清理工作记忆
        self.working_memory.recent_events = self.working_memory.recent_events[-10:]
        self.working_memory.cognitive_load = 0.1

    def _cleanup_expired_memories(self):
        """清理过期记忆"""
        """current_time = time.time()

        # 清理工作记忆中的超时事件（30分钟前）
        self.working_memory.recent_events = [
            event
            for event in self.working_memory.recent_events
            if current_time - event.timestamp < 1800  # 30分钟
        ]

        # 清理低重要性的缓存记忆
        if len(self.episodic_memory.memory_episodes) > 150:
            self.episodic_memory.memory_episodes = [
                episode
                for episode in self.episodic_memory.memory_episodes
                if episode.importance > 0.3  # 保留中等重要性以上
            ][
                :100
            ]  # 限制缓存大小"""

    def _update_system_state(self):
        """更新系统状态 - 包含存储检查"""
        current_time = time.time()

        # 清理过期记忆
        if current_time - self.stats.get("last_cleanup_time", 0) > 60:
            self._cleanup_expired_memories()
            self.stats["last_cleanup_time"] = current_time

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "is_running": self.is_running,
            "cognitive_load": self.working_memory.cognitive_load,
            "working_memory_usage": len(self.working_memory.recent_events),
            # "episodic_memory_usage": len(self.episodic_memory.episodic_memories),
            "processing_stats": self.stats,
        }
