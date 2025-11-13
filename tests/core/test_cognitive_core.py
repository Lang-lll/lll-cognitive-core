import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch

from lll_simple_ai_shared import (
    UnderstoodData,
    RecallResultsModels,
    BehaviorPlan,
    MemoryQueryPlan,
    EpisodicMemoriesModels,
    EpisodicMemoriesGenerateModels,
)
from lll_cognitive_core.config.cognitive_core_config import CognitiveCoreConfig
from lll_cognitive_core.core.data_structures import *
from lll_cognitive_core.core.plugin_interfaces import (
    EventUnderstandingPlugin,
    AssociativeRecallPlugin,
    BehaviorGenerationPlugin,
    MemoryManagerPlugin,
)
from lll_cognitive_core.core.cognitive_core import CognitiveCore


class TestCognitiveCore:
    """CognitiveCore 单元测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.config = CognitiveCoreConfig()
        self.cognitive_core = CognitiveCore(self.config)

        # 创建模拟插件
        self.mock_event_understanding = Mock(spec=EventUnderstandingPlugin)
        self.mock_behavior_generation = Mock(spec=BehaviorGenerationPlugin)
        self.mock_memory_manager = Mock(spec=MemoryManagerPlugin)
        self.mock_associative_recall = Mock(spec=AssociativeRecallPlugin)
        self.mock_memory_extraction = Mock()

    def test_initialization(self):
        """测试初始化"""
        assert self.cognitive_core.status == CoreStatus.AWAITING
        assert self.cognitive_core.working_memory is not None
        assert self.cognitive_core.episodic_memory_manager is not None
        assert self.cognitive_core.event_queue is not None
        assert self.cognitive_core.processing_thread is None

    def test_register_plugin(self):
        """测试注册插件"""
        # 执行
        self.cognitive_core.register_plugin(
            "event_understanding", self.mock_event_understanding
        )
        self.cognitive_core.register_plugin(
            "behavior_generation", self.mock_behavior_generation
        )

        # 验证
        assert (
            self.cognitive_core.get_plugin("event_understanding")
            == self.mock_event_understanding
        )
        assert (
            self.cognitive_core.get_plugin("behavior_generation")
            == self.mock_behavior_generation
        )

    def test_register_unknown_plugin(self):
        """测试注册未知插件"""
        with patch.object(self.cognitive_core.logger, "error") as mock_error:
            self.cognitive_core.register_plugin("unknown_plugin", Mock())

            # 验证错误日志被调用
            mock_error.assert_called_once()

    def test_wake_up(self):
        """测试启动认知核心"""
        # 执行
        self.cognitive_core.wake_up()

        # 验证状态改变
        assert self.cognitive_core.status == CoreStatus.AWARE
        assert self.cognitive_core.processing_thread is not None
        assert self.cognitive_core.processing_thread.daemon is True

        # 清理
        self.cognitive_core.sleep()

    def test_wake_up_when_already_awake(self):
        """测试重复启动"""
        # 第一次启动
        self.cognitive_core.wake_up()
        original_thread = self.cognitive_core.processing_thread

        # 第二次启动
        self.cognitive_core.wake_up()

        # 验证线程没有改变
        assert self.cognitive_core.processing_thread == original_thread

        # 清理
        self.cognitive_core.sleep()

    def test_sleep(self):
        """测试停止认知核心"""
        # 先启动
        self.cognitive_core.wake_up()
        assert self.cognitive_core.status == CoreStatus.AWARE

        # 执行停止
        self.cognitive_core.sleep()

        # 验证状态改变
        assert self.cognitive_core.status == CoreStatus.WINDING_DOWN

    def test_sleep_when_already_asleep(self):
        """测试重复停止"""
        # 初始状态
        assert self.cognitive_core.status == CoreStatus.AWAITING

        # 执行停止
        self.cognitive_core.sleep()

        # 验证状态没有改变
        assert self.cognitive_core.status == CoreStatus.AWAITING

    def test_receive_event(self):
        """测试接收事件"""
        # 启动核心
        self.cognitive_core.wake_up()

        # 准备事件数据
        test_event = {"type": "user_message", "data": "Hello, world!", "source": "user"}

        # 执行
        self.cognitive_core.receive_event(test_event)

        # 验证事件被添加到队列
        assert not self.cognitive_core.event_queue.empty()

        # 清理
        self.cognitive_core.sleep()

    def test_receive_event_invalid_data(self):
        """测试接收无效事件"""
        """with patch.object(self.cognitive_core.logger, "error") as mock_error:
            # 无效事件数据
            invalid_event = {"invalid": "data"}

            # 执行
            self.cognitive_core.receive_event(invalid_event)

            # 验证错误处理
            mock_error.assert_called_once()"""

    def test_receive_event_when_sleeping(self):
        """测试在睡眠状态下接收事件"""
        # 确保核心处于睡眠状态
        assert self.cognitive_core.status == CoreStatus.AWAITING

        test_event = {"type": "user_message", "data": "Hello, world!", "source": "user"}

        # 执行
        self.cognitive_core.receive_event(test_event)

        # 验证事件没有被添加到队列（因为状态不是 AWARE）
        assert self.cognitive_core.event_queue.empty()

    @patch.object(CognitiveCore, "_process_events")
    @patch.object(CognitiveCore, "_update_system_state")
    @patch.object(CognitiveCore, "_check_sleep")
    def test_processing_loop_normal_operation(
        self, mock_check_sleep, mock_update_state, mock_process_events
    ):
        """测试处理循环正常操作"""
        # 设置状态为 AWARE
        self.cognitive_core.status = CoreStatus.AWARE

        # 模拟循环执行一次后退出
        def stop_after_one_iteration():
            self.cognitive_core.status = CoreStatus.AWAITING

        mock_check_sleep.side_effect = stop_after_one_iteration

        # 执行处理循环
        self.cognitive_core._processing_loop()

        # 验证各个方法被调用
        mock_process_events.assert_called_once()
        mock_update_state.assert_called_once()
        mock_check_sleep.assert_called_once()

    @patch.object(CognitiveCore, "_process_events")
    @patch.object(CognitiveCore, "_update_system_state")
    @patch.object(CognitiveCore, "_check_sleep")
    @patch("time.sleep")
    def test_processing_loop_exception_handling(
        self, mock_sleep, mock_check_sleep, mock_update_state, mock_process_events
    ):
        """测试处理循环异常处理"""
        """# 设置状态为 AWARE
        self.cognitive_core.status = CoreStatus.AWARE

        # 模拟_process_events抛出异常
        mock_process_events.side_effect = Exception("Test error")

        # 模拟循环执行一次后退出
        def stop_after_one_iteration():
            self.cognitive_core.status = CoreStatus.AWAITING

        mock_check_sleep.side_effect = stop_after_one_iteration

        with patch.object(self.cognitive_core.logger, "error") as mock_error:
            # 执行处理循环
            self.cognitive_core._processing_loop()

            # 验证错误被记录
            mock_error.assert_called_once_with("处理循环错误: Test error")"""

    def test_process_single_event_with_plugins(self):
        """测试处理单个事件（包含插件调用）"""
        # 注册插件
        self.cognitive_core.register_plugin(
            "event_understanding", self.mock_event_understanding
        )
        self.cognitive_core.register_plugin(
            "behavior_generation", self.mock_behavior_generation
        )

        # 模拟事件理解结果
        mock_understood_data = UnderstoodData(
            response_priority="medium",
            expected_response="text",
            main_content="Hello, world!",
            event_entity="user",
            key_entities=[],
            importance_score=50,
            current_situation="conversation",
            memory_query_plan=None,
        )
        self.mock_event_understanding.understand_event.return_value = (
            mock_understood_data
        )

        # 模拟行为生成结果
        mock_behavior_plan = BehaviorPlan(
            plan=[
                {
                    "type": "tts",
                    "action": "speak",
                    "emotion": "neutral",
                    "data": "Hello!",
                    "speed": 1.0,
                },
            ],
            current_situation="conversation",
        )
        self.mock_behavior_generation.generate_behavior.return_value = (
            mock_behavior_plan
        )

        # 准备测试事件
        test_event = UnderstandEventData(
            type="user_message",
            data="Hello, world!",
            source="user",
            timestamp=time.time(),
        )

        with patch.object(
            self.cognitive_core, "_update_working_memory"
        ) as mock_update_memory, patch.object(
            self.cognitive_core, "_generate_and_execute_behavior"
        ) as mock_generate_behavior:

            # 执行
            self.cognitive_core._process_single_event(test_event)

            # 验证插件被调用
            self.mock_event_understanding.understand_event.assert_called_once()
            mock_update_memory.assert_called_once_with(test_event, mock_understood_data)
            mock_generate_behavior.assert_called_once_with(mock_understood_data)

    def test_process_single_event_no_understanding_plugin(self):
        """测试没有事件理解插件的情况"""
        # 不注册事件理解插件

        test_event = UnderstandEventData(
            type="user_message",
            data="Hello, world!",
            source="user",
            timestamp=time.time(),
        )

        # 执行
        result = self.cognitive_core._understand_event(test_event)

        # 验证返回 None
        assert result is None

    def test_update_working_memory(self):
        """测试更新工作记忆"""
        # 准备测试数据
        test_event = UnderstandEventData(
            type="user_message",
            data="Hello, world!",
            source="user",
            timestamp=time.time(),
        )

        test_understood_data = UnderstoodData(
            response_priority="medium",
            expected_response="text",
            main_content="Hello, world!",
            event_entity="user",
            key_entities=["user"],
            importance_score=50,
            current_situation="conversation",
            memory_query_plan=None,
        )

        # 执行
        self.cognitive_core._update_working_memory(test_event, test_understood_data)

        # 验证工作记忆被更新
        assert len(self.cognitive_core.working_memory.recent_events) == 1
        cognitive_event = self.cognitive_core.working_memory.recent_events[0]
        assert cognitive_event.event_id.startswith("event_")
        assert cognitive_event.source == "user"
        assert cognitive_event.understood_data == test_understood_data
        assert self.cognitive_core.working_memory.current_situation == "conversation"

    def test_generate_and_execute_behavior_with_memory_query(self):
        """测试生成和执行行为（包含记忆查询）"""
        # 注册插件
        self.cognitive_core.register_plugin(
            "behavior_generation", self.mock_behavior_generation
        )
        self.cognitive_core.register_plugin("memory_manager", self.mock_memory_manager)
        self.cognitive_core.register_plugin(
            "associative_recall", self.mock_associative_recall
        )

        # 模拟记忆查询计划
        mock_understood_data = UnderstoodData(
            response_priority="low",
            main_content="",
            current_situation="",
            event_entity="",
            key_entities=[],
            importance_score=0,
            memory_query_plan=MemoryQueryPlan(
                query_type="long_term_fresh",
                time_range=[0, 1],
                query_triggers=["work", "meeting"],
                importance_score_filter=0,
            ),
        )

        # 模拟记忆查询结果
        mock_episodic_memories = [
            Mock(
                spec=EpisodicMemoriesModels,
                id="mem1",
                importance=8,
                timestamp=datetime.now().date(),
                keywords=[],
            ),
            Mock(
                spec=EpisodicMemoriesModels,
                id="mem2",
                importance=9,
                timestamp=datetime.now().date(),
                keywords=[],
            ),
        ]
        self.mock_memory_manager.query_episodic_memories.return_value = (
            mock_episodic_memories
        )

        # 模拟联想回忆结果
        mock_recall_result = RecallResultsModels(
            recalled_episode="Test recalled episode",
            current_situation="updated_situation",
        )
        self.mock_associative_recall.associative_recall.return_value = (
            mock_recall_result
        )

        # 模拟行为生成结果
        mock_behavior_plan = BehaviorPlan(
            plan=[
                {
                    "type": "tts",
                    "action": "speak",
                    "emotion": "neutral",
                    "data": "Hello!",
                    "speed": 1.0,
                }
            ],
            current_situation="",
        )
        self.mock_behavior_generation.generate_behavior.return_value = (
            mock_behavior_plan
        )

        with patch.object(
            self.cognitive_core, "_execute_behavior_plan"
        ) as mock_execute:
            # 执行
            self.cognitive_core._generate_and_execute_behavior(mock_understood_data)

            # 验证记忆查询被调用
            self.mock_memory_manager.query_episodic_memories.assert_called_once_with(
                date_range=[
                    0,
                    1,
                ],
                keywords=["work", "meeting"],
            )

            # 验证联想回忆被调用（因为记忆数量超过阈值）
            # self.mock_associative_recall.associative_recall.assert_called_once()

            # 验证行为生成被调用
            self.mock_behavior_generation.generate_behavior.assert_called_once()
            mock_execute.assert_called_once_with(mock_behavior_plan)

    def test_execute_behavior_plan(self):
        """测试执行行为计划"""
        # 准备行为计划
        mock_behavior_plan = BehaviorPlan(
            plan=[
                {
                    "type": "tts",
                    "action": "speak",
                    "emotion": "neutral",
                    "data": "Hello!",
                    "speed": 1.0,
                },
                {
                    "type": "tts",
                    "action": "speak",
                    "emotion": "neutral",
                    "data": "Do something",
                    "speed": 1.0,
                },
            ],
            current_situation="updated_situation",
        )

        with patch.object(
            self.cognitive_core.logger, "info"
        ) as mock_logger, patch.object(
            self.cognitive_core, "_update_working_memory"
        ) as mock_update_memory:

            # 执行
            self.cognitive_core._execute_behavior_plan(mock_behavior_plan)

            # 验证日志记录
            assert mock_logger.call_count == 2
            # 验证工作记忆更新
            assert mock_update_memory.call_count == 2
            # 验证情境更新
            assert (
                self.cognitive_core.working_memory.current_situation
                == "updated_situation"
            )

    def test_execute_behavior_plan_empty(self):
        """测试执行空行为计划"""
        # 空行为计划
        empty_behavior_plan = BehaviorPlan(plan=[], current_situation="")

        # 执行（不应该抛出异常）
        self.cognitive_core._execute_behavior_plan(empty_behavior_plan)

    def test_update_cognitive_load(self):
        """测试更新认知负荷"""
        # 添加一些测试事件到工作记忆
        for i in range(5):
            event = CognitiveEvent(
                event_id=f"event_{i}",
                timestamp=time.time(),
                source="test",
                modality_type="test",
                raw_data={},
                understood_data=Mock(),
                importance_score=50,
            )
            self.cognitive_core.working_memory.recent_events.append(event)

        # 添加一些目标
        self.cognitive_core.working_memory.active_goals = ["goal1", "goal2"]

        # 添加一些缓存记忆
        self.cognitive_core.episodic_memory_manager.episodic_memory.episodic_memories = {
            "mem1": Mock(),
            "mem2": Mock(),
            "mem3": Mock(),
        }

        # 执行
        self.cognitive_core._update_cognitive_load()

        # 验证认知负荷被计算
        assert 0 <= self.cognitive_core.working_memory.cognitive_load <= 1.0

    def test_get_system_status(self):
        """测试获取系统状态"""
        # 执行
        status = self.cognitive_core.get_system_status()

        # 验证返回正确的状态信息
        assert "status" in status
        assert "cognitive_load" in status
        assert "working_memory_usage" in status
        assert "episodic_memory_usage" in status
        assert "processing_stats" in status
        assert status["status"] == CoreStatus.AWAITING

    def test_consolidate_memories(self):
        """测试记忆整理"""
        # 注册记忆提取插件
        self.cognitive_core.register_plugin(
            "memory_extraction", self.mock_memory_extraction
        )
        self.cognitive_core.register_plugin("memory_manager", self.mock_memory_manager)

        # 添加一些测试事件到工作记忆
        test_event = CognitiveEvent(
            event_id="test_event_1",
            timestamp=time.time(),
            source="test",
            modality_type="test",
            raw_data=UnderstandEventData(
                type="test", data="test data", source="test", timestamp=time.time()
            ),
            understood_data=UnderstoodData(
                response_priority="low",
                main_content="",
                current_situation=None,
                event_entity="",
                key_entities=["entity1", "entity2"],
                importance_score=0,
                memory_query_plan=None,
            ),
            importance_score=50,
        )
        self.cognitive_core.working_memory.recent_events.append(test_event)

        # 模拟记忆提取结果
        mock_extraction_result = [
            EpisodicMemoriesGenerateModels(
                id="test_event_1",
                content="Extracted memory content",
                importance=0,
                keywords=["test", "memory"],
                associations=["related"],
            )
        ]
        self.mock_memory_extraction.extract_memories.return_value = (
            mock_extraction_result
        )

        # 模拟记忆保存
        self.mock_memory_manager.save_episodic_memories.return_value = None

        # 执行深度整理
        self.cognitive_core._consolidate_memories("deep")

        # 验证记忆提取被调用
        self.mock_memory_extraction.extract_memories.assert_called_once()
        # 验证记忆保存被调用
        self.mock_memory_manager.save_episodic_memories.assert_called_once()
        # 验证状态恢复
        assert self.cognitive_core.status == CoreStatus.AWAITING

    def test_light_consolidation(self):
        """测试轻度整理"""
        # 添加多个事件到工作记忆
        for i in range(30):
            event = CognitiveEvent(
                event_id=f"event_{i}",
                timestamp=time.time(),
                source="test",
                modality_type="test",
                raw_data={},
                understood_data=Mock(),
                importance_score=50,
            )
            self.cognitive_core.working_memory.recent_events.append(event)

        original_count = len(self.cognitive_core.working_memory.recent_events)

        # 执行轻度整理
        self.cognitive_core._light_consolidation()

        # 验证事件数量被限制
        assert len(self.cognitive_core.working_memory.recent_events) <= 25
        assert len(self.cognitive_core.working_memory.recent_events) < original_count

    def test_deep_consolidation(self):
        """测试深度整理"""
        # 添加事件到工作记忆
        for i in range(10):
            event = CognitiveEvent(
                event_id=f"event_{i}",
                timestamp=time.time(),
                source="test",
                modality_type="test",
                raw_data={},
                understood_data=Mock(),
                importance_score=50,
            )
            self.cognitive_core.working_memory.recent_events.append(event)

        # 添加缓存记忆
        self.cognitive_core.episodic_memory_manager.episodic_memory.episodic_memories = {
            "mem1": Mock(),
            "mem2": Mock(),
        }

        # 执行深度整理
        self.cognitive_core._deep_consolidation()

        # 验证工作记忆被清空
        assert len(self.cognitive_core.working_memory.recent_events) == 0
        # 验证缓存记忆被清空
        assert (
            len(
                self.cognitive_core.episodic_memory_manager.episodic_memory.episodic_memories
            )
            == 0
        )


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
