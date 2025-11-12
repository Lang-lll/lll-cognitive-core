from typing import Optional, List
from lll_simple_ai_shared import (
    UnderstoodData,
    RecallResultsModels,
    BehaviorPlan,
    EpisodicMemoriesGenerateModels,
    EpisodicMemoriesModels,
)
from .data_structures import *


# 默认插件实现基类
class EventUnderstandingPlugin:
    def understand_event(self, raw_event: UnderstandEventInput) -> Dict:
        # 事件理解
        return UnderstoodData


class AssociativeRecallPlugin:
    def associative_recall(self, recall_request: AssociativeRecallInput) -> str:
        return RecallResultsModels


class BehaviorGenerationPlugin:
    def generate_behavior(self, cognitive_state: GenerateBehaviorInput) -> Dict:
        return BehaviorPlan


class BehaviorExecutionPlugin:
    def execute_behavior_plan(self, action: Any):
        pass


class MemoryExtractionPlugin:
    """记忆提取插件基类 - 只定义与CognitiveCore交互的接口"""

    def extract_memories(self, data: ExtractMemoriesInput) -> Dict:
        """从工作记忆中提取结构化记忆 - 核心接口方法"""
        return List[EpisodicMemoriesGenerateModels]


class MemoryManagerPlugin:
    """记忆管理插件基类"""

    def save_episodic_memories(
        self, episodic_memories: List[EpisodicMemoriesModels]
    ) -> bool:
        """
        保存情景记忆到持久化存储

        入参:
            episodic_memories: 要保存的记忆片段列表

        出参: 是否保存成功
        """
        raise NotImplementedError("子类必须实现save_episodic_memories方法")

    def query_episodic_memories(
        self,
        date_range: Optional[List[int | str]],
        importance_min: Optional[int],
        keywords: Optional[List[str]],
        associations: Optional[List[str]],
    ) -> List[EpisodicMemoriesModels]:
        """
        从存储加载情景记忆

        入参:
            date_range: 时间范围，YYYY-MM-DD格式
            importance_min: 重要程度过滤
            keywords: 关键词
            associations: 联想词
        """
        raise NotImplementedError("子类必须实现query_episodic_memories方法")
