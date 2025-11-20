from dataclasses import dataclass
from typing import Literal


@dataclass
class CognitiveCoreConfig:
    # 超过多少条历史记忆就使用专门的回想任务处理
    episodic_memories_direct_threshold: int = 5
    # 一次循环最多执行多少次事件
    max_processed_count_on_loop: int = 10
    # 最大联想记忆数，超过时会过滤部分
    max_associative_recall_items: int = 30
    # 联想记忆超长时的截断模式，默认保留后面的
    associative_recall_truncate_mode: Literal["first", "last"] = "last"
    # 苏醒阶段回顾几天的记忆
    morning_max_days_back: int = 14
    # 苏醒阶段只取最近几天的记忆
    morning_max_back_days: int = 2
    # 苏醒阶段回顾记忆重要性过滤
    morning_memory_min_importance: int = 50
