from dataclasses import dataclass


@dataclass
class CognitiveCoreConfig:
    # 超过多少条历史记忆就使用专门的回想任务处理
    episodic_memories_direct_threshold: int = 5
    # 一次循环最多执行多少次事件
    max_processed_count_on_loop: int = 10
