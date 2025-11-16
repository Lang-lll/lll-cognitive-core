from typing import List, Literal, Tuple
from lll_simple_ai_shared import EpisodicMemoriesModels


class CognitiveCorePluginDefaultAssociativeRecallFilterPlugin:
    def episodic_memories_filter(
        self,
        episodic_memories: List[EpisodicMemoriesModels],
        limit: int,
        truncate_mode: Literal["first", "last"] = "last",
    ) -> Tuple[List[EpisodicMemoriesModels], bool]:
        was_truncated = len(episodic_memories) > limit

        if not was_truncated:
            return episodic_memories, False

        if truncate_mode == "first":
            # 保留前limit个
            return episodic_memories[:limit], True

        elif truncate_mode == "last":
            # 保留后limit个
            return episodic_memories[-limit:], True

        else:
            # 默认使用first模式
            return episodic_memories[-limit:], True
