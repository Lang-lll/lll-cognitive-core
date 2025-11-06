from typing import List, Dict
from datetime import datetime
from lll_simple_ai_shared import EpisodicMemoriesModels
from .data_structures import EpisodicMemory
from .plugin_interfaces import MemoryManagerPlugin


class CacheMemoryManager(MemoryManagerPlugin):
    def __init__(self):
        self.episodic_memory = EpisodicMemory()

    def query_episodic_memories(
        self, date_range, importance_min=0, keywords=None, associations=None
    ) -> List[EpisodicMemoriesModels]:
        """
        多维度记忆查询
        支持时间范围、重要性过滤、关键词和联想词查询
        """
        try:
            # 解析时间范围
            start_date, end_date = self.parse_date_range(date_range)

            # 通过time_index.json快速筛选相关日期
            time_index = self.episodic_memory.time_index
            keyword_index = self.episodic_memory.keyword_index

            idList: List[str] = []
            episodic_memories: List[EpisodicMemoriesModels] = []

            for date_str, dateIdList in time_index.items():
                current_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # 时间范围过滤
                if not (start_date <= current_date <= end_date):
                    continue

                idList.extend(dateIdList)

            if keywords is not None:
                for keyword, keywordIdList in keyword_index.items():
                    if keyword in keywords:
                        idList = list(dict.fromkeys(idList + keywordIdList))

            for id in idList:
                memory = self.episodic_memory.episodic_memories.get(id, None)
                if memory is not None:
                    episodic_memories.append(memory)

            return episodic_memories
        except Exception as e:
            print(f"查询缓存记忆错误: {e}")
            return []

    def save_episodic_memories(self, episodic_memories: List[EpisodicMemoriesModels]):
        """
        记忆整理方法 - 处理多天记忆的整理和索引更新
        """
        if not episodic_memories:
            return

        # 按日期分组记忆
        memories_by_date: Dict[str, List[EpisodicMemoriesModels]] = (
            self.group_memories_by_date(episodic_memories)
        )

        # 处理每个日期的记忆文件
        for memory in episodic_memories:
            self.episodic_memory.episodic_memories[memory.id] = memory

        # 更新索引
        time_index = self.episodic_memory.time_index
        keyword_index = self.episodic_memory.keyword_index

        for date_str, memories in memories_by_date.items():
            if date_str not in time_index:
                time_index[date_str] = []

            memory_ids = [memory.id for memory in memories]
            time_index[date_str].extend(memory_ids)
            time_index[date_str] = list(dict.fromkeys(time_index[date_str]))

            for memory in memories:
                for keyword in memory.keywords:
                    if keyword not in keyword_index:
                        keyword_index[keyword] = []
                    if memory.id not in keyword_index[keyword]:
                        keyword_index[keyword].append(memory.id)

    def group_memories_by_date(
        self, memories: List[EpisodicMemoriesModels]
    ) -> Dict[str, List[EpisodicMemoriesModels]]:
        """按日期分组记忆"""
        memories_by_date: Dict[str, List[EpisodicMemoriesModels]] = {}

        for memory in memories:
            # 从timestamp提取日期
            date_str = memory.timestamp.strftime("%Y-%m-%d")

            if date_str not in memories_by_date:
                memories_by_date[date_str] = []

            memories_by_date[date_str].append(memory)

        return memories_by_date

    def parse_date_range(self, date_range):
        """解析时间范围，支持多种格式"""
        if isinstance(date_range, list) and len(date_range) == 2:
            # [起始日期, 结束日期] 格式
            start_date = datetime.strptime(date_range[0], "%Y-%m-%d").date()
            end_date = datetime.strptime(date_range[1], "%Y-%m-%d").date()
        else:
            # 默认返回今天
            start_date = datetime.now().date()
            end_date = datetime.now().date()

        return start_date, end_date
