import os
import json
from typing import List, Dict
from datetime import datetime
from lll_simple_ai_shared import EpisodicMemoriesModels
from ..core.plugin_interfaces import MemoryManagerPlugin


class CognitiveCorePluginDefaultMemoryManager(MemoryManagerPlugin):
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
            time_index = self.load_time_index()
            relevant_dates = []
            for date_str, meta in time_index["indexed_dates"].items():
                current_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # 时间范围过滤
                if not (start_date <= current_date <= end_date):
                    continue

                # 重要性范围过滤
                if meta["importance_range"][1] < importance_min:
                    continue

                # 关键词预过滤（如果有的话）
                if keywords and not any(
                    kw in meta.get("keywords", []) for kw in keywords
                ):
                    continue

                # 联想词预过滤（如果有的话）
                if associations and not any(
                    assoc in meta.get("associations", []) for assoc in associations
                ):
                    continue

                relevant_dates.append(date_str)

            # 加载相关日期的文件进行精细筛选
            results: List[EpisodicMemoriesModels] = []
            for date_str in relevant_dates:
                daily_memories = self.load_daily_memories(date_str)

                for memory in daily_memories:
                    # 重要性过滤
                    if memory.importance < importance_min:
                        continue

                    # 关键词精确匹配
                    keyword_match = True
                    if keywords:
                        keyword_match = any(kw in memory.keywords for kw in keywords)
                        if not keyword_match:
                            continue

                    # 联想词匹配
                    association_match = True
                    if associations:
                        association_match = any(
                            assoc in memory.associations for assoc in associations
                        )
                        if not association_match:
                            continue

                    # 所有条件都满足
                    results.append(memory)

            return results
        except Exception as e:
            print(f"查询记忆错误: {e}")
            return []

    def save_episodic_memories(self, episodic_memories: List[EpisodicMemoriesModels]):
        """
        记忆整理方法 - 处理多天记忆的整理和索引更新
        """
        if not episodic_memories:
            return

        # 按日期分组记忆
        memories_by_date = self.group_memories_by_date(episodic_memories)

        # 处理每个日期的记忆文件
        for date_str, memories in memories_by_date.items():
            self.process_single_date_memories(date_str, memories)

        # 更新全局索引
        self.update_global_indexes(memories_by_date)

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

    def process_single_date_memories(
        self, date_str: str, new_memories: List[EpisodicMemoriesModels]
    ):
        """处理单个日期的记忆文件"""
        # 构造文件名
        filename = f"memory_{date_str}.jsonl"
        filepath = os.path.join("memory/daily", filename)

        # 读取现有记忆（如果文件存在）
        existing_memories = []
        if os.path.exists(filepath):
            existing_memories = self.load_daily_memories(date_str)

        # 合并新旧记忆（基于ID去重）
        all_memories = self.merge_memories(existing_memories, new_memories)

        # 保存到文件
        self.save_memories_to_file(filepath, all_memories)

    def update_global_indexes(
        self, memories_by_date: Dict[str, List[EpisodicMemoriesModels]]
    ):
        """更新所有全局索引"""
        # 读取现有索引
        time_index = self.load_time_index()
        keyword_index = self.load_keyword_index()
        association_index = self.load_association_index()

        # 更新每个日期的索引
        for date_str, memories in memories_by_date.items():
            self.update_date_in_indexes(
                date_str, memories, time_index, keyword_index, association_index
            )

        # 保存更新后的索引
        self.save_time_index(time_index)
        self.save_keyword_index(keyword_index)
        self.save_association_index(association_index)

    def update_date_in_indexes(
        self,
        date_str: str,
        memories: List[EpisodicMemoriesModels],
        time_index: Dict,
        keyword_index: Dict,
        association_index: Dict,
    ):
        """更新单个日期在所有索引中的信息"""

        # 1. 更新时间索引
        if date_str not in time_index["indexed_dates"]:
            time_index["indexed_dates"][date_str] = {
                "memory_count": 0,
                "keywords": set(),
                "associations": set(),
            }

        date_meta = time_index["indexed_dates"][date_str]
        date_meta["memory_count"] = len(memories)

        # 2. 更新关键词和联想词索引
        for memory in memories:
            if memory.keywords is not None:
                # 更新关键词索引
                for keyword in memory.keywords:
                    if keyword not in keyword_index:
                        keyword_index[keyword] = set()
                    keyword_index[keyword].add(memory.id)
                    date_meta["keywords"].add(keyword)

            if memory.associations is not None:
                # 更新联想词索引
                for association in memory.associations:
                    if association not in association_index:
                        association_index[association] = set()
                    association_index[association].add(memory.id)
                    date_meta["associations"].add(association)

        # 转换set为list以便JSON序列化
        date_meta["keywords"] = list(date_meta["keywords"])
        date_meta["associations"] = list(date_meta["associations"])

    def merge_memories(
        self, existing: List[EpisodicMemoriesModels], new: List[EpisodicMemoriesModels]
    ) -> List[EpisodicMemoriesModels]:
        """合并记忆，基于ID去重，新的覆盖旧的"""
        memory_dict = {}

        # 先添加现有记忆
        for memory in existing:
            memory_dict[memory.id] = memory

        # 用新记忆覆盖（如果ID相同）
        for memory in new:
            memory_dict[memory.id] = memory

        return list(memory_dict.values())

    def save_memories_to_file(
        self, filepath: str, memories: List[EpisodicMemoriesModels]
    ):
        """保存记忆到JSONL文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            for memory in memories:
                # 转换为字典并确保timestamp是字符串
                memory_dict = memory.dict()
                memory_dict["timestamp"] = memory.timestamp.isoformat()
                f.write(json.dumps(memory_dict, ensure_ascii=False) + "\n")

    def load_daily_memories(self, date_str: str) -> List[EpisodicMemoriesModels]:
        """加载单个日期的记忆文件"""
        filename = f"memory_{date_str}.jsonl"
        filepath = os.path.join("memory/daily", filename)

        if not os.path.exists(filepath):
            return []

        memories = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                # 转换字符串timestamp回datetime对象
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                memories.append(EpisodicMemoriesModels(**data))

        return memories

    def load_time_index(self) -> Dict:
        return self.load_generic_index("memory/index/time_index.json")

    def save_time_index(self, time_index: Dict):
        self.save_generic_index("memory/index/time_index.json", time_index)

    def load_keyword_index(self) -> Dict:
        """加载关键词索引文件"""
        return self.load_generic_index("memory/index/keyword_index.json")

    def save_keyword_index(self, keyword_index: Dict):
        """保存关键词索引文件"""
        self.save_generic_index("memory/index/keyword_index.json", keyword_index)

    def load_association_index(self) -> Dict:
        """加载联想词索引文件"""
        return self.load_generic_index("memory/index/association_index.json")

    def save_association_index(self, association_index: Dict):
        """保存联想词索引文件"""
        self.save_generic_index(
            "memory/index/association_index.json", association_index
        )

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

    def load_generic_index(self, filepath: str) -> Dict:
        """通用索引加载函数"""
        if not os.path.exists(filepath):
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            # 将列表转换回set以便操作
            processed_index = {}
            for key, id_list in index_data.items():
                processed_index[key] = set(id_list)

            return processed_index

        except (json.JSONDecodeError, KeyError) as e:
            print(f"加载索引文件 {filepath} 失败: {e}")
            return {}

    def save_generic_index(self, filepath: str, index_data: Dict):
        """通用索引保存函数"""
        try:
            # 将set转换为list以便JSON序列化
            serializable_index = {}
            for key, id_set in index_data.items():
                serializable_index[key] = list(id_set)

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 保存文件
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"保存索引文件 {filepath} 失败: {e}")
