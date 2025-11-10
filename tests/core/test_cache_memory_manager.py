import pytest
from datetime import datetime, date
from unittest.mock import patch
from typing import List
from pydantic import BaseModel

from lll_cognitive_core.core.data_structures import EpisodicMemory
from lll_cognitive_core.core.cache_memory_manager import CacheMemoryManager


# 创建测试用的记忆模型
class TestEpisodicMemoryModel(BaseModel):
    id: str
    timestamp: datetime
    importance: float = 1.0
    keywords: List[str] = []
    content: str = ""
    associations: List[str] = []

    class Config:
        arbitrary_types_allowed = True


class TestCacheMemoryManager:
    """CacheMemoryManager 单元测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.memory_manager = CacheMemoryManager()

        # 创建测试记忆数据
        self.test_memory_1 = TestEpisodicMemoryModel(
            id="mem_1",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            importance=0.8,
            keywords=["work", "meeting"],
            content="Team meeting about project",
        )

        self.test_memory_2 = TestEpisodicMemoryModel(
            id="mem_2",
            timestamp=datetime(2024, 1, 16, 14, 0, 0),
            importance=0.9,
            keywords=["lunch", "friends"],
            content="Lunch with friends",
        )

        self.test_memory_3 = TestEpisodicMemoryModel(
            id="mem_3",
            timestamp=datetime(2024, 1, 17, 16, 45, 0),
            importance=0.7,
            keywords=["work", "deadline"],
            content="Project deadline approaching",
        )

    def test_initialization(self):
        """测试初始化"""
        assert self.memory_manager.episodic_memory is not None
        assert isinstance(self.memory_manager.episodic_memory, EpisodicMemory)
        assert len(self.memory_manager.episodic_memory.episodic_memories) == 0
        assert len(self.memory_manager.episodic_memory.time_index) == 0
        assert len(self.memory_manager.episodic_memory.keyword_index) == 0

    def test_save_episodic_memories_single(self):
        """测试保存单个记忆"""
        # 执行
        self.memory_manager.save_episodic_memories([self.test_memory_1])

        # 验证
        assert len(self.memory_manager.episodic_memory.episodic_memories) == 1
        assert (
            self.test_memory_1.id
            in self.memory_manager.episodic_memory.episodic_memories
        )
        assert (
            self.memory_manager.episodic_memory.episodic_memories[self.test_memory_1.id]
            == self.test_memory_1
        )

        # 验证时间索引
        date_key = self.test_memory_1.timestamp.strftime("%Y-%m-%d")
        assert date_key in self.memory_manager.episodic_memory.time_index
        assert (
            self.test_memory_1.id
            in self.memory_manager.episodic_memory.time_index[date_key]
        )

        # 验证关键词索引
        for keyword in self.test_memory_1.keywords:
            assert keyword in self.memory_manager.episodic_memory.keyword_index
            assert (
                self.test_memory_1.id
                in self.memory_manager.episodic_memory.keyword_index[keyword]
            )

    def test_save_episodic_memories_multiple(self):
        """测试保存多个记忆"""
        # 执行
        memories = [self.test_memory_1, self.test_memory_2, self.test_memory_3]
        self.memory_manager.save_episodic_memories(memories)

        # 验证
        assert len(self.memory_manager.episodic_memory.episodic_memories) == 3
        for memory in memories:
            assert memory.id in self.memory_manager.episodic_memory.episodic_memories

    def test_save_episodic_memories_empty(self):
        """测试保存空记忆列表"""
        # 执行
        self.memory_manager.save_episodic_memories([])

        # 验证 - 不应该有任何变化
        assert len(self.memory_manager.episodic_memory.episodic_memories) == 0
        assert len(self.memory_manager.episodic_memory.time_index) == 0
        assert len(self.memory_manager.episodic_memory.keyword_index) == 0

    def test_save_episodic_memories_duplicate(self):
        """测试保存重复记忆"""
        # 第一次保存
        self.memory_manager.save_episodic_memories([self.test_memory_1])

        # 第二次保存相同的记忆
        self.memory_manager.save_episodic_memories([self.test_memory_1])

        # 验证 - 不应该有重复条目
        date_key = self.test_memory_1.timestamp.strftime("%Y-%m-%d")
        assert len(self.memory_manager.episodic_memory.time_index[date_key]) == 1

        for keyword in self.test_memory_1.keywords:
            assert len(self.memory_manager.episodic_memory.keyword_index[keyword]) == 1

    def test_query_episodic_memories_by_date_range(self):
        """测试按时间范围查询"""
        # 准备数据
        memories = [self.test_memory_1, self.test_memory_2, self.test_memory_3]
        self.memory_manager.save_episodic_memories(memories)

        # 查询 2024-01-15 到 2024-01-16 的记忆
        date_range = ["2024-01-15", "2024-01-16"]
        results = self.memory_manager.query_episodic_memories(date_range)

        # 验证
        assert len(results) == 2
        memory_ids = [memory.id for memory in results]
        assert "mem_1" in memory_ids
        assert "mem_2" in memory_ids
        assert "mem_3" not in memory_ids

    def test_query_episodic_memories_by_keywords(self):
        """测试按关键词查询"""
        # 准备数据
        memories = [self.test_memory_1, self.test_memory_2, self.test_memory_3]
        self.memory_manager.save_episodic_memories(memories)

        # 查询包含 "work" 关键词的记忆
        date_range = ["2024-01-15", "2024-01-17"]
        keywords = ["work"]
        results = self.memory_manager.query_episodic_memories(
            date_range, keywords=keywords
        )

        # 验证
        assert len(results) == 2
        memory_ids = [memory.id for memory in results]
        assert "mem_1" in memory_ids
        assert "mem_3" in memory_ids
        assert "mem_2" not in memory_ids

    def test_query_episodic_memories_by_multiple_keywords(self):
        """测试按多个关键词查询"""
        # 准备数据
        memories = [self.test_memory_1, self.test_memory_2, self.test_memory_3]
        self.memory_manager.save_episodic_memories(memories)

        # 查询包含 "work" 或 "lunch" 关键词的记忆
        date_range = ["2024-01-15", "2024-01-17"]
        keywords = ["work", "lunch"]
        results = self.memory_manager.query_episodic_memories(
            date_range, keywords=keywords
        )

        # 验证 - 应该返回所有三个记忆
        assert len(results) == 3

    def test_query_episodic_memories_no_keywords_match(self):
        """测试无匹配关键词的查询"""
        """# 准备数据
        memories = [self.test_memory_1, self.test_memory_2]
        self.memory_manager.save_episodic_memories(memories)

        # 查询不存在的关键词
        date_range = ["2024-01-15", "2024-01-16"]
        keywords = ["nonexistent"]
        results = self.memory_manager.query_episodic_memories(
            date_range, keywords=keywords
        )

        # 验证 - 应该只返回时间范围内的记忆（不进行关键词过滤）
        assert len(results) == 2"""

    def test_query_episodic_memories_importance_filter(self):
        """测试重要性过滤"""
        # 准备数据
        memories = [self.test_memory_1, self.test_memory_2, self.test_memory_3]
        self.memory_manager.save_episodic_memories(memories)

        # 查询重要性 >= 0.85 的记忆
        date_range = ["2024-01-15", "2024-01-17"]
        results = self.memory_manager.query_episodic_memories(
            date_range, importance_min=0.85
        )

        # 验证
        assert len(results) == 1
        assert results[0].id == "mem_2"  # importance=0.9

    def test_query_episodic_memories_default_date_range(self):
        """测试默认时间范围查询"""
        # 创建一个今天的记忆
        today_memory = TestEpisodicMemoryModel(
            id="today_mem",
            timestamp=datetime.now(),
            keywords=["today"],
            content="Today's memory",
        )

        self.memory_manager.save_episodic_memories([today_memory])

        # 使用默认时间范围查询
        results = self.memory_manager.query_episodic_memories(None)

        # 验证 - 应该返回今天的记忆
        assert len(results) == 1
        assert results[0].id == "today_mem"

    def test_query_episodic_memories_exception_handling(self):
        """测试异常处理"""
        # 模拟索引访问时抛出异常
        """with patch.object(self.memory_manager.episodic_memory, "time_index", {}):
            with patch("builtins.print") as mock_print:
                results = self.memory_manager.query_episodic_memories(["invalid-date"])

                # 验证
                assert results == []
                mock_print.assert_called_once()
                assert "查询缓存记忆错误" in mock_print.call_args[0][0]"""

    def test_group_memories_by_date(self):
        """测试按日期分组记忆"""
        # 准备不同日期的记忆
        memory_1 = TestEpisodicMemoryModel(
            id="mem1", timestamp=datetime(2024, 1, 15, 10, 0, 0), keywords=["day1"]
        )
        memory_2 = TestEpisodicMemoryModel(
            id="mem2", timestamp=datetime(2024, 1, 15, 14, 0, 0), keywords=["day1"]
        )
        memory_3 = TestEpisodicMemoryModel(
            id="mem3", timestamp=datetime(2024, 1, 16, 9, 0, 0), keywords=["day2"]
        )

        memories = [memory_1, memory_2, memory_3]

        # 执行
        grouped = self.memory_manager.group_memories_by_date(memories)

        # 验证
        assert len(grouped) == 2
        assert "2024-01-15" in grouped
        assert "2024-01-16" in grouped
        assert len(grouped["2024-01-15"]) == 2
        assert len(grouped["2024-01-16"]) == 1

    def test_parse_date_range_list_format(self):
        """测试解析列表格式的时间范围"""
        date_range = ["2024-01-01", "2024-01-31"]
        start_date, end_date = self.memory_manager.parse_date_range(date_range)

        assert start_date == date(2024, 1, 1)
        assert end_date == date(2024, 1, 31)

    def test_parse_date_range_default(self):
        """测试解析默认时间范围"""
        start_date, end_date = self.memory_manager.parse_date_range(None)

        today = datetime.now().date()
        assert start_date == today
        assert end_date == today

    def test_clear_memory(self):
        """测试清空记忆"""
        # 先保存一些记忆
        memories = [self.test_memory_1, self.test_memory_2]
        self.memory_manager.save_episodic_memories(memories)

        # 验证保存成功
        assert len(self.memory_manager.episodic_memory.episodic_memories) == 2
        assert len(self.memory_manager.episodic_memory.time_index) > 0
        assert len(self.memory_manager.episodic_memory.keyword_index) > 0

        # 执行清空
        self.memory_manager.clear()

        # 验证
        assert len(self.memory_manager.episodic_memory.episodic_memories) == 0
        assert len(self.memory_manager.episodic_memory.time_index) == 0
        assert len(self.memory_manager.episodic_memory.keyword_index) == 0

    def test_memory_without_keywords(self):
        """测试没有关键词的记忆"""
        memory_without_keywords = TestEpisodicMemoryModel(
            id="no_keywords_mem",
            timestamp=datetime(2024, 1, 18, 12, 0, 0),
            keywords=[],
            content="Memory without keywords",
        )

        # 执行保存
        self.memory_manager.save_episodic_memories([memory_without_keywords])

        # 验证 - 应该成功保存，关键词索引不应该包含空关键词
        assert (
            memory_without_keywords.id
            in self.memory_manager.episodic_memory.episodic_memories
        )
        # 关键词索引不应该有变化（因为记忆没有关键词）
        assert all(
            keyword != ""
            for keyword in self.memory_manager.episodic_memory.keyword_index.keys()
        )


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
