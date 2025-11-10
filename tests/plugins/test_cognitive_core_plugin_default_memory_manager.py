import pytest
import json
import os
import tempfile
from datetime import datetime, date
from unittest.mock import patch, MagicMock
from typing import List
from pydantic import BaseModel

from lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager import (
    CognitiveCorePluginDefaultMemoryManager,
)


# 创建测试用的记忆模型
class TestEpisodicMemoryModel(BaseModel):
    id: str
    timestamp: datetime
    importance: float = 1.0
    keywords: List[str] = []
    content: str = ""
    associations: List[str] = []

    def dict(self):
        """转换为字典用于序列化"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "keywords": self.keywords,
            "content": self.content,
            "associations": self.associations,
        }


class TestCognitiveCorePluginDefaultMemoryManager:
    """CognitiveCorePluginDefaultMemoryManager 单元测试类"""

    def setup_method(self):
        """测试前置设置"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.memory_dir = os.path.join(self.temp_dir, "memory")
        self.daily_dir = os.path.join(self.memory_dir, "daily")
        self.index_dir = os.path.join(self.memory_dir, "index")

        os.makedirs(self.daily_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        self.memory_manager = CognitiveCorePluginDefaultMemoryManager()

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

    def teardown_method(self):
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_index_files(self):
        """创建测试索引文件"""
        # 创建时间索引
        time_index = {
            "indexed_dates": {
                "2024-01-15": {
                    "memory_count": 1,
                    "keywords": ["work", "meeting"],
                    "associations": [],
                    "importance_range": [0.8, 0.8],
                },
                "2024-01-16": {
                    "memory_count": 1,
                    "keywords": ["lunch", "friends"],
                    "associations": [],
                    "importance_range": [0.9, 0.9],
                },
                "2024-01-17": {
                    "memory_count": 1,
                    "keywords": ["work", "deadline"],
                    "associations": [],
                    "importance_range": [0.7, 0.7],
                },
            }
        }

        with open(os.path.join(self.index_dir, "time_index.json"), "w") as f:
            json.dump(time_index, f)

        # 创建关键词索引
        keyword_index = {
            "work": ["mem_1", "mem_3"],
            "meeting": ["mem_1"],
            "lunch": ["mem_2"],
            "friends": ["mem_2"],
            "deadline": ["mem_3"],
        }

        with open(os.path.join(self.index_dir, "keyword_index.json"), "w") as f:
            json.dump(keyword_index, f)

        # 创建联想词索引
        association_index = {}
        with open(os.path.join(self.index_dir, "association_index.json"), "w") as f:
            json.dump(association_index, f)

    def create_test_daily_files(self):
        """创建测试每日记忆文件"""
        # 创建 2024-01-15 的记忆文件
        with open(os.path.join(self.daily_dir, "memory_2024-01-15.jsonl"), "w") as f:
            f.write(json.dumps(self.test_memory_1.dict()) + "\n")

        # 创建 2024-01-16 的记忆文件
        with open(os.path.join(self.daily_dir, "memory_2024-01-16.jsonl"), "w") as f:
            f.write(json.dumps(self.test_memory_2.dict()) + "\n")

        # 创建 2024-01-17 的记忆文件
        with open(os.path.join(self.daily_dir, "memory_2024-01-17.jsonl"), "w") as f:
            f.write(json.dumps(self.test_memory_3.dict()) + "\n")

    @patch.object(CognitiveCorePluginDefaultMemoryManager, "load_time_index")
    @patch.object(CognitiveCorePluginDefaultMemoryManager, "load_daily_memories")
    def test_query_episodic_memories_by_date_range(
        self, mock_load_daily, mock_load_time_index
    ):
        """测试按时间范围查询"""
        # 直接返回时间索引数据，避免文件读取
        mock_load_time_index.return_value = {
            "indexed_dates": {
                "2024-01-15": {
                    "memory_count": 1,
                    "keywords": ["work", "meeting"],
                    "associations": [],
                    "importance_range": [0.8, 0.8],
                },
                "2024-01-16": {
                    "memory_count": 1,
                    "keywords": ["lunch", "friends"],
                    "associations": [],
                    "importance_range": [0.9, 0.9],
                },
                "2024-01-17": {
                    "memory_count": 1,
                    "keywords": ["work", "deadline"],
                    "associations": [],
                    "importance_range": [0.7, 0.7],
                },
            }
        }

        # 模拟每日记忆加载
        def load_daily_side_effect(date_str):
            if date_str == "2024-01-15":
                return [self.test_memory_1]
            elif date_str == "2024-01-16":
                return [self.test_memory_2]
            elif date_str == "2024-01-17":
                return [self.test_memory_3]
            return []

        mock_load_daily.side_effect = load_daily_side_effect

        # 查询 2024-01-15 到 2024-01-16 的记忆
        date_range = ["2024-01-15", "2024-01-16"]
        results = self.memory_manager.query_episodic_memories(date_range)

        # 验证
        assert len(results) == 2
        memory_ids = [memory.id for memory in results]
        assert "mem_1" in memory_ids
        assert "mem_2" in memory_ids
        assert "mem_3" not in memory_ids

    @patch.object(CognitiveCorePluginDefaultMemoryManager, "load_time_index")
    @patch.object(CognitiveCorePluginDefaultMemoryManager, "load_daily_memories")
    def test_query_episodic_memories_by_keywords(
        self, mock_load_daily, mock_load_time_index
    ):
        """测试按关键词查询"""
        # 直接返回时间索引数据，避免文件读取
        mock_load_time_index.return_value = {
            "indexed_dates": {
                "2024-01-15": {
                    "memory_count": 1,
                    "keywords": ["work", "meeting"],
                    "associations": [],
                    "importance_range": [0.8, 0.8],
                },
                "2024-01-16": {
                    "memory_count": 1,
                    "keywords": ["lunch", "friends"],
                    "associations": [],
                    "importance_range": [0.9, 0.9],
                },
                "2024-01-17": {
                    "memory_count": 1,
                    "keywords": ["work", "deadline"],
                    "associations": [],
                    "importance_range": [0.7, 0.7],
                },
            }
        }

        # 模拟每日记忆加载
        def load_daily_side_effect(date_str):
            if date_str == "2024-01-15":
                return [self.test_memory_1]
            elif date_str == "2024-01-16":
                return [self.test_memory_2]
            elif date_str == "2024-01-17":
                return [self.test_memory_3]
            return []

        mock_load_daily.side_effect = load_daily_side_effect

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

    @patch.object(CognitiveCorePluginDefaultMemoryManager, "load_time_index")
    @patch.object(CognitiveCorePluginDefaultMemoryManager, "load_daily_memories")
    def test_query_episodic_memories_importance_filter(
        self, mock_load_daily, mock_load_time_index
    ):
        """测试重要性过滤"""
        # 直接返回时间索引数据，避免文件读取
        mock_load_time_index.return_value = {
            "indexed_dates": {
                "2024-01-15": {
                    "memory_count": 1,
                    "keywords": ["work", "meeting"],
                    "associations": [],
                    "importance_range": [0.8, 0.8],
                },
                "2024-01-16": {
                    "memory_count": 1,
                    "keywords": ["lunch", "friends"],
                    "associations": [],
                    "importance_range": [0.9, 0.9],
                },
                "2024-01-17": {
                    "memory_count": 1,
                    "keywords": ["work", "deadline"],
                    "associations": [],
                    "importance_range": [0.7, 0.7],
                },
            }
        }

        # 模拟每日记忆加载
        def load_daily_side_effect(date_str):
            if date_str == "2024-01-15":
                return [self.test_memory_1]
            elif date_str == "2024-01-16":
                return [self.test_memory_2]
            elif date_str == "2024-01-17":
                return [self.test_memory_3]
            return []

        mock_load_daily.side_effect = load_daily_side_effect

        # 查询重要性 >= 0.85 的记忆
        date_range = ["2024-01-15", "2024-01-17"]
        results = self.memory_manager.query_episodic_memories(
            date_range, importance_min=0.85
        )

        # 验证
        assert len(results) == 1
        assert results[0].id == "mem_2"  # importance=0.9

    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.os.path.exists"
    )
    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.open"
    )
    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.json.dump"
    )
    def test_save_episodic_memories_single(
        self, mock_json_dump, mock_open, mock_exists
    ):
        """测试保存单个记忆"""
        # 模拟文件不存在
        mock_exists.return_value = False

        # 执行
        self.memory_manager.save_episodic_memories([self.test_memory_1])

        # 验证文件操作被调用
        assert mock_open.called
        assert mock_json_dump.called

    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.os.path.exists"
    )
    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.open"
    )
    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.json.dump"
    )
    def test_save_episodic_memories_multiple(
        self, mock_json_dump, mock_open, mock_exists
    ):
        """测试保存多个记忆"""
        # 模拟文件不存在
        mock_exists.return_value = False

        # 执行
        memories = [self.test_memory_1, self.test_memory_2, self.test_memory_3]
        self.memory_manager.save_episodic_memories(memories)

        # 验证文件操作被调用
        assert mock_open.called
        assert mock_json_dump.called

    def test_save_episodic_memories_empty(self):
        """测试保存空记忆列表"""
        # 执行
        self.memory_manager.save_episodic_memories([])

        # 验证 - 不应该抛出异常

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

    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.os.path.exists"
    )
    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.open"
    )
    def test_query_episodic_memories_exception_handling(self, mock_open, mock_exists):
        """测试异常处理"""
        # 模拟文件读取时抛出异常
        mock_exists.return_value = True
        mock_open.side_effect = Exception("File read error")

        with patch("builtins.print") as mock_print:
            results = self.memory_manager.query_episodic_memories(
                ["2024-01-01", "2024-01-31"]
            )

            # 验证
            assert results == []
            mock_print.assert_called_once()
            assert "查询记忆错误" in mock_print.call_args[0][0]

    @patch(
        "lll_cognitive_core.plugins.cognitive_core_plugin_default_memory_manager.os.path.exists"
    )
    def test_load_daily_memories_file_not_exists(self, mock_exists):
        """测试加载不存在的每日记忆文件"""
        mock_exists.return_value = False

        results = self.memory_manager.load_daily_memories("2024-01-01")

        assert results == []

    def test_merge_memories(self):
        """测试记忆合并"""
        memory_1 = TestEpisodicMemoryModel(
            id="mem1", timestamp=datetime.now(), content="Original"
        )
        memory_1_updated = TestEpisodicMemoryModel(
            id="mem1", timestamp=datetime.now(), content="Updated"
        )
        memory_2 = TestEpisodicMemoryModel(
            id="mem2", timestamp=datetime.now(), content="Another"
        )

        existing = [memory_1, memory_2]
        new = [memory_1_updated]

        merged = self.memory_manager.merge_memories(existing, new)

        # 验证
        assert len(merged) == 2
        # 验证 mem1 被更新
        mem1_merged = next(m for m in merged if m.id == "mem1")
        assert mem1_merged.content == "Updated"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
