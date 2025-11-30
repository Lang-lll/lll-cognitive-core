import os
import json
from typing import List, Any

from lll_simple_ai_shared import (
    ActionIndexModels,
    ActionCategoryModels,
    ActionDataModels,
)
from ..core.plugin_interfaces import ActionManagerPlugin
from ..utils.parse_json_to_models import parse_json_to_models

INDEX_PATH = "memory/index/action_index.json"


class CognitiveCorePluginDefaultActionManager(ActionManagerPlugin):
    def __init__(self):
        self.main_index: List[ActionIndexModels] | None = None

    def get_main_index(self) -> List[ActionIndexModels]:
        """获取主索引"""
        if self.main_index is None:
            print(f"{self._load_json_str(INDEX_PATH, [])}")
            self.main_index = parse_json_to_models(
                self._load_json_str(INDEX_PATH, []), List[ActionIndexModels]
            )
        return self.main_index

    def get_category_actions(self, category_name) -> List[ActionCategoryModels]:
        """获取分类下的动作列表"""
        return parse_json_to_models(
            self._load_json_str(f"memory/action/categories/{category_name}.json", []),
            List[ActionCategoryModels],
        )

    def get_action_data(self, category_name, action_id) -> ActionDataModels | None:
        """获取具体动作数据"""
        return parse_json_to_models(
            self._load_json_str(
                f"memory/action/data/{category_name}/{action_id}.json", None
            ),
            ActionDataModels,
        )

    def _load_json_str(self, file_path: str, fallback: Any):
        """加载JSON文件"""
        try:
            if not os.path.exists(file_path):
                return fallback
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(
                f"""动作查询加载json失败: {e}
路径: {file_path}"""
            )
            return fallback
