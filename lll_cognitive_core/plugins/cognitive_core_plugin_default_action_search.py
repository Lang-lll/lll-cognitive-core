import os
import json

from ..core.plugin_interfaces import ActionSearchPlugin

INDEX_PATH = "memory/actions_index.json"


class CognitiveCorePluginDefaultMorningSituation(ActionSearchPlugin):
    def __init__(self):
        self.main_index = None

    def get_main_index(self):
        """获取主索引"""
        if self.main_index is None:
            self.main_index = self._load_json("index.json", [])
        return self.main_index

    def get_category_actions(self, category_name):
        """获取分类下的动作列表"""
        return self._load_json(f"memory/actions/category/{category_name}.json", [])

    def get_action_data(self, category_name, action_id):
        """获取具体动作数据"""
        return self._load_json(
            f"memory/actions/data/{category_name}/{action_id}.json", {}
        )

    def _load_json(self, file_path, fallback):
        """加载JSON文件"""
        try:
            if not os.path.exists(file_path):
                return fallback
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"动作查询加载json失败: {e}")
            return fallback
