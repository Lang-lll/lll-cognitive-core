import json
import requests
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any
from ..core.plugin_interfaces import BehaviorExecutionPlugin


@dataclass
class DefaultBehaviorExecutionInitOptions:
    protocol: str
    host: str
    port: int
    path: str

    @property
    def url(self):
        """构建完整的URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"


class CognitiveCorePluginDefaultBehaviorExecution(BehaviorExecutionPlugin):
    def __init__(self, options: DefaultBehaviorExecutionInitOptions):
        self._options = options or DefaultBehaviorExecutionInitOptions()

    def execute_behavior_plan(self, action: Any):
        """
        执行行为计划，向指定地址发送action数据

        Args:
            action: 要发送的行为数据，可以是BaseModel、Dict或任何可JSON序列化的对象
        """
        try:
            # 构建完整的URL
            url = self._options.url

            # 统一处理数据序列化
            if isinstance(action, BaseModel):
                # 如果是BaseModel，使用model_dump()
                json_data = action.model_dump()
            elif isinstance(action, dict):
                # 如果是字典，直接使用
                json_data = action
            else:
                # 其他类型，尝试直接序列化
                json_data = action

            # 发送POST请求
            response = requests.post(
                url,
                json=json_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
        except json.JSONEncoder as e:
            print(f"JSON序列化失败: {e}")
