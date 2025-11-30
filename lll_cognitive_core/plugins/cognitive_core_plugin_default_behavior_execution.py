import json
import requests
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any
from ..core.plugin_interfaces import BehaviorExecutionPlugin


@dataclass
class CognitiveCorePluginDefaultBehaviorExecutionOptions:
    protocol: str
    host: str
    port: int
    path: str

    @property
    def url(self):
        """构建完整的URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"


class CognitiveCorePluginDefaultBehaviorExecution(BehaviorExecutionPlugin):
    def __init__(self, options: CognitiveCorePluginDefaultBehaviorExecutionOptions):
        self._options = options or CognitiveCorePluginDefaultBehaviorExecutionOptions()

    def execute_tts_action(self, action):
        try:
            url = self._options.url
            json_data = action.model_dump()

            response = requests.post(
                url,
                json=json_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"""tts任务发送失败: {e}""")

    def execute_motion_action(self, action, options):

        try:
            url = self._options.url
            action_data = action.model_dump()

            response = requests.post(
                url,
                json={
                    "type": options.type,
                    "speed": options.speed,
                    "intensity": options.intensity,
                    "action_data": action_data,
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"""motion任务发送失败: {e}""")
