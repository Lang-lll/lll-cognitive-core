import json
import requests
from dataclasses import dataclass
from typing import Any
from ..core.plugin_interfaces import BehaviorExecutionPlugin
from ..utils.simple_heartbeat_client import SimpleHeartbeatClient


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


# TODO: 执行队列
class CognitiveCorePluginDefaultBehaviorExecution(BehaviorExecutionPlugin):
    def __init__(self, options: CognitiveCorePluginDefaultBehaviorExecutionOptions):
        self._options = options or CognitiveCorePluginDefaultBehaviorExecutionOptions()
        self._heartbeat_client = SimpleHeartbeatClient(
            self._options.url,
            {"type": "register", "plugin_name": "cognitive_core", "version": "0.1"},
            {"type": "heartbeat"},
        )
        self._heartbeat_client.start()

    def receive_registered(self):
        self._heartbeat_client.receive_registered()

    def receive_heartbeat(self):
        self._heartbeat_client.receive_heartbeat()

    def publish_status(self, status):
        try:
            url = self._options.url

            response = requests.post(
                url,
                json={
                    type: "publish",
                    "to_plugin": ["humanoid_server"],
                    "message": {status: status},
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"""tts任务发送失败: {e}""")

    def execute_tts_action(self, action):
        try:
            url = self._options.url
            json_data = action.model_dump()

            response = requests.post(
                url,
                json={
                    "type": "publish",
                    "to_plugin": ["humanoid_server"],
                    "message": json_data,
                },
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
                    "type": "publish",
                    "to_plugin": ["humanoid_server"],
                    "message": {
                        "type": options.type,
                        "action_id": options.action_id,
                        "speed": options.speed,
                        "intensity": options.intensity,
                        "action_data": action_data,
                    },
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"""motion任务发送失败: {e}""")

    def execute_wait_action(self, action):
        try:
            url = self._options.url
            json_data = action.model_dump()

            response = requests.post(
                url,
                json={
                    "type": "publish",
                    "to_plugin": ["humanoid_server"],
                    "message": json_data,
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"""tts任务发送失败: {e}""")
