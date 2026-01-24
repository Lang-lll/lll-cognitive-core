import requests
from dataclasses import dataclass
from typing import Any
from ..core.plugin_interfaces import CommunicationPlugin
from ..utils.simple_heartbeat_client import SimpleHeartbeatClient


@dataclass
class CognitiveCorePluginDefaultCommunicationOptions:
    protocol: str
    host: str
    port: int
    path: str

    @property
    def url(self):
        """构建完整的URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"


# TODO: 执行队列
class CognitiveCorePluginDefaultCommunication(CommunicationPlugin):
    def __init__(self, options: CognitiveCorePluginDefaultCommunicationOptions):
        self._options = options or CognitiveCorePluginDefaultCommunicationOptions()
        self._heartbeat_client = SimpleHeartbeatClient(
            self._options.url,
            {
                "type": "register",
                "message": {
                    "plugin_name": "cognitive_core",
                    "version": "0.1",
                    "transportUrl": "http://localhost:9101",
                },
            },
            {"type": "heartbeat", "plugin_name": "cognitive_core"},
        )
        self._heartbeat_client.start()

    def receive_messages(self, message):
        # TODO: logger
        type = message.get("type")

        if type == "registered":
            self._heartbeat_client.receive_registered()
        elif type == "heartbeat":
            self._heartbeat_client.receive_heartbeat()

    def send_message(self, data):
        type = data.get("type")
        url = self._options.url

        try:
            if type == "action" or type == "publish_status":
                response = requests.post(
                    url,
                    json={
                        "type": "publish",
                        "to_plugin": ["humanoid_server"],
                        "message": data.get("message"),
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=20,
                )
                response.raise_for_status()
                print(f"""发送任务: {data}， 结果: {response.json()}""")
        except Exception as e:
            print(f"""任务发送失败: {e}""")
