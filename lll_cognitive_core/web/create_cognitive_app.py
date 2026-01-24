from flask import Flask, request, jsonify
from ..core.cognitive_core import CognitiveCore
from ..config.cognitive_core_config import CognitiveCoreConfig
from ..core.plugin_interfaces import CommunicationPlugin


def create_cognitive_app(config: CognitiveCoreConfig = None):
    """创建Flask应用"""
    app = Flask(__name__)

    cognitive_core = CognitiveCore(config)

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"success": True})

    @app.route("/get-system-status", methods=["GET"])
    def get_system_status():
        return jsonify({"success": True, "data": cognitive_core.get_system_status()})

    @app.route("/webhook/orchestrator", methods=["POST"])
    def receive_event():
        data = request.json

        if not data:
            return jsonify({"success": False, "error": "缺少参数"})

        type = data.get("type")

        if not type:
            return jsonify({"success": False, "error": "缺少type参数"})

        communication: CommunicationPlugin = cognitive_core.get_plugin("communication")

        if type and communication:
            # TODO: 把http放到插件
            if type == "registered" or type == "heartbeat":
                communication.receive_messages(data)
            elif type == "wake_up":
                cognitive_core.wake_up()
            elif type == "sleep":
                cognitive_core.sleep()
            elif type == "publish_status":
                cognitive_core.publish_status()
            else:
                cognitive_core.receive_event(data)

        return jsonify({"success": True})

    return app, cognitive_core
