from flask import Flask, request, jsonify
from ..core.cognitive_core import CognitiveCore
from ..config.cognitive_core_config import CognitiveCoreConfig


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

    @app.route("/receive-event", methods=["POST"])
    def receive_event():
        data = request.json

        if not data:
            return jsonify({"success": False, "error": "缺少参数"})

        type = data.get("type")

        if not type:
            return jsonify({"success": False, "error": "缺少type参数"})

        if type == "wake_up":
            cognitive_core.wake_up()
        elif type == "sleep":
            cognitive_core.sleep()
        else:
            cognitive_core.receive_event(data)

        return jsonify({"success": True})

    return app, cognitive_core
