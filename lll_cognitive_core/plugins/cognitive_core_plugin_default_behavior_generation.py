from openai import OpenAI

from lll_simple_ai_shared import (
    BehaviorPlan,
    behavior_task_format_inputs,
)
from ..config.create_openai_config import CreateOpenaiConfig
from ..core.data_structures import GenerateBehaviorInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultBehaviorGeneration:
    def __init__(self, client: OpenAI = None, config: CreateOpenaiConfig = None):
        self._client = client
        self._config = config

    def generate_behavior(
        self, raw_event: GenerateBehaviorInput
    ) -> BehaviorPlan | None:
        # 行为生成
        return get_chat_response(
            GetChatResponseInput(
                client=self._client,
                config=self._config,
                input_template="",
                format_inputs_func=behavior_task_format_inputs,
                inputs=raw_event,
                data_model=BehaviorPlan,
            )
        )
