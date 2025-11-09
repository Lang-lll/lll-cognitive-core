from openai import OpenAI

from lll_simple_ai_shared import UnderstoodData, understand_task_format_inputs
from ..config.create_openai_config import CreateOpenaiConfig
from ..core.data_structures import UnderstandEventInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultEventUnderstanding:
    def __init__(self, client: OpenAI = None, config: CreateOpenaiConfig = None):
        self._client = client
        self._config = config

    def understand_event(
        self, raw_event: UnderstandEventInput
    ) -> UnderstoodData | None:
        # 事件理解
        return get_chat_response(
            GetChatResponseInput(
                client=self._client,
                config=self._config,
                input_template="",
                format_inputs_func=understand_task_format_inputs,
                inputs=raw_event,
                data_model=UnderstoodData,
            )
        )
