from openai import OpenAI

from lll_simple_ai_shared import (
    RecallResultsModels,
    associative_recall_task_format_inputs,
)
from ..config.create_openai_config import CreateOpenaiConfig
from ..core.data_structures import AssociativeRecallInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultAssociativeRecall:
    def __init__(self, client: OpenAI = None, config: CreateOpenaiConfig = None):
        self._client = client
        self._config = config

    def associative_recall(
        self, raw_event: AssociativeRecallInput
    ) -> RecallResultsModels | None:
        # 回想
        return get_chat_response(
            GetChatResponseInput(
                client=self._client,
                config=self._config,
                input_template="",
                format_inputs_func=associative_recall_task_format_inputs,
                inputs=raw_event,
                data_model=RecallResultsModels,
            )
        )
