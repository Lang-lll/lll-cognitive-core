from openai import OpenAI

from lll_simple_ai_shared import (
    EpisodicMemoriesGenerateModels,
    extract_memories_task_format_inputs,
)
from ..config.create_openai_config import CreateOpenaiConfig
from ..core.data_structures import ExtractMemoriesInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultMemoryExtraction:
    def __init__(self, client: OpenAI = None, config: CreateOpenaiConfig = None):
        self._client = client
        self._config = config

    def extract_memories(
        self, raw_event: ExtractMemoriesInput
    ) -> EpisodicMemoriesGenerateModels | None:
        # 记忆整理
        return get_chat_response(
            GetChatResponseInput(
                client=self._client,
                config=self._config,
                input_template="",
                format_inputs_func=extract_memories_task_format_inputs,
                inputs=raw_event,
                data_model=EpisodicMemoriesGenerateModels,
            )
        )
