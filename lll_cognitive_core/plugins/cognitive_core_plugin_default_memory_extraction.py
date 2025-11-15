from typing import List
from lll_simple_ai_shared import (
    EpisodicMemoriesGenerateModels,
    extract_memories_system_template,
    extract_memories_task_format_inputs,
)
from ..core.data_structures import DefaultPluginInitOptions, ExtractMemoriesInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultMemoryExtraction:
    def __init__(self, options: DefaultPluginInitOptions):
        self._options = options or DefaultPluginInitOptions()

    def extract_memories(
        self, raw_event: ExtractMemoriesInput
    ) -> List[EpisodicMemoriesGenerateModels] | None:
        # 记忆整理
        return get_chat_response(
            GetChatResponseInput(
                client=self._options.client,
                config=self._options.config,
                input_template=extract_memories_system_template,
                format_inputs_func=extract_memories_task_format_inputs,
                task_pre_messages=self._options.task_pre_messages,
                inputs=raw_event,
                data_model=List[EpisodicMemoriesGenerateModels],
            )
        )
