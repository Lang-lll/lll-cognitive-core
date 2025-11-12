from lll_simple_ai_shared import (
    UnderstoodData,
    understand_system_template,
    understand_task_format_inputs,
)
from ..core.data_structures import DefaultPluginInitOptions, UnderstandEventInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultEventUnderstanding:
    def __init__(self, options: DefaultPluginInitOptions):
        self._options = options or DefaultPluginInitOptions()

    def understand_event(
        self, raw_event: UnderstandEventInput
    ) -> UnderstoodData | None:
        # 事件理解
        return get_chat_response(
            GetChatResponseInput(
                client=self._options.client,
                config=self._options.config,
                input_template=understand_system_template,
                format_inputs_func=understand_task_format_inputs,
                add_messages=self._options.add_messages,
                inputs=raw_event,
                data_model=UnderstoodData,
            )
        )
