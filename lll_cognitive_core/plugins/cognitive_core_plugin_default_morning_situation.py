from lll_simple_ai_shared import (
    MorningSituationModels,
    morning_situation_system_template,
    morning_situation_task_format_inputs,
)
from ..core.data_structures import DefaultPluginInitOptions, MorningSituationInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultMorningSituation:
    def __init__(self, options: DefaultPluginInitOptions):
        self._options = options or DefaultPluginInitOptions()

    def generate_morning_situation(
        self, data: MorningSituationInput
    ) -> MorningSituationModels | None:
        # 起床
        return get_chat_response(
            GetChatResponseInput(
                client=self._options.client,
                config=self._options.config,
                input_template=morning_situation_system_template,
                format_inputs_func=morning_situation_task_format_inputs,
                task_pre_messages=self._options.task_pre_messages,
                inputs=data,
                data_model=MorningSituationModels,
            )
        )
