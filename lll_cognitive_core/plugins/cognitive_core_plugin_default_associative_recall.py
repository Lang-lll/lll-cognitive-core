from lll_simple_ai_shared import (
    RecallResultsModels,
    associative_recall_system_template,
    associative_recall_task_format_inputs,
)
from ..core.data_structures import DefaultPluginInitOptions, AssociativeRecallInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultAssociativeRecall:
    def __init__(self, options: DefaultPluginInitOptions):
        self._options = options or DefaultPluginInitOptions()

    def associative_recall(
        self, raw_event: AssociativeRecallInput
    ) -> RecallResultsModels | None:
        # 回想
        return get_chat_response(
            GetChatResponseInput(
                client=self._options.client,
                config=self._options.config,
                input_template=associative_recall_system_template,
                format_inputs_func=associative_recall_task_format_inputs,
                task_pre_messages=self._options.task_pre_messages,
                inputs=raw_event,
                data_model=RecallResultsModels,
            )
        )
