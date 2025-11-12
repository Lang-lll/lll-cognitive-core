from lll_simple_ai_shared import (
    BehaviorPlan,
    behavior_system_template,
    behavior_task_format_inputs,
)
from ..core.data_structures import DefaultPluginInitOptions, GenerateBehaviorInput
from ..utils.get_chat_response import GetChatResponseInput, get_chat_response


class CognitiveCorePluginDefaultBehaviorGeneration:
    def __init__(self, options: DefaultPluginInitOptions):
        self._options = options or DefaultPluginInitOptions()

    def generate_behavior(
        self, raw_event: GenerateBehaviorInput
    ) -> BehaviorPlan | None:
        # 行为生成
        return get_chat_response(
            GetChatResponseInput(
                client=self._options.client,
                config=self._options.config,
                input_template=behavior_system_template,
                format_inputs_func=behavior_task_format_inputs,
                add_messages=self._options.add_messages,
                inputs=raw_event,
                data_model=BehaviorPlan,
            )
        )
