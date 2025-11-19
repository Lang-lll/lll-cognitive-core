from .core.cognitive_core import CognitiveCore
from .web.create_cognitive_app import create_cognitive_app
from .web.create_openai import create_openai
from .core.plugin_interfaces import (
    MorningSituationPlugin,
    EventUnderstandingPlugin,
    AssociativeRecallPlugin,
    BehaviorGenerationPlugin,
    MemoryManagerPlugin,
)
from .config.cognitive_core_config import CognitiveCoreConfig
from .config.create_openai_config import CreateOpenaiConfig
from .utils.get_chat_response import get_chat_response, GetChatResponseInput
from .utils.generate_template_prompt import generate_template_prompt
from .core.data_structures import DefaultPluginInitOptions

from .plugins.cognitive_core_plugin_default_morning_situation import (
    CognitiveCorePluginDefaultMorningSituation,
)
from .plugins.cognitive_core_plugin_default_event_understanding import (
    CognitiveCorePluginDefaultEventUnderstanding,
)
from .plugins.cognitive_core_plugin_default_associative_recall import (
    CognitiveCorePluginDefaultAssociativeRecall,
)
from .plugins.cognitive_core_plugin_default_associative_recall_filter import (
    CognitiveCorePluginDefaultAssociativeRecallFilter,
)
from .plugins.cognitive_core_plugin_default_behavior_generation import (
    CognitiveCorePluginDefaultBehaviorGeneration,
)
from .plugins.cognitive_core_plugin_default_behavior_execution import (
    CognitiveCorePluginDefaultBehaviorExecution,
    CognitiveCorePluginDefaultBehaviorExecutionOptions,
)
from .plugins.cognitive_core_plugin_default_memory_extraction import (
    CognitiveCorePluginDefaultMemoryExtraction,
)
from .plugins.cognitive_core_plugin_default_memory_manager import (
    CognitiveCorePluginDefaultMemoryManager,
)

__version__ = "0.1.0"
__all__ = [
    "CognitiveCore",
    "create_cognitive_app",
    "create_openai",
    "MorningSituationPlugin",
    "EventUnderstandingPlugin",
    "AssociativeRecallPlugin",
    "BehaviorGenerationPlugin",
    "MemoryManagerPlugin",
    "CognitiveCorePluginDefaultMemoryManager",
    "CognitiveCoreConfig",
    "CreateOpenaiConfig",
    "get_chat_response",
    "generate_template_prompt",
    "GetChatResponseInput",
    "DefaultPluginInitOptions",
    "CognitiveCorePluginDefaultMorningSituation",
    "CognitiveCorePluginDefaultEventUnderstanding",
    "CognitiveCorePluginDefaultAssociativeRecall",
    "CognitiveCorePluginDefaultAssociativeRecallFilter",
    "CognitiveCorePluginDefaultBehaviorGeneration",
    "CognitiveCorePluginDefaultBehaviorExecution",
    "CognitiveCorePluginDefaultBehaviorExecutionOptions",
    "CognitiveCorePluginDefaultMemoryExtraction",
]
