from .core.cognitive_core import CognitiveCore
from .web.create_cognitive_app import create_cognitive_app
from .plugins.cognitive_core_plugin_default_memory_manager import (
    CognitiveCorePluginDefaultMemoryManager,
)
from .core.plugin_interfaces import MemoryManagerPlugin

__version__ = "0.1.0"
__all__ = [
    "CognitiveCore",
    "create_cognitive_app",
    "MemoryManagerPlugin",
    "CognitiveCorePluginDefaultMemoryManager",
]
