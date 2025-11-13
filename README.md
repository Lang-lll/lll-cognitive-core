# AI Cognitive Core

AI 认知核心

## 安装

```bash
pip install lll-cognitive-core
```

## 使用

```python
import os
from dotenv import load_dotenv
from lll_cognitive_core import (
    create_cognitive_app,
    create_openai,
    CognitiveCoreConfig,
    CreateOpenaiConfig,
    DefaultPluginInitOptions,
    CognitiveCorePluginDefaultEventUnderstanding,
    CognitiveCorePluginDefaultAssociativeRecall,
    CognitiveCorePluginDefaultBehaviorGeneration,
    CognitiveCorePluginDefaultBehaviorExecution,
    CognitiveCorePluginDefaultBehaviorExecutionOptions,
    CognitiveCorePluginDefaultMemoryExtraction,
    CognitiveCorePluginDefaultMemoryManager,
)


def main():
    core_config = CognitiveCoreConfig()
    _, cognitive_core = create_cognitive_app(core_config)

    # 加载.env文件，或者你也可以用其他方式
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_config = CreateOpenaiConfig(
        base_url="",
        api_key=api_key,
        model="deepseek-chat",
        pre_messages=[{"role": "system", "content": "You are a helpful assistant."}],
    )
    client = create_openai(openai_config)

    cognitive_core.register_plugin(
        "event_understanding",
        CognitiveCorePluginDefaultEventUnderstanding(
            DefaultPluginInitOptions(
                client=client, config=openai_config, add_messages=[]
            )
        ),
    )

    cognitive_core.register_plugin(
        "associative_recall",
        CognitiveCorePluginDefaultAssociativeRecall(
            DefaultPluginInitOptions(
                client=client, config=openai_config, add_messages=[]
            )
        ),
    )

    cognitive_core.register_plugin(
        "behavior_generation",
        CognitiveCorePluginDefaultBehaviorGeneration(
            DefaultPluginInitOptions(
                client=client, config=openai_config, add_messages=[]
            )
        ),
    )

    cognitive_core.register_plugin(
        "behavior_execution",
        CognitiveCorePluginDefaultBehaviorExecution(
            CognitiveCorePluginDefaultBehaviorExecutionOptions(
                protocol="http", host="", port=80, path="/"
            ),
        ),
    )

    cognitive_core.register_plugin(
        "memory_extraction",
        CognitiveCorePluginDefaultMemoryExtraction(
            DefaultPluginInitOptions(
                client=client, config=openai_config, add_messages=[]
            )
        ),
    )

    cognitive_core.register_plugin(
        "memory_manager",
        CognitiveCorePluginDefaultMemoryManager(),
    )

    app.run(
        host="0.0.0.0",
        port=9000,
    )
```
