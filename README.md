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
    CognitiveCorePluginDefaultAssociativeRecallFilter,
    CognitiveCorePluginDefaultBehaviorGeneration,
    CognitiveCorePluginDefaultBehaviorExecution,
    CognitiveCorePluginDefaultBehaviorExecutionOptions,
    CognitiveCorePluginDefaultMemoryExtraction,
    CognitiveCorePluginDefaultMemoryManager,
)

from lll_simple_ai_shared import (
    understand_output_json_template,
    associative_recall_output_json_template,
    behavior_output_json_template,
    extract_memories_output_json_template,
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
                client=client,
                config=openai_config,
                task_pre_messages=[
                    {
                        "role": "system",
                        "content": understand_output_json_template.render(),
                    }
                ],
            )
        ),
    )

    cognitive_core.register_plugin(
        "associative_recall",
        CognitiveCorePluginDefaultAssociativeRecall(
            DefaultPluginInitOptions(
                client=client,
                config=openai_config,
                task_pre_messages=[
                    {
                        "role": "system",
                        "content": associative_recall_output_json_template.render(),
                    }
                ],
            )
        ),
    )

    cognitive_core.register_plugin(
        "behavior_generation",
        CognitiveCorePluginDefaultBehaviorGeneration(
            DefaultPluginInitOptions(
                client=client,
                config=openai_config,
                task_pre_messages=[
                    {
                        "role": "system",
                        "content": behavior_output_json_template.render(),
                    }
                ],
            )
        ),
    )

    cognitive_core.register_plugin(
        "behavior_execution",
        CognitiveCorePluginDefaultBehaviorExecution(
            CognitiveCorePluginDefaultBehaviorExecutionOptions(
                protocol="http", host="127.0.0.1", port=80, path="/webhook"
            ),
        ),
    )

    cognitive_core.register_plugin(
        "memory_extraction",
        CognitiveCorePluginDefaultMemoryExtraction(
            DefaultPluginInitOptions(
                client=client,
                config=openai_config,
                task_pre_messages=[
                    {
                        "role": "system",
                        "content": extract_memories_output_json_template.render(),
                    }
                ],
            )
        ),
    )

    cognitive_core.register_plugin(
        "memory_manager",
        CognitiveCorePluginDefaultMemoryManager(),
    )

    cognitive_core.register_plugin(
        "associative_recall_filter",
        CognitiveCorePluginDefaultAssociativeRecallFilter(),
    )

    app.run(
        host="0.0.0.0",
        port=9000,
    )
```
