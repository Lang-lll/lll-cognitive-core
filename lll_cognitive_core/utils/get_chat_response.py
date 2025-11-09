from pydantic import BaseModel
from typing import Dict, Any
from dataclasses import dataclass
from openai import OpenAI
from ..config.create_openai_config import CreateOpenaiConfig
from .generate_template_prompt import generate_template_prompt


@dataclass
class GetChatResponseInput:
    client: OpenAI
    config: CreateOpenaiConfig
    input_template: str
    format_inputs_func: Any
    inputs: Dict[str, Any]
    data_model: BaseModel


def get_chat_response(data: GetChatResponseInput):
    try:
        pre_messages = data.config.pre_messages or []

        if data.client is not None and data.config is not None:
            response = data.client.chat.completions.create(
                model=data.config.model,
                messages=list(
                    dict.fromkeys(
                        pre_messages
                        + [
                            {
                                "role": "system",
                                "content": generate_template_prompt(
                                    data.input_template,
                                    data.format_inputs_func,
                                    data.inputs,
                                ),
                            }
                        ]
                    )
                ),
                response_format={"type": "json_object"},
            )

            return data.data_model.model_validate_json(
                response.choices[0].message.content
            )
    except Exception as e:
        print(f"调用模型错误: {e}")
        return None
