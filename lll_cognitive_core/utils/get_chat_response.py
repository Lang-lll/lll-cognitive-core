from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Type, Union
from dataclasses import dataclass
from openai import OpenAI
from ..config.create_openai_config import CreateOpenaiConfig
from .generate_template_prompt import generate_template_prompt
from .parse_json_to_models import parse_jsonstr_to_models


@dataclass
class GetChatResponseInput:
    client: OpenAI
    config: CreateOpenaiConfig
    input_template: str
    format_inputs_func: Any
    inputs: BaseModel
    data_model: Type[Union[BaseModel, List[BaseModel]]]
    task_pre_messages: Optional[List[Dict[str, str]]] = None


def get_chat_response(data: GetChatResponseInput):
    try:
        pre_messages = data.config.pre_messages or []
        all_messages = list(
            pre_messages
            + (data.task_pre_messages or [])
            + [
                {
                    "role": "user",
                    "content": generate_template_prompt(
                        data.input_template,
                        data.format_inputs_func,
                        data.inputs,
                    ),
                }
            ]
        )

        if data.client is not None and data.config is not None:
            response = data.client.chat.completions.create(
                model=data.config.model,
                messages=all_messages,
                response_format={"type": "json_object"},
                timeout=data.config.timeout,
                temperature=data.config.temperature,
            )

            return parse_jsonstr_to_models(
                response.choices[0].message.content, data.data_model
            )
    except Exception as e:
        print(f"调用模型错误: {e}")
        return None
