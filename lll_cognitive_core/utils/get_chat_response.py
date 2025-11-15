from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Type, Union
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
            )

            print(f"模型响应: {response.choices[0].message.content}")
            return parse_response_data(
                response.choices[0].message.content, data.data_model
            )
    except Exception as e:
        print(f"调用模型错误: {e}")
        return None


def parse_response_data(
    response_content: str, data_model: Any
) -> Union[BaseModel, List[BaseModel], None]:
    """
    简化的响应数据解析方法
    """
    try:
        import json

        # 首先解析 JSON
        parsed_data = json.loads(response_content)

        # 检查是否是列表类型
        if isinstance(parsed_data, list):
            # 从 data_model 中提取列表项的类型
            if hasattr(data_model, "__origin__") and data_model.__origin__ is list:
                item_type = data_model.__args__[0]
                if issubclass(item_type, BaseModel):
                    return [item_type.model_validate(item) for item in parsed_data]
            else:
                print("错误: data_model 不是 List 类型")
                return None
        else:
            # 单个 BaseModel 类型
            if issubclass(data_model, BaseModel):
                return data_model.model_validate(parsed_data)
            else:
                print("错误: data_model 不是 BaseModel 子类")
                return None

    except Exception as e:
        print(f"解析响应数据错误: {e}")
        return None
