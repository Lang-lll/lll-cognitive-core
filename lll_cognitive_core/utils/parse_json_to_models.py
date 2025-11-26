from pydantic import BaseModel
from typing import List, Any, Union


def parse_jsonstr_to_models(
    response_content: str, data_model: Any
) -> Union[BaseModel, List[BaseModel], None]:
    """
    简化的响应数据解析方法
    """
    try:
        import json

        # 首先解析 JSON
        parsed_data = json.loads(response_content)

        return parse_json_to_models(parsed_data, data_model)

    except Exception as e:
        print(f"解析响应数据错误: {e}")
        return None


def parse_json_to_models(
    parsed_data: Any, data_model: Any
) -> Union[BaseModel, List[BaseModel], None]:
    """
    简化的响应数据解析方法
    """
    try:
        import json

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
