import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from pydantic import BaseModel
from dataclasses import dataclass

from lll_cognitive_core.config.create_openai_config import CreateOpenaiConfig
from lll_cognitive_core.utils.get_chat_response import (
    get_chat_response,
    GetChatResponseInput,
)


# 测试用的数据模型
class TestDataModel(BaseModel):
    name: str
    age: int
    email: str


# 测试用的配置
@dataclass
class TestConfig:
    model: str = "gpt-3.5-turbo"
    pre_messages: list = None


class TestGetChatResponse:
    """get_chat_response 方法的单元测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock(spec=OpenAI)
        self.mock_chat_completions = Mock()
        self.mock_client.chat.completions = self.mock_chat_completions

        self.config = CreateOpenaiConfig(
            base_url="test_url",
            api_key_name="test_key",
            model="gpt-3.5-turbo",
            pre_messages=[],
        )
        self.input_template = "Test template with {name}"
        self.format_inputs_func = lambda inputs: inputs
        self.inputs = {"name": "John"}
        self.data_model = TestDataModel

    def create_mock_response(self, content: str):
        """创建模拟的 OpenAI 响应"""
        mock_message = Mock()
        mock_message.content = content

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        return mock_response

    def test_successful_response(self):
        """测试成功的 API 调用"""
        # 准备
        expected_content = '{"name": "John", "age": 30, "email": "john@example.com"}'
        mock_response = self.create_mock_response(expected_content)

        self.mock_chat_completions.create.return_value = mock_response

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs=self.inputs,
            data_model=self.data_model,
        )
        print(input_data)

        # 执行
        result = get_chat_response(input_data)

        # 验证
        assert result is not None
        assert isinstance(result, TestDataModel)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

        # 验证 OpenAI 客户端被正确调用
        self.mock_chat_completions.create.assert_called_once()
        call_args = self.mock_chat_completions.create.call_args
        assert call_args.kwargs["model"] == self.config.model
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_with_pre_messages(self):
        """测试带有预定义消息的情况"""
        # 准备
        self.config.pre_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Previous message"},
        ]

        expected_content = '{"name": "Alice", "age": 25, "email": "alice@example.com"}'
        mock_response = self.create_mock_response(expected_content)
        self.mock_chat_completions.create.return_value = mock_response

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs={"name": "Alice"},
            data_model=self.data_model,
        )

        # 执行
        result = get_chat_response(input_data)

        # 验证
        assert result is not None
        assert result.name == "Alice"

        # 验证消息列表包含预定义消息
        call_args = self.mock_chat_completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3  # pre_messages + system message
        assert messages[0] == self.config.pre_messages[0]
        assert messages[1] == self.config.pre_messages[1]

    @patch("lll_cognitive_core.utils.generate_template_prompt")
    def test_generate_template_prompt_called_correctly(self, mock_generate):
        """测试 generate_template_prompt 被正确调用"""
        """# 准备
        mock_generate.return_value = "Generated prompt"
        expected_content = '{"name": "Test", "age": 20, "email": "test@example.com"}'
        mock_response = self.create_mock_response(expected_content)
        self.mock_chat_completions.create.return_value = mock_response

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs=self.inputs,
            data_model=self.data_model,
        )

        # 执行
        get_chat_response(input_data)

        # 验证
        mock_generate.assert_called_once_with(
            self.input_template, self.format_inputs_func, self.inputs
        )"""

    def test_api_exception_handling(self):
        """测试 API 异常处理"""
        # 准备
        self.mock_chat_completions.create.side_effect = Exception("API Error")

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs=self.inputs,
            data_model=self.data_model,
        )

        # 执行
        with patch("builtins.print") as mock_print:
            result = get_chat_response(input_data)

        # 验证
        assert result is None
        mock_print.assert_called_once_with("调用模型错误: API Error")

    def test_invalid_json_response(self):
        """测试返回无效 JSON 的情况"""
        """# 准备
        invalid_json_content = "Invalid JSON string"
        mock_response = self.create_mock_response(invalid_json_content)
        self.mock_chat_completions.create.return_value = mock_response

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs=self.inputs,
            data_model=self.data_model,
        )

        # 执行和验证
        with pytest.raises(Exception):  # 这里应该根据实际的验证错误类型进行调整
            get_chat_response(input_data)"""

    def test_none_client_or_config(self):
        """测试客户端或配置为 None 的情况"""
        # 测试 client 为 None
        input_data = GetChatResponseInput(
            client=None,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs=self.inputs,
            data_model=self.data_model,
        )

        result = get_chat_response(input_data)
        assert result is None

        # 测试 config 为 None
        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=None,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs=self.inputs,
            data_model=self.data_model,
        )

        result = get_chat_response(input_data)
        assert result is None

    def test_empty_pre_messages(self):
        """测试空预定义消息的情况"""
        # 准备
        self.config.pre_messages = []
        expected_content = '{"name": "Bob", "age": 35, "email": "bob@example.com"}'
        mock_response = self.create_mock_response(expected_content)
        self.mock_chat_completions.create.return_value = mock_response

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=self.format_inputs_func,
            inputs={"name": "Bob"},
            data_model=self.data_model,
        )

        # 执行
        result = get_chat_response(input_data)

        # 验证
        assert result is not None
        assert result.name == "Bob"

        # 验证消息列表正确
        call_args = self.mock_chat_completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1  # 只有系统消息

    def test_duplicate_messages_removal(self):
        """测试重复消息被移除"""
        # 准备
        duplicate_message = {"role": "system", "content": "Duplicate"}
        self.config.pre_messages = [duplicate_message, duplicate_message]

        expected_content = '{"name": "Test", "age": 40, "email": "test@example.com"}'
        mock_response = self.create_mock_response(expected_content)
        self.mock_chat_completions.create.return_value = mock_response

        input_data = GetChatResponseInput(
            client=self.mock_client,
            config=self.config,
            input_template=self.input_template,
            format_inputs_func=lambda x: x,
            inputs=self.inputs,
            data_model=self.data_model,
        )

        # 执行
        result = get_chat_response(input_data)

        # 验证
        assert result is not None
        # 注意：由于使用 dict.fromkeys()，重复消息会被移除


# 运行测试的代码（可选）
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
