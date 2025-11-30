from openai import OpenAI
from ..config.create_openai_config import CreateOpenaiConfig


def create_openai(config: CreateOpenaiConfig = None):
    return OpenAI(api_key=config.api_key, base_url=config.base_url)
