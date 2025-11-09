import os
from openai import OpenAI
from ..config.create_openai_config import CreateOpenaiConfig


def create_openai(config: CreateOpenaiConfig = None):
    return OpenAI(api_key=os.environ.get(config.api_key_name), base_url=config.base_url)
