from typing import List, Dict
from dataclasses import dataclass


@dataclass
class CreateOpenaiConfig:
    base_url: str
    api_key: str
    model: str
    timeout = 60
    temperature = 1.1
    pre_messages: List[Dict[str, str]]
