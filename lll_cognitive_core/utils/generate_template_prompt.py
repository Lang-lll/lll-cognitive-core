from pydantic import BaseModel
from jinja2 import Template
from typing import Dict, Any


def generate_template_prompt(
    input_template: str, format_inputs_func, inputs: BaseModel
):
    formatted_inputs = format_inputs_func(inputs.model_dump())
    return Template(input_template).render(**formatted_inputs)
