from jinja2 import Template
from typing import Dict, Any


def generate_template_prompt(
    input_template: str, format_inputs_func, inputs: Dict[str, Any]
):
    formatted_inputs = format_inputs_func(inputs)
    return Template(input_template).render(**formatted_inputs)
