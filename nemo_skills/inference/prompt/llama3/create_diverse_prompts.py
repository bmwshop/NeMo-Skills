"""File to create diverse prompts to test robustness of our models."""

import re
from pathlib import Path
from variants import VARIANTS


PROMPT_SKELETON = """
system: ""

user: |-
  {}

  {question}{context}

prompt_template: |-
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  {system}<|eot_id|><|start_header_id|>user<|end_header_id|>

  {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

  {generation}

stop_phrases: ["<|eot_id|>"]
"""


def populate_prompt_template(template, instruction):
    result = template.replace("{}", instruction).strip()
    return result


def main():
    output_dir = Path(__file__).parent.parent / "llama3-robustness"
    for idx, variant in enumerate(VARIANTS):
        prompt = populate_prompt_template(PROMPT_SKELETON, variant)

        output_file = output_dir / f"prompt_{idx}.yaml"
        with open(output_file, "w") as out_w:
            out_w.write(prompt)

        # break


if __name__ == '__main__':
    main()