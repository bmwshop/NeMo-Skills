# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# needs to define NEMO_SKILLS_TEST_TRTLLM_MODEL to run these tests
# needs to define NEMO_SKILLS_TEST_NEMO_MODEL to run these tests
# you'd also need 2+ GPUs to run this test
# the metrics are assuming llama3-8b-base as the model and will fail for other models

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1] / 'pipeline'))
from compute_metrics import compute_metrics


def test_trtllm_run_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_TRTLLM_MODEL to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type tensorrt_llm \
    --output_dir {output_path} \
    --benchmarks gsm8k:0 \
    --num_gpus 2 \
    --num_nodes 1 \
    +prompt=code_base \
    ++prompt.few_shot_examples.examples_type=gsm8k_text_with_code \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++split_name=test \
    batch_size=8 \
    max_samples=20 \
"""
    subprocess.run(
        cmd,
        shell=True,
    )  # not checking the error as it's expected to finish with non-zero error code

    # running compute_metrics to check that results are expected
    correct_answer, wrong_answer, no_answer, total = compute_metrics([f"{output_path}/gsm8k/output-greedy.jsonl"])
    assert correct_answer == 40.0
    assert wrong_answer == 45.0
    assert no_answer == 15.0
    assert total == 20

    # double checking that code was actually executed
    with open(f"{output_path}/gsm8k/output-greedy.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' in elem['generated_solution']
        assert elem['error_message'] != '<not_executed>'