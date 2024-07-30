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

import json
import logging
import os
import shutil
import subprocess
from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict, field
from os import path
from pathlib import Path
from typing import Optional

from nemo_skills.utils import nested_dataclass, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass
class SympyGraderConfig:
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=lambda: {'sandbox_type': 'local'})
    num_parallel_requests: int = 100
    in_memory_lines: int = 1500
    include_percentage: bool = True
    tolerance: float = 1e-4
    timeout: float = 10.0
    ignore_cache: bool = False


@nested_dataclass
class LlmGraderConfig:
    batch_size: int = 100  # lower if running into rate limits
    tokens_to_generate: int = 4096  # will auto-lower to max possible for NGC models
    use_batch_api: bool = True  # only supported for OpenAI models!
    base_url: Optional[str] = None
    judge_model: str = "gpt-4-1106-preview"
    # defaults to True to avoid regenerating judgements unless necessary
    skip_filled: bool = True


@nested_dataclass
class MathGraderConfig:
    # Sandbox configuration {sandbox_params}
    grading_type: str = "sympy"  # sympy or llm
    grading_config: dict = field(default_factory=dict)

    extract_from_boxed: bool = True
    # only used if extract_from_boxed is False
    extract_regex: str = r"The final answer is (.+)$"


def math_grader(cfg):
    eval_config = MathGraderConfig(**cfg.eval_config)
    if eval_config.grading_type == "sympy":
        from nemo_skills.code_execution.sandbox import get_sandbox

        grading_config = SympyGraderConfig(**eval_config.grading_config)
        sandbox = get_sandbox(**grading_config.sandbox)
        grading_config = asdict(grading_config)
        grading_config.pop('sandbox')
        sandbox.batch_evaluate_results(
            prediction_jsonl_files=cfg.prediction_jsonl_files,
            **grading_config,
            extract_from_boxed=eval_config.extract_from_boxed,
            extract_regex=eval_config.extract_regex,
        )
        return

    from tqdm import tqdm

    from nemo_skills.code_execution.math_grader import extract_answer
    from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config
    from nemo_skills.inference.server.model import get_model

    grading_config = LlmGraderConfig(**eval_config.grading_config)

    if grading_config.use_batch_api and grading_config.base_url:
        raise ValueError("Batch API is only supported for OpenAI models!")

    llm = get_model(
        server_type='openai',
        base_url=grading_config.base_url,
        model=grading_config.judge_model,
    )
    prompt = Prompt(config=get_prompt_config('openai/math-answer-judge'))

    # assuming everything fits in memory for simplicity
    for jsonl_file in unroll_files(cfg.prediction_jsonl_files):
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]

        # TODO: should we try to judge both ways here as well? How to aggregate?
        if grading_config.skip_filled and all('judgement' in data_point for data_point in data):
            continue

        data_points = []

        if grading_config.use_batch_api:
            for data_point in data:
                # adding required fields for judgement prompt
                to_add = data_point.copy()
                to_add['predicted_answer'] = extract_answer(
                    data_point['generation'],
                    extract_from_boxed=eval_config.extract_from_boxed,
                    extract_regex=eval_config.extract_regex,
                )
                data_points.append(to_add)

            request_metadata = llm.batch_generate(
                prompts=[prompt.build_string(data_point) for data_point in data_points],
                tokens_to_generate=grading_config.tokens_to_generate,
            )
            # saving the request id to be able to retrieve results when they are ready
            with open(jsonl_file + '-batch-request-id', 'wt', encoding='utf-8') as fout:
                fout.write(json.dumps({'request_id': request_metadata.id}))
            LOG.info('Submitted batch evaluation request to OpenAI. Please wait for the results to be ready.')
            LOG.info('The current status and final results can be accessed through summarize_results.py')
            LOG.info('Request metadata: %s', str(request_metadata))
        else:
            output_file = jsonl_file + '-judgement'
            starting_idx = 0
            if grading_config.skip_filled:
                try:
                    with open(output_file, "rt", encoding="utf-8") as fin:
                        starting_idx = len(fin.readlines())
                except FileNotFoundError:
                    LOG.warning(f"File `{output_file}` not found, starting from scratch")
            data = data[starting_idx:]

            # saving to a tmp file to avoid corrupting original generation in case something goes wrong
            with open(
                output_file, "at" if grading_config.skip_filled else "wt", encoding="utf-8", buffering=1
            ) as fout:
                for data_point in tqdm(data, initial=starting_idx, total=len(data) + starting_idx):
                    # adding required fields for judgement prompt
                    to_add = data_point.copy()
                    to_add['predicted_answer'] = extract_answer(
                        data_point['generation'],
                        extract_from_boxed=eval_config.extract_from_boxed,
                        extract_regex=eval_config.extract_regex,
                    )
                    data_points.append(to_add)

                    if len(data_points) == grading_config.batch_size:
                        outputs = llm.generate(
                            prompts=[prompt.build_string(data_point) for data_point in data_points],
                            tokens_to_generate=grading_config.tokens_to_generate,
                        )
                        for output in outputs:
                            fout.write(json.dumps({'judgement': output['generation']}) + "\n")
                        data_points = []

                # collecting the final batch
                if len(data_points) > 0:
                    outputs = llm.generate(
                        prompts=[prompt.build_string(data_point) for data_point in data_points],
                        tokens_to_generate=grading_config.tokens_to_generate,
                    )
                    for output in outputs:
                        fout.write(json.dumps({'judgement': output['generation']}) + "\n")

            # fusing back into original file
            with open(jsonl_file, 'wt', encoding='utf-8') as fout, open(output_file, 'rt', encoding='utf-8') as fin:
                for data_point, judgement_line in zip(data, fin):
                    data_point.update(json.loads(judgement_line))
                    fout.write(json.dumps(data_point) + "\n")

            # removing judgement file
            Path(output_file).unlink()


def code_grader(cfg):
    # TODO: need to move it to a separate docker (either our sandbox or separate srun)
    from evalplus.evaluate import evaluate
    from omegaconf import OmegaConf

    from nemo_skills.evaluation.code_utils import preprocess_code

    # processing each generation separately (TODO: evalplus can do it together, but need to figure out the format)
    for jsonl_file in unroll_files(cfg.prediction_jsonl_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line)) for line in f]
        # all changes will be done with a new key "completion", so it's ok to write to the same file
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        eval_config = {
            "samples": jsonl_file,
            "base_only": False,
            "parallel": None,
            "i_just_wanna_run": False,
            "test_details": False,
            "min_time_limit": 1,
            "gt_time_limit_factor": 4.0,
            "mini": False,
            "noextreme": False,
            "version": "default",
        }
        eval_config.update(OmegaConf.to_container(cfg.eval_config))
        evaluate(Namespace(**eval_config))
        with open(jsonl_file[:-6] + '_eval_results.json', 'rt', encoding="utf-8") as fin:
            evalplus_grades = json.load(fin)
        # adding is_correct key to allow compute_metrics to work
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample['is_correct'] = evalplus_grades['eval'][sample['task_id']][0]['base_status'] == "pass"
                sample['is_correct-plus'] = (
                    sample['is_correct'] and evalplus_grades['eval'][sample['task_id']][0]['plus_status'] == "pass"
                )
                f.write(json.dumps(sample) + "\n")

        # moving eval file as otherwise evalplus does not want to recompute metrics if it's present..
        shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')


def if_grader(cfg):
    for jsonl_file in unroll_files(cfg.prediction_jsonl_files):
        parent_dir = Path(jsonl_file).absolute().parent
        cmd = (
            'cd /opt/benchmarks/google-research && python -m instruction_following_eval.evaluation_main '
            f'--input_data={jsonl_file} '
            f'--input_response_data={jsonl_file} '
            f'--output_dir={parent_dir} '
        )
        subprocess.run(cmd, shell=True, check=True)
        # fusing eval metrics back into the generation file
        with open(jsonl_file, "rt", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

        with open(parent_dir / 'eval_results_loose.jsonl', 'rt', encoding="utf-8") as f:
            eval_results = [json.loads(line) for line in f]
        for sample, eval_result in zip(samples, eval_results):
            sample['loose_eval'] = eval_result

        with open(parent_dir / 'eval_results_strict.jsonl', 'rt', encoding="utf-8") as f:
            eval_results = [json.loads(line) for line in f]
        for sample, eval_result in zip(samples, eval_results):
            sample['strict_eval'] = eval_result

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # removing metric files to avoid reusing them
        (parent_dir / 'eval_results_loose.jsonl').unlink()
        (parent_dir / 'eval_results_strict.jsonl').unlink()


def bfcl_grader(cfg):
    """Grader for Berkeley Function Calling Leaderboard."""
    eval_category = "all"

    for jsonl_file in unroll_files(cfg.prediction_jsonl_files):
        # Create the result and score folder
        # Right now calling the folder what Llama3-8B-Instruct would create but it is only used for extracting functions out of the output
        # TODO fix these hardcoded paths
        _REPO_ROOT_DIR = "/opt/benchmarks/gorilla/berkeley-function-call-leaderboard"
        _RESULT_DIR = path.join(_REPO_ROOT_DIR, "result/meta-llama_Meta-Llama-3-8B-Instruct")
        _SCORE_DIR = path.join(_REPO_ROOT_DIR, "score")

        cmd = f'mkdir -p {_RESULT_DIR} && mkdir -p {_SCORE_DIR}'
        subprocess.run(cmd, shell=True, check=True)

        output_dir = path.join(path.join(_REPO_ROOT_DIR, _RESULT_DIR))

        # Prepare the API file content
        if eval_category != "ast":
            api_keys_required = ["RAPID-API-KEY", "EXCHANGERATE-API-KEY", "OMDB-API-KEY", "GEOCODE-API-KEY"]
            api_keys = []
            try:
                api_keys = [{api_key: os.environ[api_key.replace("-", "_")]} for api_key in api_keys_required]
            except KeyError:
                raise SystemExit(f"Missing APIs, check environment variable - {api_keys_required}")

            API_FILE = path.join(_REPO_ROOT_DIR, "function_credential_config.json")
            with open(API_FILE, "w") as writer:
                json.dump(api_keys, writer)

            subprocess.run(
                f'cd {_REPO_ROOT_DIR} && python apply_function_credential_config.py', shell=True, check=True
            )

        # Read results and dump them in separate test-wise files in output_dir
        with open(jsonl_file, "rt", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
            test_category_dict = defaultdict(list)
            for sample in samples:
                sample["result"] = sample["generation"]
                test_category = sample["test_category"]
                test_category_dict[test_category].append(sample)

            for test_category, test_samples in test_category_dict.items():
                output_file = path.join(output_dir, f"gorilla_openfunctions_v1_test_{test_category}.json")
                with open(output_file, "w") as writer:
                    for test_sample in test_samples:
                        writer.write(json.dumps(test_sample) + "\n")

        # Run the evaluation
        # Allow for selective eval
        cmd = f'cd {path.join(_REPO_ROOT_DIR, "eval_checker")} && python eval_runner.py --model meta-llama/Meta-Llama-3-8B-Instruct --test {eval_category}'
        subprocess.run(cmd, shell=True, check=True)

        # Remove the output files and copy the score
        parent_dir = Path(jsonl_file).absolute().parent
        cmd = f'rm {_RESULT_DIR}/* && cp -r {_SCORE_DIR}/* {parent_dir}'
        subprocess.run(cmd, shell=True, check=True)


def arena_grader(cfg):
    from tqdm import tqdm

    from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config
    from nemo_skills.inference.server.model import get_model

    eval_config = LlmGraderConfig(**cfg.eval_config)
    assert eval_config.batch_size % 2 == 0  # required due to how everything is implement, can fix later

    if eval_config.use_batch_api and eval_config.base_url:
        raise ValueError("Batch API is only supported for OpenAI models!")

    llm = get_model(
        server_type='openai',
        base_url=eval_config.base_url,
        model=eval_config.judge_model,
    )
    prompt = Prompt(config=get_prompt_config('openai/arena-judge'))

    # assuming everything fits in memory for simplicity
    for jsonl_file in unroll_files(cfg.prediction_jsonl_files):
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]

        if eval_config.skip_filled and all(
            'judgement-gen-base' in data_point and 'judgement-base-gen' in data_point for data_point in data
        ):
            continue

        data_points = []

        if eval_config.use_batch_api:
            for data_point in data:
                # adding required fields for judgement prompt
                to_add = data_point.copy()
                to_add['answer_1'] = data_point['generation']
                to_add['answer_2'] = data_point['baseline_answer']
                data_points.append(to_add)
                # reversing the answers
                to_add = data_point.copy()
                to_add['answer_2'] = data_point['generation']
                to_add['answer_1'] = data_point['baseline_answer']
                data_points.append(to_add)

            request_metadata = llm.batch_generate(
                prompts=[prompt.build_string(data_point) for data_point in data_points],
                tokens_to_generate=eval_config.tokens_to_generate,
            )
            # saving the request id to be able to retrieve results when they are ready
            with open(jsonl_file + '-batch-request-id', 'wt', encoding='utf-8') as fout:
                fout.write(json.dumps({'request_id': request_metadata.id}))
            LOG.info('Submitted batch evaluation request to OpenAI. Please wait for the results to be ready.')
            LOG.info('The current status and final results can be accessed through summarize_results.py')
            LOG.info('Request metadata: %s', str(request_metadata))
        else:
            output_file = jsonl_file + '-judgement'
            starting_idx = 0
            if eval_config.skip_filled:
                try:
                    with open(output_file, "rt", encoding="utf-8") as fin:
                        starting_idx = len(fin.readlines())
                except FileNotFoundError:
                    LOG.warning(f"File `{output_file}` not found, starting from scratch")
            data = data[starting_idx:]

            # saving to a tmp file to avoid corrupting original generation in case something goes wrong
            with open(output_file, "at" if eval_config.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
                for data_point in tqdm(data, initial=starting_idx, total=len(data) + starting_idx):
                    # adding required fields for judgement prompt
                    to_add = data_point.copy()
                    to_add['answer_1'] = data_point['generation']
                    to_add['answer_2'] = data_point['baseline_answer']
                    to_add['judgement_mode'] = 'gen-base'
                    data_points.append(to_add)
                    # reversing the answers
                    to_add = data_point.copy()
                    to_add['answer_2'] = data_point['generation']
                    to_add['answer_1'] = data_point['baseline_answer']
                    to_add['judgement_mode'] = 'base-gen'
                    data_points.append(to_add)

                    if len(data_points) == eval_config.batch_size:
                        outputs = llm.generate(
                            prompts=[prompt.build_string(data_point) for data_point in data_points],
                            tokens_to_generate=eval_config.tokens_to_generate,
                        )
                        to_write = {}
                        for idx, (output, original_data_point) in enumerate(zip(outputs, data_points)):
                            to_write[f'judgement-{original_data_point["judgement_mode"]}'] = output['generation']
                            if idx % 2 != 0:
                                fout.write(json.dumps(to_write) + "\n")
                                to_write = {}
                        data_points = []

                # collecting the final batch
                if len(data_points) > 0:
                    outputs = llm.generate(
                        prompts=[prompt.build_string(data_point) for data_point in data_points],
                        tokens_to_generate=eval_config.tokens_to_generate,
                    )
                    to_write = {}
                    for idx, (output, original_data_point) in enumerate(zip(outputs, data_points)):
                        to_write[f'judgement-{original_data_point["judgement_mode"]}'] = output['generation']
                        if idx % 2 != 0:
                            fout.write(json.dumps(to_write) + "\n")
                            to_write = {}

            # fusing back into original file
            with open(jsonl_file, 'wt', encoding='utf-8') as fout, open(output_file, 'rt', encoding='utf-8') as fin:
                for data_point, judgement_line in zip(data, fin):
                    data_point.update(json.loads(judgement_line))
                    fout.write(json.dumps(data_point) + "\n")

            # removing judgement file
            Path(output_file).unlink()
