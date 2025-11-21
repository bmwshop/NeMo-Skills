import argparse
from nemo_skills.pipeline.cli import generate, wrap_arguments
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate QA pairs with specified template')
parser.add_argument('--template', type=str, help='Template configuration to use', required=True)
args = parser.parse_args()

template = args.template

cluster = "hsg"
# input_file = "/nemo_run/code/recipes/long_context_sdg/src/test.jsonl"
input_file = "/workspace/DATA/lcr_docs/acgilms-16384-130000.jsonl" # all v3 files
output_dir = f"/workspace/DATA/lc/lcrqagen_v3a/{template}"
prompt_config = f"/nemo_run/code/recipes/long_context_sdg/prompts/v3a/gen_qa_{template}.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

if not os.path.exists(prompt_config):
    # raise FileNotFoundError(f"Prompt configuration file not found: {prompt_config}")
    print(f"Prompt configuration file not found: {prompt_config}")
    exit(1)

dependent_jobs = 2
num_servers = 1

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",

generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.tokens_to_generate=32768 "
        f"++inference.endpoint_type=text "
        f"++max_concurrent_requests=8 "
        f"++inference.temperature=0.3 "
        f"++inference.top_p=0.9 "
    ),
    cluster=cluster,
    input_file=input_file,
    output_dir=output_dir,
    model=teacher_model,
    server_type="sglang",
    server_args="--context-len 262144 --ep-size 8",
    dependent_jobs=dependent_jobs,
    postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_qa.py {output_dir}/output.jsonl {output_dir}/qa.jsonl",
    # Server parameters
    num_chunks=num_servers,
    server_gpus=4,
    server_nodes=2,
)
# dependent_jobs=dependent_jobs,
# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
