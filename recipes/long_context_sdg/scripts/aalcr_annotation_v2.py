from nemo_skills.pipeline.cli import generate, wrap_arguments
import argparse

parser = argparse.ArgumentParser(description='Generate QA pairs with specified template')
parser.add_argument('--experiment_suffix', type=str, help='Experiment suffix', default="")
parser.add_argument('--servers', type=int, help='Number of servers', default=10)
args = parser.parse_args()

cluster = "hsg"
# input_file = "/nemo_run/code/recipes/long_context_sdg/src/test.jsonl"
input_file = "/workspace/DATA/lc/lcrdocq_topic_annotation/lcrdocq.jsonl"
output_dir = f"/workspace/DATA/lc/aalcr_v2_{args.experiment_suffix}"
prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/annotate_aalcr_v2.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

num_servers = args.servers
#  f"++max_concurrent_requests=512 "
#  server_type="vllm",

generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.tokens_to_generate=131072 "
        f"++inference.endpoint_type=text "
        f"++max_concurrent_requests=16 "
        f"++inference.temperature=0.6 "
        f"++inference.top_p=0.9 "
    ),
    cluster=cluster,
    input_file=input_file,
    output_dir=output_dir,
    model=teacher_model,
    server_type="sglang",
    server_args="--context-len 262144 --ep-size 8",
    postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_aalcr.py {output_dir}/output.jsonl {output_dir}/aalcr_v2.jsonl",
    # Server parameters
    num_chunks=num_servers,
    server_gpus=4,
    server_nodes=2,
)

# server_args="--async-scheduling"
