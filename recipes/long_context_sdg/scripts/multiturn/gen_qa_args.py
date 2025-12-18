from nemo_skills.pipeline.cli import generate, wrap_arguments
import argparse

cluster = "hsg"
MAX_SEQ_LEN = 524288
# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate QA pairs with specified template')

parser.add_argument('--input_split', type=int, help='Split number of the input file to process', required=True)
parser.add_argument('--servers', type=int, help='Number of servers', default=1)
parser.add_argument('--jobs', type=int, help='Number of jobs', default=3)
parser.add_argument('--concurrent_requests', type=int, help='Number of concurrent requests', default=16)
parser.add_argument('--tokens_to_generate', type=int, help='Number of tokens to generate', default=16384)

args = parser.parse_args()

split_number = args.input_split

# input_file = f"/workspace/DATA/lc/multiturn/nickel-capybara_noncommercial.shuf.prompts.jsonl"

input_file = f"/workspace/DATA/lc/multiturn/nc_{split_number}_prompt.jsonl"
output_dir = f"/workspace/DATA/lc/multiturn/synth_turns_{split_number}"



# prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/gen_trace.yaml"
prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/multiturn/orig.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

dependent_jobs = args.jobs
num_servers = args.servers
concurrent_requests = args.concurrent_requests

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",
# f"++chat_template_kwargs.reasoning_effort=high "
generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.tokens_to_generate={args.tokens_to_generate} "
        f"++inference.endpoint_type=text "
        f"++max_concurrent_requests={concurrent_requests} "
        f"++inference.temperature=0.3 "
        f"++inference.top_p=0.9 "
    ),
    cluster=cluster,
    input_file=input_file,
    output_dir=output_dir,
    model=teacher_model,
    server_type="sglang",
    server_args=f"--context-len {MAX_SEQ_LEN} --ep-size 8",
    num_chunks=num_servers,
    dependent_jobs=dependent_jobs,
    server_gpus=4,
    server_nodes=2,
)

# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
