from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "hsg"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate QA pairs with specified template')
parser.add_argument('--template', type=str, help='Template configuration to use', required=True)
parser.add_argument('--category', type=str, help='Category of input file', required=True)
args = parser.parse_args()

template = args.template
category = args.category
if category == "sec":
    input_file = f"/workspace/DATA/lc/lcrqagen_v3a/sec/{template}/intermediate_p.jsonl"
    output_dir = f"/workspace/DATA/lc/lcrqagen_v3a/sec/{template}/traces"
else:
    input_file = f"/workspace/DATA/lc/lcrqagen_v3a/{template}/intermediate_p.jsonl"
    output_dir = f"/workspace/DATA/lc/lcrqagen_v3a/{template}/traces"


# prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/gen_trace.yaml"
prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/gen_trace_1a.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

dependent_jobs = 1
num_servers = 32

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",
# f"++chat_template_kwargs.reasoning_effort=high "
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
    num_chunks=num_servers,
    dependent_jobs=dependent_jobs,
    postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_trace.py --input {output_dir}/output.jsonl --output {output_dir}/trace.jsonl",
    server_gpus=4,
    server_nodes=2,
)

# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
