from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "hsg"
tokens_to_generate = 65536

# input_file = "/workspace/DATA/lc/lcrqagen_v3/g/76/traces_b/trace.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_v3/g/76/traces_b/critiques"

input_file = "/workspace/DATA/lc/lcrqagen_v3/g/76/traces/trace.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v3/g/76/traces/critiques"

input_file = "/workspace/DATA/lc/lcrqagen_v3/g/69/traces/trace.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v3/g/69/traces/critiques"

input_file = "/workspace/DATA/lc/lcrqagen_v3/g/69/traces_b/trace.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v3/g/69/traces_b/critiques"

# input_file = "/workspace/DATA/lc/lcrqagen_v3/c/tr/traces_b/trace.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_v3/c/tr/traces_b/critiques"

# input_file = "/workspace/DATA/lc/lcrqagen_v4/sec/general/i.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_v4/sec/general/critiques"

# input_file = "/workspace/DATA/lc/lcrqagen_v3/c/tr/intermediate_p.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_v3/c/tr/traces_b"

# input_file = "/workspace/DATA/lc/lcrqagen_v2/Government_Consultations-16384-130000_qa.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_v2/traces"

# input_file = "/workspace/DATA/lc/lcrqagen_a_v2/intermediate_p.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_a_v2/traces"

# prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/gen_trace.yaml"
prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/critique.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

dependent_jobs = 1
num_servers = 16

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",
# f"++chat_template_kwargs.reasoning_effort=high "
generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.tokens_to_generate={tokens_to_generate} "
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
    server_gpus=4,
    server_nodes=2,
)

# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_trace.py {output_dir}/output.jsonl {output_dir}/trace.jsonl",

# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
