from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "hsg"
# input_file = "/nemo_run/code/recipes/long_context_sdg/src/test.jsonl"
input_file = "/workspace/DATA/lc/lcrqagen_v3/g/69/intermediate_p.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v3/g/69/traces"
prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/gen_trace.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

dependent_jobs = 1
num_servers = 32

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",

generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++chat_template_kwargs.reasoning_effort=high "
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
    server_gpus=4,
    server_nodes=2,
)

# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
