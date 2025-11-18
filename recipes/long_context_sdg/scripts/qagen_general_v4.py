from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "hsg"
# input_file = "/nemo_run/code/recipes/long_context_sdg/src/test.jsonl"
input_file = "/workspace/DATA/lcr_docs/Academia-16384-130000.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v4/a/general"

input_file = "/workspace/DATA/lcr_docs/Legal-16384-130000.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v4/l/general"

input_file = "/workspace/DATA/lcr_docs/Marketing-16384-130000.jsonl"
output_dir = "/workspace/DATA/lc/lcrqagen_v4/m/general"

# input_file = "/workspace/DATA/lcr_docs/Industry_Reports-16384-130000.jsonl"
# output_dir = "/workspace/DATA/lc/lcrqagen_v4/i/general"

# input_file = "/workspace/DATA/lcr_docs/Survey_Reports-16384-130000.jsonl"
#output_dir = "/workspace/DATA/lc/lcrqagen_v4/s/general"

#input_file = "/workspace/DATA/lcr_docs/Company_Documents-16384-130000.jsonl"
#output_dir = "/workspace/DATA/lc/lcrqagen_v4/c/general"

# input_file = "/workspace/DATA/lcr_docs/Government_Consultations-16384-130000.jsonl"
#output_dir = "/workspace/DATA/lc/lcrqagen_v4/g/general"


prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/gen_qa_general_v4.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"
dependent_jobs = 1

num_servers = 16

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",
#         f"++chat_template_kwargs.reasoning_effort=high "
generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.tokens_to_generate=16384 "
        f"++inference.endpoint_type=text "
        f"++max_concurrent_requests=8 "
        f"++inference.temperature=0.6 "
        f"++inference.top_p=0.9 "
    ),
    cluster=cluster,
    input_file=input_file,
    output_dir=output_dir,
    model=teacher_model,
    server_type="sglang",
    dependent_jobs=dependent_jobs,
    server_args="--context-len 262144 --ep-size 8",
    postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_qa.py {output_dir}/output.jsonl {output_dir}/qa.jsonl",
    # Server parameters
    num_chunks=num_servers,
    server_gpus=4,
    server_nodes=2,
)
# dependent_jobs=dependent_jobs,
# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
