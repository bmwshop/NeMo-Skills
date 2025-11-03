from nemo_skills.pipeline.cli import generate, wrap_arguments

cluster = "hsg"
# input_file = "/nemo_run/code/recipes/long_context_sdg/src/test.jsonl"
input_file = "/workspace/DATA/LONG_CONTEXT/ANNOTATED_TOPIC_TEST/lcr.jsonl"
output_dir = "/workspace/DATA/LONG_CONTEXT/ANNOTATED_TOPIC_TEST"
prompt_config = "/nemo_run/code/recipes/long_context_sdg/prompts/annotate_document_topic.yaml"


generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++chat_template_kwargs.reasoning_effort=high "
        f"++max_concurrent_requests=512 "
        f"++inference.tokens_to_generate=8000 "
        f"++inference.endpoint_type=text "
        f"++inference.temperature=1.0 "
        f"++inference.top_p=1.0 "
    ),
    cluster=cluster,
    input_file=input_file,
    output_dir=output_dir,
    model="/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507",
    server_type="vllm",
    postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
    # Server parameters
    # num_chunks=10,
    server_gpus=4,
    server_nodes=1,
    server_args="--async-scheduling"
)
