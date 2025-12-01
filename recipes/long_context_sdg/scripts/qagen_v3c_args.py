import argparse
from nemo_skills.pipeline.cli import generate, wrap_arguments
import os


def get_category(archetype):
    if archetype in range(6, 69):
        return "Company_Documents"
    elif archetype in range(1, 6):
        return "Academia"
    elif archetype in range(69, 80):
        return "Government_Consultations"
    elif archetype in range(80, 86):
        return "Legal"
    elif archetype in range(86, 94):
        return "Industry_Reports"
    elif archetype in range(94, 100):
        return "Marketing"
    elif archetype == 100:
        return "Survey_Reports"
    else:
        return None
# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate QA pairs with specified archetype')
parser.add_argument('--archetype', type=int, help='Archetype number', required=True)
parser.add_argument('--category', type=str, help='Category of input file', required=True)
parser.add_argument('--servers', type=int, help='Number of servers', default=1)
parser.add_argument('--jobs', type=int, help='Number of jobs', default=3)
parser.add_argument('--concurrent_requests', type=int, help='Number of concurrent requests', default=16)
parser.add_argument('--data_chunk_size', type=int, help='Size of data chunk', default=16384)
parser.add_argument('--data_maxlength', type=int, help='Maximum length of data', default=130000)
parser.add_argument('--tokens_to_generate', type=int, help='Number of tokens to generate', default=32768)
args = parser.parse_args()

MAX_SEQ_LEN = 524288
TEMPERATURE = 0.3
TOP_P = 0.9

archetype = args.archetype
template = str(archetype)

category = args.category


if category == "sec":
    assert args.archetype in range(6, 69), "Invalid archetype number for sec category; only company archetypes are supported"
    input_file = f"/workspace/DATA/lcr_docs/simple/sec/sec-{args.data_chunk_size}-{args.data_maxlength}.simple.jsonl"
    output_dir = f"/workspace/DATA/lc/lcrqagen_v3c/sec/{template}-{args.data_chunk_size}-{args.data_maxlength}"
else:
    # we compute the category based on the archetype number
    category = get_category(args.archetype)
    assert category is not None, f"Archetype {args.archetype} out of range"
    if args.data_maxlength == 130000:
        prefix = "128k"
    elif args.data_maxlength == 31000:
        prefix = "32k"
    else:
        raise ValueError(f"Invalid data_maxlength: {args.data_maxlength}")
    input_file = f"/workspace/DATA/lcr_docs/simple/{prefix}/lcr-{category}-{args.data_chunk_size}-{args.data_maxlength}.simple.jsonl"
    output_dir = f"/workspace/DATA/lc/lcrqagen_v3c/{template}-{args.data_chunk_size}-{args.data_maxlength}"

cluster = "hsg"
# input_file = "/workspace/DATA/lcr_docs/acgilms-16384-130000.jsonl" # all v3 files
# input_file = "/workspace/DATA/lc/lcrqagen_v4/sec-16384-130000.jsonl" # just the sec
# output_dir = f"/workspace/DATA/lc/lcrqagen_v3a/sec/{template}"

prompt_config = f"/nemo_run/code/recipes/long_context_sdg/prompts/v3c/{template}.yaml"
# teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Instruct-2507"
teacher_model = "/hf_models/Qwen_Qwen3-235B-A22B-Thinking-2507"

# if not os.path.exists(prompt_config):
#     # raise FileNotFoundError(f"Prompt configuration file not found: {prompt_config}")
#     print(f"Prompt configuration file not found: {prompt_config}")
#     exit(1)

dependent_jobs = args.jobs
num_servers = args.servers
concurrent_requests = args.concurrent_requests

#  f"++max_concurrent_requests=512 "
#  server_type="vllm",

generate(
    ctx=wrap_arguments(
        f"++skip_filled=True "
        f"++prompt_config={prompt_config} "
        f"++inference.tokens_to_generate={args.tokens_to_generate} "
        f"++inference.endpoint_type=text "
        f"++max_concurrent_requests={concurrent_requests} "
        f"++inference.temperature={TEMPERATURE} "
        f"++inference.top_p={TOP_P} "
    ),
    cluster=cluster,
    input_file=input_file,
    output_dir=output_dir,
    model=teacher_model,
    server_type="sglang",
    server_args=f"--context-len {MAX_SEQ_LEN} --ep-size 8",
    dependent_jobs=dependent_jobs,
    postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_v3c.py --input {output_dir}/output.jsonl --output {output_dir}/qa.jsonl",
    # Server parameters
    num_chunks=num_servers,
    server_gpus=4,
    server_nodes=2,
)
# dependent_jobs=dependent_jobs,
# server_args="--async-scheduling"
# postprocess_cmd=f"python /nemo_run/code/recipes/long_context_sdg/scripts/postprocess_domain_gens.py {output_dir}/output.jsonl {output_dir}/annotated_topics.jsonl",
