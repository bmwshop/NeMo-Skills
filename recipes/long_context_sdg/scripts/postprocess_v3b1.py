import json
import argparse
import re
# import ast

REQUIRED = ["question", "answer", "reasoning_trace", "facts_used", "used_docs"]


def parse_json_after_think(s: str):
    # 1) take everything after the first </think>
    if "</think>" not in s:
        print("Warning: No </think> found in the generation.")
        after = s.strip()
    else:
        after = s.split("</think>", 1)[-1].strip()

    # 2) Parse the structured text format
    obj = {}
    
    # Extract QUESTION section
    question_match = re.search(r'QUESTION:\s*(.*?)(?=\n\s*ANSWER:|\n\s*FACTS_USED:|\n\s*REASONING_TRACE:|\n\s*USED_DOCS:|$)', after, re.DOTALL)
    if question_match:
        obj["question"] = question_match.group(1).strip()
    
    # Extract ANSWER section
    answer_match = re.search(r'ANSWER:\s*(.*?)(?=\n\s*FACTS_USED:|\n\s*REASONING_TRACE:|\n\s*USED_DOCS:|$)', after, re.DOTALL)
    if answer_match:
        obj["answer"] = answer_match.group(1).strip()
    
    # Extract FACTS_USED section
    facts_match = re.search(r'FACTS_USED:\s*(.*?)(?=\n\s*REASONING_TRACE:|\n\s*USED_DOCS:|$)', after, re.DOTALL)
    if facts_match:
        facts_text = facts_match.group(1).strip()
        # Parse facts into a list of strings (F1 (D#): fact, F2 (D#): fact, ...)
        facts_lines = [line.strip() for line in facts_text.split('\n') if line.strip()]
        obj["facts_used"] = facts_lines
    
    # Extract REASONING_TRACE section
    reasoning_match = re.search(r'REASONING_TRACE:\s*(.*?)(?=\n\s*USED_DOCS:|$)', after, re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        # Parse reasoning steps into a list (1) ..., 2) ..., ...)
        reasoning_lines = [line.strip() for line in reasoning_text.split('\n') if line.strip()]
        obj["reasoning_trace"] = reasoning_lines
    
    # Extract USED_DOCS section
    used_docs_match = re.search(r'USED_DOCS:\s*(.*?)$', after, re.DOTALL)
    if used_docs_match:
        used_docs_text = used_docs_match.group(1).strip()
        # Parse comma-separated document IDs
        # Remove parenthetical comment if present
        used_docs_text = re.sub(r'\(.*?\)', '', used_docs_text)
        used_docs = [doc.strip() for doc in used_docs_text.split(',') if doc.strip()]
        obj["used_docs"] = used_docs
    
    # Check for required fields
    if not all(k in obj for k in REQUIRED):
        print(f"Warning: Missing required fields in the generation. Found: {obj.keys()}")
        return None

    # Convert answer to string if needed
    if isinstance(obj["answer"], int):
        obj["answer"] = str(obj["answer"])
    elif isinstance(obj["answer"], float):
        obj["answer"] = str(obj["answer"])
    elif isinstance(obj["answer"], list):
        obj["answer"] = ','.join(str(x) for x in obj["answer"])
    
    return obj

def extract_generation_fields(input_file, output_file):
    num_good = 0
    num_lines = 0
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            num_lines += 1
            try:
                line = line.strip()
            except MemoryError:
                print(f"Skipping oversized line {line_num} (MemoryError)")
                continue

            if not line:
                continue
            data = json.loads(line)
            generation = data.get("generation", "").strip()
            if generation:
                try:
                    # Safely parse the generation string to a dict
                    # gen_dict = ast.literal_eval(generation.split("<|channel|>final<|message|>")[-1].strip())
                    gen_dict = parse_json_after_think(generation)
                    if gen_dict is not None:
                        data |= gen_dict
                except Exception as e:
                    print(f"Warning: Could not parse generation for line {line_num}: {line_num}\nError: {e}")
                    gen_dict = None
            else:
                gen_dict = None
            
            # f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            if gen_dict is not None:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                num_good += 1
    return num_good, num_lines
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract answer and reasoning trace from generation field in JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    num_good, num_lines = extract_generation_fields(args.input, args.output)
    print(f"Extracted {num_good} good generations out of {num_lines} lines")
