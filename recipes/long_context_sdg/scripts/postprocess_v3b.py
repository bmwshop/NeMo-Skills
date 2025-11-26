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

    # 2) strip optional ```json ... ``` fences
    after = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", after, flags=re.DOTALL)

    # 3) find the first JSON object with the standard decoder (avoids brittle regex {…} grabs)
    dec = json.JSONDecoder()
    # locate first '{'
    i = after.find("{")
    if i == -1:
        raise ValueError("No JSON object start found in sample")
    obj, end = dec.raw_decode(after[i:])
    if not all(k in obj for k in REQUIRED):
        print(f"Warning: Missing required fields in the generation: {obj.keys()}")
        return None

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
