import json
import argparse
import re
# import ast

def parse_json_after_think(s: str):
    # 1) take everything after the first </think>
    if "</think>" not in s:
        raise ValueError("No </think> found in the generation.")

    before, after = s.split("</think>", 1)
    after = after.strip()
    before = before.strip()

    obj = {} # the new reasoning trace and answer
    obj["reasoning_trace"] = before
    obj["answer"] = after
    return obj

def extract_generation_fields(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            try:
                line = line.strip()
            except MemoryError:
                print(f"Skipping oversized line {line_num} (MemoryError)")
                continue
                
            if not line:
                continue
            data = json.loads(line)
            
            if "answer" in data: # keep the original answer for reference
                data["answer0"] = data.pop("answer")
            if "reasoning_trace" in data: # keep the original reasoning trace for reference
                data["reasoning0"] = data.pop("reasoning_trace")
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reasoning trace and answer from generation field in JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    extract_generation_fields(args.input, args.output)
