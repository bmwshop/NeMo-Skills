import json
import argparse
import re
# import ast

def parse_json_after_think(s: str):
    # 1) take everything after the first </think>
    if "</think>" not in s:
        raise ValueError("No </think> found in the generation.")

    after = s.split("</think>", 1)[-1].strip()

    # 2) strip optional ```json ... ``` fences
    after = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", after, flags=re.DOTALL)

    # 3) find the first JSON object with the standard decoder (avoids brittle regex {…} grabs)
    dec = json.JSONDecoder()
    # locate first '{'
    i = after.find("{")
    if i == -1:
        raise ValueError("No JSON object start found after </think>.")
    obj, end = dec.raw_decode(after[i:])
    return obj

def extract_generation_fields(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract domain and subtopics from generation field in JSONL.")
    parser.add_argument("input_file", type=str, help="Input JSONL file path")
    parser.add_argument("output_file", type=str, help="Output JSONL file path")
    args = parser.parse_args()

    extract_generation_fields(args.input_file, args.output_file)
