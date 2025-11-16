import json
import argparse
import re
# import ast

def parse_json_after_think(s: str):
    # 1) take everything after the first </think>
    if "</think>" not in s:
        raise ValueError("No </think> found in the generation.")

    before, after = s.split("</think>", 1)
    if "<think>" in before: # reasoning trace is everything between the first <think> and the last </think>
        before = before.split("<think>")[-1].strip()
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
            if line_num % 1000 == 0:
                print(f"Processing line {line_num}")

            if "context" in data: # vital field
                data["docs"] = data.pop("context")
            else:
                print(f"Warning: Could not parse context for line {line_num}")
                continue

            if "question" in data: # not a vital field
                data["question0"] = data.pop("question")

            if "new_question_aug_qwen3_235b" in data: # vital field
                data["question"] = data.pop("new_question_aug_qwen3_235b")
                if "question:\n\n" in data["question"] and not data["question"].endswith("question:\n\n"):
                    data["question"] = data["question"].split("question:\n\n")[-1].strip()
                elif "**Question:**  \n" in data["question"] and not data["question"].endswith("**Question:**  \n"):
                    data["question"] = data["question"].split("**Question:**  \n")[-1].strip()

                elif "challenging question" in data["question"]: # the question starts after ":\n\n" but not if it is at the end of the string
                    idx_of_colon = data["question"].find(":\n\n", data["question"].find("challenging question"))
                    if idx_of_colon  != -1 and idx_of_colon != len(data["question"]) - 2:
                        data["question"] = data["question"].split(":\n\n", idx_of_colon)[-1].strip()

            else:
                print(f"Warning: Could not parse question for line {line_num}")
                continue
            
            if "answer" in data: # not a vital field
                data["answer0"] = data.pop("answer")
            if "reasoning_trace" in data: # not a vital field
                data["reasoning0"] = data.pop("reasoning_trace")
            
            generation = data.get("new_question_cot_answer", "").strip() # we extract the reasoning trace and answer from the generation field
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
