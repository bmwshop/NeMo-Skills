import re
import json
import sys
import logging
import traceback
import argparse

# adapted from /lustre/fsw/portfolios/llmservice/users/skriman/lgen/scripts/convert_summ_qa_2.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)

def remove_special_prefix(text):
    text = text.strip()
    # Remove a bracketed prefix, e.g., "<...>", "[...]", or "(...)", optionally followed by a colon.
    bracket_pattern = r'^[<\[\(]([^>\]\)]+)[>\]\)]\s*:?\s*'
    match = re.match(bracket_pattern, text)
    if match and len(match.group(1)) < 40:
        text = text[len(match.group(0)):]
        return text.strip()
    # Remove any prefix ending with a colon if it's less than 40 characters.
    colon_pattern = r'^([^:]+):\s*'
    match = re.match(colon_pattern, text)
    if match and len(match.group(1)) < 40:
        text = text[len(match.group(0)):]
    return text.strip()

def convert_string_to_training_data(s):
    # Splits the string into turns by matching "User:" or "Assistant:" prefixes.
    pattern = r'(User|Assistant): (.*?)(?=(?:User|Assistant):|$)'
    matches = re.findall(pattern, s, flags=re.DOTALL | re.IGNORECASE)
    conversations = []
    for speaker, text in matches:
        conversations.append({
            "from": speaker,
            "value": text.strip()
        })
    return {
        "system": "",
        "mask": "User",
        "conversations": conversations
    }

def convert_line_to_jsonl(line):
    data = json.loads(line)
    original_conversation = convert_string_to_training_data(data["prompt"])
    generation = data["generation"]
    if "</think>" not in generation:
        logging.warning("No </think> found in the generation.")
        return None

    post_think = generation.split("</think>", 1)[-1]
    raw_segments = post_think.split("~~~~~~~~~~")
    segments = []
    for seg in raw_segments:
        cleaned = remove_special_prefix(seg.strip())
        if cleaned:
            segments.append(cleaned)
    # Skip line if number of segments (turns) is not even.
    if len(segments) % 2 != 0:
        logging.warning("Number of segments is not even.")
        return None
    referencing_turns = []
    # Assume the first (non-empty) segment is User, then Assistant, and so on.
    for i in range(0, len(segments), 2):
        user_turn = segments[i]
        assistant_turn = segments[i+1]
        if not user_turn or not assistant_turn:
            return None
        referencing_turns.append([
            {"from": "User", "value": user_turn},
            {"from": "Assistant", "value": assistant_turn}
        ])
    output = {
        "original_conversation": original_conversation,
        "referencing_turns": referencing_turns
    }
    return output

def process_file(input_filename, output_filename, max_lines=None):
    line_count = 0
    written_count = 0
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                converted = convert_line_to_jsonl(line)
                if converted is None:
                    continue
                out_str = json.dumps(converted, ensure_ascii=False)
                try:
                    json.loads(out_str)
                except Exception as e:
                    logging.error("Output line is not valid JSON (line %d): %s", line_count, out_str)
                    continue
                outfile.write(out_str + "\n")
                written_count += 1
            except Exception as e:
                logging.error("Error processing line %d: %s", line_count, str(e))
                logging.error("Line content: %s", line)
                logging.error(traceback.format_exc())
                continue
            if max_lines is not None and line_count >= max_lines:
                logging.info("Reached maximum lines to process: %d", max_lines)
                break

    logging.info(f"Wrote {written_count} lines out of {line_count} inputs to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert output to blend format.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file.')
    parser.add_argument('output_file', type=str, help='Path to the output JSONL file.')
    args = parser.parse_args()
    process_file(args.input_file, args.output_file)
