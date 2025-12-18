import json
import argparse
from tqdm import tqdm

# adapted from /lustre/fsw/portfolios/llmservice/users/skriman/lgen/scripts/convert_conv_to_prompt.py

# this will need to be refactored so it can take the oai format.
# we read the input line by line  and each line becomes a json object.
# there are only fields, "metadata" and "messages".
# we look at the metadata.all_turns_token_count.tool_calls field. if it is not zero, we skip the line.
# next, we iterate over the messages list. we need to check the "role" field. we see at least one message with role "system" and content that is not "", we need to discard this sample as well.


def convert_to_prompt_format(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in):
            convo = json.loads(line)
            conversation_text = []
            for message in convo["conversations"]:
                conversation_text.append(f"{message['from']}: {message['value']}")
            full_conversation = "\n".join(conversation_text)
            output_data = {"prompt": full_conversation}
            f_out.write(json.dumps(output_data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert conversation JSONL to prompt format.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file.')
    parser.add_argument('output_file', type=str, help='Path to the output JSONL file.')
    args = parser.parse_args()
    convert_to_prompt_format(args.input_file, args.output_file)