#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fill a meta-template with archetype/task pairs from a JSONL file."
    )
    parser.add_argument(
        "--meta_template",
        help="Path to the meta-template YAML file with {archetype} and {task} placeholders.",
        default="general.yaml",
    )
    parser.add_argument(
        "--jsonl",
        help='Path to JSONL file with fields "index", "archetype", and "task" on each line.',
        default="aalcr_v2.jsonl",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="..",
        help="Directory to write per-index YAML files (default: ..).",
    )
    args = parser.parse_args()

    meta_template_path = Path(args.meta_template)
    jsonl_path = Path(args.jsonl)
    outdir = Path(args.outdir)

    # Read the meta-template once
    meta_template_text = meta_template_path.read_text(encoding="utf-8")

    # Ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e

            # Required fields: index, archetype, instruction
            try:
                idx = str(record["index"])
                archetype = record["archetype"]
                task = record["task"]
                sample_questions = '\n  - '.join(record["sample_questions"])
            except KeyError as missing:
                raise RuntimeError(
                    f"Missing required field {missing} in JSON on line {line_no}"
                )

            # Fill the meta-template
            filled = meta_template_text.format(
                archetype=archetype,
                task=task,
                sample_questions=sample_questions,
            )

            # Write to index.yaml
            out_path = outdir / f"{idx}.yaml"
            out_path.write_text(filled, encoding="utf-8")

            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
