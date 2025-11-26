#!/usr/bin/env python3
import sys, json, shutil

REQUIRED = {
    "question": "",
    "answer": "",
    "reasoning_trace": "",
    "used_docs": [],
}

inp = sys.argv[1]
bak = inp + ".bak"

# make backup
shutil.copy(inp, bak)

# rewrite original file
with open(bak, encoding="utf-8") as f, open(inp, "w", encoding="utf-8") as g:
    for line in f:
        obj = json.loads(line)
        for k, v in REQUIRED.items():
            obj.setdefault(k, v)
        g.write(json.dumps(obj) + "\n")

print(f"updated in place, backup saved as {bak}")

