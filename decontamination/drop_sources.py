#!/usr/bin/env python3
# decontamination/drop_sources.py
"""
Drop every row of a training JSONL whose source_path contains a given string,
e.g. the MedExpQA rows of CuratedQA, whose licence is not fully permissive:

python3 decontamination/drop_sources.py \
    --input data/fixed/gpt_oss_8retries_v3_fixed.jsonl \
    --output data/fixed/gpt_oss_8retries_v3_fixed_no_medexqa.jsonl \
    --source /medexpqa/

Kept lines are copied byte for byte, so their schema is unchanged.
"""

import argparse
import json
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--source", action="append", required=True,
               help="substring of source_path to drop; repeatable")
args = p.parse_args()

kept, dropped = 0, Counter()
with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        src = json.loads(line).get("source_path") or ""
        if any(s in src for s in args.source):
            dropped[src] += 1
        else:
            fout.write(line)
            kept += 1

print(f"kept {kept}, dropped {sum(dropped.values())} -> {args.output}")
for src, n in dropped.most_common():
    print(f"  {n:7d}  {src}")
if not dropped:
    raise SystemExit("nothing matched --source; check the spelling")
