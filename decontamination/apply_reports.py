#!/usr/bin/env python3
# decontamination/apply_reports.py
"""
Apply decontamination.py's contamination reports to the original JSONL.

decontamination.py saves its filtered output as an Arrow dataset with two
extra columns (conversation_id, prompt_token_ids), which is not what axolotl
reads. This drops the rows its reports name from the *original* JSONL and
copies every other line byte for byte, so the schema is unchanged.

python3 decontamination/apply_reports.py \
    --input data/fixed/gpt_oss_8retries_v3_fixed_no_medexqa.jsonl \
    --reports <decontam out>/gpt_oss_8retries_v3_fixed_no_medexqa/contamination_reports \
    --output data/decontaminated/gpt_oss_8retries_v3_fixed_no_medexqa_decontam.jsonl

Rows carry no conversation_id, so decontamination.py names them
auto_id_<row index>; the index is the line number among non-empty lines,
which is the order load_dataset("json") reads them in.

writes <output>                    the kept rows
       <output>.removed.jsonl      one line per removed row: its index, every
                                   benchmark (and benchmark row) it matched,
                                   its source_path if it has one, its prompt
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SUFFIX = "__contamination_report.json"

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--reports", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
args = p.parse_args()

hits = defaultdict(dict)             # row index -> {benchmark: benchmark row}
per_bench = Counter()
reports = sorted(args.reports.glob("*" + SUFFIX))
if not reports:
    raise SystemExit(f"no *{SUFFIX} in {args.reports}")
for rep in reports:
    bench = rep.name[:-len(SUFFIX)]
    for conv_id, eval_idx in json.loads(rep.read_text()).items():
        if not conv_id.startswith("auto_id_"):
            raise SystemExit(f"{rep}: id {conv_id!r} is not auto_id_<n>; "
                             "the input has its own conversation_id column")
        hits[int(conv_id[len("auto_id_"):])][bench] = eval_idx
        per_bench[bench] += 1

args.output.parent.mkdir(parents=True, exist_ok=True)
removed_path = Path(str(args.output) + ".removed.jsonl")
n = kept = 0
by_source = Counter()
with open(args.input, encoding="utf-8") as fin, \
        open(args.output, "w", encoding="utf-8") as fout, \
        open(removed_path, "w", encoding="utf-8") as frem:
    for line in fin:
        if not line.strip():
            continue
        idx, n = n, n + 1
        if idx not in hits:
            fout.write(line)
            kept += 1
            continue
        row = json.loads(line)
        src = row.get("source_path") or ""
        by_source[src] += 1
        prompt = next((m["value"] for m in row.get("conversations", [])
                       if m.get("from") in ("human", "user")), "")
        frem.write(json.dumps({"index": idx, "matches": hits[idx], "source_path": src,
                               "prompt": prompt}, ensure_ascii=False) + "\n")

if max(hits, default=-1) >= n:
    raise SystemExit(f"report names row {max(hits)} but {args.input} has {n} rows: "
                     "reports are from a different file")
print(f"{args.input}: {n} rows, removed {n - kept}, kept {kept} -> {args.output}")
for bench, c in per_bench.most_common():
    print(f"  {c:7d}  {bench}")
if any(by_source):
    print("removed rows by source_path:")
    for src, c in by_source.most_common():
        print(f"  {c:7d}  {src}")
