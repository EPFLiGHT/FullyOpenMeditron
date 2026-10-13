#!/usr/bin/env python3
"""Sample audit items from the MeditronFO synthetic training sets.

Writes samples.json next to this script, ready to commit to the HF Static Space.
Run from a machine that can read the JSONL paths below.
"""

import json
import random
from pathlib import Path

SEED = 2001
N_PER_SUBSET = 20
OUT = Path(__file__).parent / "samples.json"

SOURCES = {
    "Synthetic QA": "data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl",
    "Guidelines QA": "data/guidelines_qa/guidelines_qa_v3_full.jsonl",
    "Synthetic MOOVE": "data/synthetic_moove/synthetic_moove_v2_distillation_1try.jsonl",
}


def extract(conv):
    """First user turn and first assistant turn following it."""
    q = a = ""
    for m in conv:
        role, val = m.get("from"), m.get("value", "")
        if role in ("user", "human") and not q:
            q = val
        elif role in ("assistant", "gpt") and q and not a:
            a = val
    return q, a


def unescape(s):
    return (s or "").replace("\\n", "\n").replace("\\t", "\t")


def main():
    random.seed(SEED)
    rows = []
    skipped = 0

    for subset, path in SOURCES.items():
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]

        n = min(N_PER_SUBSET, len(lines))
        if n < N_PER_SUBSET:
            print(f"warning: {subset} has only {len(lines)} rows")

        for line in random.sample(lines, n):
            q, a = extract(json.loads(line)["conversations"])
            q, a = unescape(q), unescape(a)
            if not q or not a:
                skipped += 1
                continue
            rows.append({"subset": subset, "question": q, "answer": a})

    # Interleave subsets so reviewers don't grade one source in a block:
    # fatigue and drift would otherwise correlate with data source.
    random.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = i

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    counts = {}
    for r in rows:
        counts[r["subset"]] = counts.get(r["subset"], 0) + 1
    print(f"wrote {len(rows)} samples to {OUT}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if skipped:
        print(f"skipped {skipped} rows with an empty question or answer")


if __name__ == "__main__":
    main()