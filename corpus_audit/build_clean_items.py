#!/usr/bin/env python3
"""Find the items of an earlier judge run in the clean corpus, so they can be judged again.

    python corpus_audit/build_clean_items.py corpus_audit/judge_outputs/judged_pro_800.jsonl \
        --out corpus_audit/clean/items_800.jsonl
    python corpus_audit/judge_moove.py --items corpus_audit/clean/items_800.jsonl \
        --out corpus_audit/clean/judged_clean_800.jsonl

Each old item is matched to the clean set of its subset (judge_moove.CLEAN_SOURCES) on the first
user turn. The truncation fix and the MOOVE-style regeneration kept every prompt and changed only
answers, so the question is the join key. One row per old item, with a status:

  unchanged    same question, same answer. Re-judging it measures the judge's own drift.
  regenerated  same question, new answer.
  removed      question not in the clean set (MedExpQA, decontamination, unusable question).
               Written with an empty answer, which judge_moove.py --items skips.

id is <subset>:<line in the clean file>, old_id the item's id in the old run.

--corpus DIR looks the items up in a data/corpora/<name>/ directory instead of the clean sets:
    python corpus_audit/build_clean_items.py corpus_audit/judge_outputs/judged_pro_800.jsonl \
        --corpus data/corpora/meditronfo_gptoss_v1 --out corpus_audit/v1/items_800.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from judge_moove import CLEAN_SOURCES, corpus_sources, extract, unescape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged", type=Path, help="jsonl from judge_moove.py (id, subset, question, answer)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, help="a data/corpora/<name>/ directory; default the clean sets")
    args = ap.parse_args()

    old = [json.loads(ln) for ln in args.judged.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_subset = defaultdict(lambda: defaultdict(list))  # subset -> question -> old items
    for r in old:
        by_subset[r["subset"]][r["question"]].append(r)

    sources = corpus_sources(args.corpus) if args.corpus else CLEAN_SOURCES
    hits = {}  # old id -> (clean line, clean answer)
    for subset, want in by_subset.items():
        with open(sources[subset], encoding="utf-8") as f:
            # same indexing as judge_moove.load: position among non-empty lines
            for i, line in enumerate(ln for ln in f if ln.strip()):
                q, a = extract(json.loads(line)["conversations"])
                q = unescape(q)
                for r in want.get(q, ()):
                    a = unescape(a)
                    prev = hits.get(r["id"])
                    # a duplicated prompt: prefer the copy whose answer is the one judged before
                    if prev is None or (prev[1] != r["answer"] and a == r["answer"]):
                        hits[r["id"]] = (i, a)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.out.open("w", encoding="utf-8") as f:
        for r in old:
            if r["id"] in hits:
                i, a = hits[r["id"]]
                status = "unchanged" if a == r["answer"] else "regenerated"
                row = {"id": f"{r['subset']}:{i}", "answer": a}
            else:
                status = "removed"
                row = {"id": r["id"], "answer": ""}
            row = {**row, "old_id": r["id"], "subset": r["subset"], "question": r["question"], "status": status}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[r["subset"], status] += 1

    print(f"wrote {len(old)} rows to {args.out}")
    for subset in by_subset:
        print(f"  {subset:<16} " + "  ".join(f"{s} {counts[subset, s]}" for s in ("unchanged", "regenerated", "removed")))


if __name__ == "__main__":
    main()
