#!/usr/bin/env python3
"""How often the answering model contradicts a multiple-choice label, with 95% intervals.

    python3 corpus_audit/label_mismatch.py \
        --answers "gpt-oss-120b=$SCRATCH/meditronfo/resampling/answers_gptoss_full" \
        --answers "Qwen3.8-27B=$SCRATCH/meditronfo/resampling/answers_qwen38_full"

From the runs of resampling/resample_answers.py:
  pass@8   every MCQ row of the corpus. resample_answers.py keeps a row once a sample picks the
           label letter and rejects it after 8 misses ("no sample passed"); the rejected rows are
           the mismatches. "confident" counts the rejects whose recorded samples all picked the same
           other letter. A lower bound on label errors: one lucky sample out of 8 keeps a row.
Curated QA's labels come from the source benchmarks' own training sets (MedQA, MedMCQA, ...), so its
pass@8 rate is the teacher's own miss rate on human labels: the baseline the synthetic sets compare to.
Stdlib only.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_corpus import wilson

SETS = {  # resampling input name -> training set
    "gpt_oss_8retries_v3_fixed_no_medexqa_decontam": "Curated QA",
    "meditron_4_synthetic_qa_v3_fixed_decontam": "Synthetic QA",
    "guidelines_qa_v3_full_fixed_decontam": "Guidelines QA",
}


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>6}/{n:<7} {100 * k / n:5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}]" if n else "n/a"


def pass_at_8(d):
    for name, subset in SETS.items():
        kept, retried = 0, 0
        for line in open(d / f"{name}_resampled.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("resample_kind") == "mcq":
                kept += 1
                retried += r.get("resample_samples", 1) > 1
        miss, confident, dropped = 0, 0, Counter()
        for line in open(d / f"{name}_rejected.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["reason"] == "no sample passed":
                miss += 1
                confident += len(r["letters"]) >= 3 and len(set(r["letters"])) == 1
            else:
                dropped[r["reason"]] += 1
        yield subset, kept + miss, miss, confident, retried, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="append", required=True, help="LABEL=resample_answers.py outdir")
    args = ap.parse_args()

    print("pass@8 over every MCQ row: label never reached in 8 samples  (n/N, %, 95% Wilson CI)")
    for spec in args.answers:
        label, _, d = spec.partition("=")
        print(f"\n{label}")
        print(f"  {'set':<15}{'label never reached':>36}{'  all samples on one other letter':>38}"
              f"{'needed more than one sample':>38}   not voted on")
        for subset, n, miss, confident, retried, dropped in pass_at_8(Path(d)):
            print(f"  {subset:<15}{pct(miss, n):>36}{pct(confident, n):>38}{pct(retried, n):>38}   {dict(dropped)}")


if __name__ == "__main__":
    main()
