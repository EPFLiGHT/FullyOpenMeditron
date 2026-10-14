#!/usr/bin/env python3
# decontamination/decontam_table.py
"""
The paper's decontamination table (appendix tab:decontam, and the counts in Section 2.3), recounted
from the reports decontaminate_corpus.sh wrote: per training set, the rows before decontamination,
the rows each benchmark flagged, and the rows removed. A row can match several benchmarks, so the
benchmark columns can add up to more than the rows removed; "Other" is the union of the remaining
benchmarks (MMLU-Pro, ARC, IFEval, HealthBench, the MOOVE sets).

python3 decontamination/decontam_table.py
"""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEC = ROOT / "data" / "decontaminated"
V4 = ROOT / "data" / "decontam_moove_v4"
# training set -> (report dir, decontaminated JSONL; its .removed.jsonl sits next to it)
SETS = {
    "Curated QA": (DEC / "reports/gpt_oss_8retries_v3_fixed_no_medexqa",
                   DEC / "gpt_oss_8retries_v3_fixed_no_medexqa_decontam.jsonl"),
    "Synthetic Curated QA": (DEC / "reports/meditron_4_synthetic_qa_v3_fixed",
                             DEC / "meditron_4_synthetic_qa_v3_fixed_decontam.jsonl"),
    "Guidelines QA": (DEC / "reports/guidelines_qa_v3_full_fixed",
                      DEC / "guidelines_qa_v3_full_fixed_decontam.jsonl"),
    # the v4 vignettes, answered by gpt-oss-120b, checked against the evaluation set and the unanimous-576 set
    "Synthetic vignettes": (V4 / "combined_reports", V4 / "gptoss_synthetic_moove_v4_decontam.jsonl"),
}
MAIN = {"medqa_test": "MedQA", "medmcqa_val": "MedMCQA", "pubmedqa_labeled": "PubMedQA",
        "medxpertqa_test": "MedXpertQA"}


def lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    cols = list(MAIN.values()) + ["Other"]
    print(f"{'Training set':22s} {'Rows before':>11s} " + " ".join(f"{c:>10s}" for c in cols) + f" {'Removed':>8s}")
    total = {c: 0 for c in ["before", *cols, "removed"]}
    for name, (reports, kept) in SETS.items():
        flagged = {os.path.basename(f).split("__")[0]: set(json.load(open(f)))
                   for f in glob.glob(str(reports / "*__contamination_report.json"))}
        removed = lines(f"{kept}.removed.jsonl")
        row = {"before": lines(kept) + removed, "removed": removed}
        row |= {MAIN[k]: len(flagged.get(k, ())) for k in MAIN}
        row["Other"] = len(set().union(*(v for k, v in flagged.items() if k not in MAIN)))
        # every removed row was flagged by some benchmark, and every flagged row was removed
        assert len(set().union(*flagged.values())) == removed, name
        for k in total:
            total[k] += row[k]
        print(f"{name:22s} {row['before']:>11,d} " + " ".join(f"{row[c]:>10,d}" for c in cols) + f" {removed:>8,d}")
    print(f"{'Total':22s} {total['before']:>11,d} " + " ".join(f"{total[c]:>10,d}" for c in cols)
          + f" {total['removed']:>8,d}")


if __name__ == "__main__":
    main()
