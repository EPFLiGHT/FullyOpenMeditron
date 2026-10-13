#!/usr/bin/env python3
"""The five error categories of judge_moove.py --categories: against the clinician, and on the corpus.

    python corpus_audit/audit_categories.py \
        --judge "Gemini 3.6 Flash=corpus_audit/v1/gemini-3.6-flash" \
        --judge "Gemma-4-31B=corpus_audit/v1/gemma4_31b"

Each directory holds one judge's --categories runs, named as for compare_judges.py --new v1, plus
judged_sample800_v1.jsonl, a fresh 200 per set of the v1 corpus. Prints
  1. each category against the clinician's flag on the 60 audited items as audited (four of the five;
     the clinician had no label-mismatch box),
  2. the category rates in the v1 corpus, per set and weighted by rows, with 95% intervals,
  3. the July 800 against the same questions in v1, paired,
  4. what became of the clinician's ten rejects.
Stdlib only.
"""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from analyze_audit import kappa_w
from analyze_corpus import wilson
from compare_clean import ROWS, flag, load, mcnemar, paired_ci

CATEGORIES = ["factual_error", "unsafe_recommendation", "ambiguous_question", "hallucinated_rationale",
              "label_mismatch"]
# judge category -> (clinician CSV column, the value that means the defect is present)
CLINICIAN = {
    "factual_error": ("factual_error", "error_found"),
    "unsafe_recommendation": ("unsafe_recommandation", "unsafe_recommandation"),
    "ambiguous_question": ("ambiguous_question", "ambiguous"),
    "hallucinated_rationale": ("hallucination", "hallucination"),
}
SHORT = {"factual_error": "factual", "unsafe_recommendation": "unsafe", "ambiguous_question": "ambiguous",
         "hallucinated_rationale": "halluc.", "label_mismatch": "label", "flag": "flagged", "any": "any cat."}


def section(t):
    print(f"\n{t}\n{'-' * len(t)}")


def has(r, c):
    """c is a category, 'flag' (any score <= -1, the July screen) or 'any' (any category)."""
    if c == "flag":
        return flag(r)
    if c == "any":
        return any(r["defects"].values())
    return r["defects"][c]


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:4.1f} [{100 * lo:4.1f}-{100 * hi:4.1f}]" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", required=True, help="LABEL=DIR, repeatable")
    ap.add_argument("--new", default="v1", help="corpus suffix of the run files and compare_clean.ROWS key")
    ap.add_argument("--items", default="corpus_audit/v1/items_800.jsonl")
    ap.add_argument("--audit", default="corpus_audit/data/meditronfo_audit_2026-07-25.csv")
    ap.add_argument("--audit-items", default="corpus_audit/v1/items_audit60.jsonl")
    ap.add_argument("--seed", type=int, default=2001)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    new = args.new

    judges = []
    for spec in args.judge:
        label, _, d = spec.partition("=")
        runs = {k: {i: r for i, r in load(Path(d) / f"judged_{k}.jsonl").items() if "defects" in r}
                for k in ("audit60_old", f"audit60_{new}", "800_old", f"800_{new}", f"sample800_{new}")}
        judges.append((label, runs))
    with open(args.audit, encoding="utf-8-sig", newline="") as f:
        audit = [r for r in csv.DictReader(f) if r.get("verdict") in ("accept", "reject")]
    a_items = {i["old_id"]: i for i in map(json.loads, open(args.audit_items, encoding="utf-8"))}
    items = [json.loads(ln) for ln in open(args.items, encoding="utf-8") if ln.strip()]

    section("Coverage (rows graded with categories)")
    for label, r in judges:
        print(f"{label:<24}" + "  ".join(f"{k} {len(v)}" for k, v in r.items()))

    section("1. Against the clinician: the 60 audited items as audited")
    print("clin = items the clinician flagged; judge = items the judge flagged; sens/spec: the judge's, "
          "taking the clinician as truth")
    for label, r in judges:
        j = r["audit60_old"]
        rows = [a for a in audit if a["src_id"] in j]
        print(f"\n{label} (n={len(rows)})")
        print(f"  {'category':<24}{'clin':>6}{'judge':>7}{'both':>6}{'sens':>12}{'spec':>12}{'kappa':>8}")
        for c, (col, yes) in CLINICIAN.items():
            pairs = [(a[col] == yes, j[a["src_id"]]["defects"][c]) for a in rows]
            pos = [b for a, b in pairs if a]
            neg = [not b for a, b in pairs if not a]
            k = kappa_w(pairs, cats=[False, True])
            print(f"  {c:<24}{len(pos):>6}{sum(b for _, b in pairs):>7}{sum(pos):>6}"
                  f"{f'{sum(pos)}/{len(pos)}':>12}{f'{sum(neg)}/{len(neg)}':>12}"
                  f"{'flat' if k is None else f'{k:+.2f}':>8}")
        lm = sum(j[a["src_id"]]["defects"]["label_mismatch"] for a in rows)
        print(f"  {'label_mismatch':<24}{'--':>6}{lm:>7}   (the clinician had no label box)")
        for c in ("any", "flag"):
            rej = [has(j[a["src_id"]], c) for a in rows if a["verdict"] == "reject"]
            acc = [not has(j[a["src_id"]], c) for a in rows if a["verdict"] != "reject"]
            what = "any category" if c == "any" else "any score <= -1"
            print(f"  {what} as a screen for the reject: catches {sum(rej)}/{len(rej)}, "
                  f"passes {sum(acc)}/{len(acc)} accepts")

    section(f"2. Category rates in the {new} corpus: fresh sample, % [95% Wilson CI]")
    cols = ["flag", "any"] + CATEGORIES
    wts = ROWS[new]
    for label, r in judges:
        s = r[f"sample800_{new}"]
        subsets = [x for x in wts if any(v["subset"] == x for v in s.values())]
        print(f"\n{label}")
        print(f"  {'set':<17}{'n':>4}" + "".join(f"{SHORT[c]:>18}" for c in cols))
        by = {x: [v for v in s.values() if v["subset"] == x] for x in subsets}
        for x in subsets:
            print(f"  {x:<17}{len(by[x]):>4}" + "".join(f"{pct(sum(has(v, c) for v in by[x]), len(by[x])):>18}"
                                                   for c in cols))
        if len(subsets) == len(wts):
            # stratified estimate: each set's rate weighted by its training rows, normal-approximation CI
            tot = sum(wts.values())
            cells = []
            for c in cols:
                p = {x: statistics.fmean(has(v, c) for v in by[x]) for x in subsets}
                est = sum(wts[x] / tot * p[x] for x in subsets)
                se = math.sqrt(sum((wts[x] / tot) ** 2 * p[x] * (1 - p[x]) / len(by[x]) for x in subsets))
                cells.append(f"{100 * est:4.1f} [{100 * max(0, est - 1.96 * se):4.1f}-{100 * (est + 1.96 * se):4.1f}]")
            print(f"  {'whole corpus':<17}{'':>4}" + "".join(f"{c:>18}" for c in cells))

    section(f"3. July 800 -> {new}, paired on the question (items still in {new}); points [95% CI], McNemar p")
    for label, r in judges:
        old, nw = r["800_old"], r[f"800_{new}"]
        pairs = [(i, old[i["old_id"]], nw[i["id"]]) for i in items
                 if i["status"] != "removed" and i["old_id"] in old and i["id"] in nw]
        print(f"\n{label}")
        print(f"  {'set':<17}{'n':>4}" + "".join(f"{SHORT[c]:>27}" for c in cols))
        for x in dict.fromkeys(i["subset"] for i, _, _ in pairs):
            sp = [(a, b) for i, a, b in pairs if i["subset"] == x]
            cells = []
            for c in cols:
                d = [(int(has(a, c)), int(has(b, c))) for a, b in sp]
                lo, hi = paired_ci(d, rng)
                fixed, broke = sum(a and not b for a, b in d), sum(b and not a for a, b in d)
                cells.append(f"{sum(a for a, _ in d)}->{sum(b for _, b in d)} "
                             f"[{100 * lo:+.0f},{100 * hi:+.0f}] p={mcnemar(fixed, broke):.2f}")
            print(f"  {x:<17}{len(sp):>4}" + "".join(f"{c:>27}" for c in cells))

    section(f"4. The clinician's rejects in the {new} corpus")
    for a in (a for a in audit if a["verdict"] == "reject"):
        it = a_items.get(a["src_id"], {})
        line = f"  #{a['sample_id']:<3} {a['subset']:<16} {it.get('status', '?'):<12}"
        for label, r in judges:
            v = r[f"audit60_{new}"].get(it.get("id"))
            if v:
                cats = ",".join(SHORT[c] for c in CATEGORIES if v["defects"][c]) or "none"
                line += f"  {label}: min {min(v['scores'].values()):+d} {cats}"
        print(line + f"   | {(a['comments'] or '').splitlines()[0][:70] if a['comments'] else ''}")


if __name__ == "__main__":
    main()
