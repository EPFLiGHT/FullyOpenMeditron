#!/usr/bin/env python3
"""Compare judges on the corpus audit: against the clinician, and on the old-vs-clean change.

    python corpus_audit/compare_judges.py \
        --judge "Gemini 3.6 Flash=corpus_audit/clean/gemini-3.6-flash" \
        --judge "Gemma-4-31B=corpus_audit/clean/gemma4_31b" \
        --judge "Gemma-4-31B thinking=corpus_audit/clean/gemma4_31b_think"

Each directory holds one judge's four runs:
  judged_audit60_old.jsonl    the 60 audited items as audited
  judged_audit60_clean.jsonl  the same items as they are in the clean corpus
  judged_800_old.jsonl        the July sample of 800
  judged_800_clean.jsonl      the same items in the clean corpus (build_clean_items.py)
compare_clean.py gives the full detail for one judge; this prints what sets the judges apart.
--new v1 reads judged_audit60_v1.jsonl and judged_800_v1.jsonl instead, for the runs of corpus_audit/v1/
(README.md there).
Stdlib only.
"""

import argparse
import csv
import itertools
import json
import random
import statistics
from pathlib import Path

from analyze_audit import kappa_ci, kappa_w
from analyze_corpus import CRITERIA, wilson
from compare_clean import ROWS, flag, load, mcnemar, paired_ci

HUMAN = {c: "human_" + c.lower().replace(" & ", "_").replace(" ", "_") for c in CRITERIA}
CS, CC = "Clinical Soundness", "Clarity & Completeness"


def section(t):
    print(f"\n{t}\n{'-' * len(t)}")


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", required=True, help="LABEL=DIR, repeatable")
    ap.add_argument("--items", default="corpus_audit/clean/items_800.jsonl")
    ap.add_argument("--audit", default="corpus_audit/data/meditronfo_audit_2026-07-25.csv")
    ap.add_argument("--audit-items", default="corpus_audit/clean/items_audit60.jsonl")
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--new", default="clean", help="the later corpus: clean, or v1 (compare_clean.ROWS key)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    judges = []
    for spec in args.judge:
        label, _, d = spec.partition("=")
        d = Path(d)
        judges.append((label, {k: load(d / f"judged_{k}.jsonl")
                               for k in ("audit60_old", f"audit60_{args.new}", "800_old", f"800_{args.new}")}))
    items = [json.loads(ln) for ln in open(args.items, encoding="utf-8") if ln.strip()]
    subsets = list(dict.fromkeys(i["subset"] for i in items))
    with open(args.audit, encoding="utf-8-sig", newline="") as f:
        audit = [r for r in csv.DictReader(f) if r.get("verdict") in ("accept", "reject")]
    a_items = {i["old_id"]: i for i in map(json.loads, open(args.audit_items, encoding="utf-8"))}
    w = max(len(label) for label, _ in judges) + 2

    section("Coverage (graded rows; a judge's failed or unparseable rows are missing)")
    for label, r in judges:
        print(f"{label:<{w}}" + "  ".join(f"{k} {len(v)}" for k, v in r.items()))

    section("Against the clinician: the 60 audited items as audited")
    print(f"{'judge':<{w}}{'catches rejects':>24}{'passes accepts':>24}{'kappa_w pooled [95% CI]':>28}"
          f"{'bias':>7}   kappa_w per criterion (CS, Saf, Fit, Conf, C&C)")
    for label, r in judges:
        j = r["audit60_old"]
        rows = [a for a in audit if a["src_id"] in j]
        rej = [a for a in rows if a["verdict"] == "reject"]
        acc = [a for a in rows if a["verdict"] != "reject"]
        tp = sum(flag(j[a["src_id"]]) for a in rej)
        tn = sum(not flag(j[a["src_id"]]) for a in acc)
        pooled = [(int(a[HUMAN[c]]), j[a["src_id"]]["scores"][c]) for a in rows for c in CRITERIA]
        k, ci = kappa_w(pooled), kappa_ci(pooled, rng)
        bias = statistics.fmean(b - a for a, b in pooled)
        per = [kappa_w([(int(a[HUMAN[c]]), j[a["src_id"]]["scores"][c]) for a in rows]) for c in CRITERIA]
        print(f"{label:<{w}}{pct(tp, len(rej)):>24}{pct(tn, len(acc)):>24}"
              f"{f'{k:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]':>28}{bias:>+7.2f}   "
              + "  ".join("flat" if x is None else f"{x:+.2f}" for x in per))

    section(f"Flag rate, July sample -> {args.new}, paired on the question")
    print(f"{'judge':<{w}}{'subset':<17}{'old':>18}{args.new:>18}{'paired diff [95% CI]':>28}{'McNemar p':>11}")
    for label, r in judges:
        old, new = r["800_old"], r[f"800_{args.new}"]
        pairs = [(i, old[i["old_id"]], new[i["id"]]) for i in items
                 if i["status"] != "removed" and i["old_id"] in old and i["id"] in new]
        for s in subsets:
            sp = [(int(flag(a)), int(flag(b))) for i, a, b in pairs if i["subset"] == s]
            allold = [flag(x) for x in old.values() if x["subset"] == s]
            if not sp:
                continue
            d = statistics.fmean(b - a for a, b in sp)
            lo, hi = paired_ci(sp, rng)
            fixed, broke = sum(a and not b for a, b in sp), sum(b and not a for a, b in sp)
            print(f"{label:<{w}}{s:<17}{f'{100 * sum(allold) / len(allold):.1f}%':>18}"
                  f"{f'{100 * statistics.fmean(b for _, b in sp):.1f}%':>18}"
                  f"{f'{100 * d:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]':>28}{mcnemar(fixed, broke):>11.3f}")
        rates = {}
        for side, rows in (("old", [(x["subset"], flag(x)) for x in old.values()]),
                           (args.new, [(i["subset"], flag(b)) for i, _, b in pairs])):
            by = {s: [f for sub, f in rows if sub == s] for s in subsets}
            if all(by.values()):
                wts = ROWS[side]
                rates[side] = sum(wts[s] * statistics.fmean(by[s]) for s in subsets) / sum(wts[s] for s in subsets)
        if len(rates) == 2:
            print(f"{label:<{w}}{'whole corpus':<17}{f'{100 * rates['old']:.1f}%':>18}{f'{100 * rates[args.new]:.1f}%':>18}")
        same = [(flag(a), flag(b)) for i, a, b in pairs if i["status"] == "unchanged"]
        if same:
            print(f"{label:<{w}}{'(unchanged answers: flag differs on '}"
                  f"{sum(a != b for a, b in same)}/{len(same)})")

    section("MOOVE-style: what the flags are for")
    for label, r in judges:
        old, new = r["800_old"], r[f"800_{args.new}"]
        mp = [(old[i["old_id"]], new[i["id"]]) for i in items
              if i["subset"] == "Synthetic MOOVE" and i["old_id"] in old and i["id"] in new]
        if not mp:
            continue
        alone = lambda x: flag(x) and all(v > -1 for c, v in x["scores"].items() if c != CC)
        d = [(a["scores"][CS], b["scores"][CS]) for a, b in mp]
        lo, hi = paired_ci(d, rng)
        print(f"{label:<{w}}flagged {sum(flag(a) for a, _ in mp)} -> {sum(flag(b) for _, b in mp)};  "
              f"on Clarity & Completeness alone {sum(alone(a) for a, _ in mp)} -> {sum(alone(b) for _, b in mp)};  "
              f"Clinical Soundness <= -1 {sum(a <= -1 for a, _ in d)} -> {sum(b <= -1 for _, b in d)}, "
              f"mean {statistics.fmean(b - a for a, b in d):+.2f} [{lo:+.2f}, {hi:+.2f}]")

    section(f"Judge vs judge on the same answers (July 800 + {args.new})")
    for (la, ra), (lb, rb) in itertools.combinations(judges, 2):
        both = [(ra[k][i], rb[k][i]) for k in ("800_old", f"800_{args.new}") for i in ra[k] if i in rb[k]]
        if not both:
            continue
        fl = [(flag(a), flag(b)) for a, b in both]
        agree = sum(a == b for a, b in fl)
        kf = kappa_w(fl, cats=[False, True])
        pooled = [(a["scores"][c], b["scores"][c]) for a, b in both for c in CRITERIA]
        print(f"{la} vs {lb}: n={len(both)}; flag agrees {100 * agree / len(fl):.0f}% "
              f"(Cohen kappa {kf:+.2f}); flagged by {la} only {sum(a and not b for a, b in fl)}, "
              f"by {lb} only {sum(b and not a for a, b in fl)}; score kappa_w pooled {kappa_w(pooled):+.2f}")

    section(f"The clinician's ten rejects and fifty accepts: flagged as audited -> in the {args.new} corpus")
    for label, r in judges:
        cells = []
        for verdict in ("reject", "accept"):
            rows = [a for a in audit if a["verdict"] == verdict]
            k0 = sum(flag(r["audit60_old"][a["src_id"]]) for a in rows if a["src_id"] in r["audit60_old"])
            kept = [a_items[a["src_id"]]["id"] for a in rows]
            k1 = sum(flag(r[f"audit60_{args.new}"][i]) for i in kept if i in r[f"audit60_{args.new}"])
            cells.append(f"{verdict}s {k0} -> {k1} of {len(rows)}")
        print(f"{label:<{w}}" + "   ".join(cells))


if __name__ == "__main__":
    main()
