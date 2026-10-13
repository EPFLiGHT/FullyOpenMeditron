#!/usr/bin/env python3
"""Analyse a large LM-judged run (no clinician labels).

    python analyze_corpus.py judged_pro_800.jsonl
    python analyze_corpus.py judged_pro_800.jsonl --baseline "Curated QA" --md

Reports per-subset quality, the defect rate under the screen that was validated against
the clinician, and the synthetic-vs-non-synthetic difference. Stdlib only.

The screen's operating point comes from the 60-item audit (--sens/--spec). Change them if
you revalidate, because every corrected number below depends on them.
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter

CRITERIA = [
    "Clinical Soundness",
    "Safety",
    "Contextual Fit",
    "Appropriate Confidence",
    "Clarity & Completeness",
]
SCALE = [-2, -1, 0, 1, 2]
BOOT = 2000
# heuristic themes for the free-text notes; substring match, lowercased
THEMES = {
    "truncated answer": ("truncat", "cut off", "cuts off", "mid-sentence", "incomplete respons"),
    "missing MCQ options": ("missing multiple-choice", "missing options", "options are missing",
                            "without listing options", "no options"),
    "unspecified referent": ("referent", "not named", "never named", "unspecified"),
    "question defect (any)": ("question defect", "defect:", "unanswerable", "under-determined"),
    "dosing problem": ("dose", "dosing", "mg/kg"),
    "guideline conflict": ("guideline", "ats/ers", "who recommend", "contraindicat"),
}


def load(path):
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "scores" in r and set(r["scores"]) == set(CRITERIA):
                rows.append(r)
            else:
                bad += 1
    return rows, bad


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mean_ci(xs, z=1.96):
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, m, m
    h = z * statistics.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h


def flagged(r, thr=-1):
    return min(r["scores"].values()) <= thr


def corrected(apparent, sens, spec):
    """Rogan-Gladen: recover true prevalence from an imperfect screen.
    Returns None when the apparent rate sits below the screen's false-positive floor,
    which means the data cannot distinguish the true rate from zero."""
    denom = sens + spec - 1
    if denom <= 0:
        return None
    p = (apparent + spec - 1) / denom
    return None if p < 0 else min(1.0, p)


def boot_diff(a, b, rng, reps=BOOT):
    """Percentile CI for the difference in defect rate, resampling items."""
    if not a or not b:
        return None
    d = []
    for _ in range(reps):
        ra = sum(rng.choices(a, k=len(a))) / len(a)
        rb = sum(rng.choices(b, k=len(b))) / len(b)
        d.append(ra - rb)
    d.sort()
    return d[int(0.025 * reps)], d[int(0.975 * reps)]


def section(t):
    print(f"\n{t}\n{'-' * len(t)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged", help="judged jsonl from judge_moove.py")
    ap.add_argument("--baseline", default=None,
                    help="subset to compare the others against, e.g. the non-synthetic set")
    ap.add_argument("--sens", type=float, default=0.90, help="screen sensitivity from the audit")
    ap.add_argument("--spec", type=float, default=0.72, help="screen specificity from the audit")
    ap.add_argument("--worst", type=int, default=3, help="worst items to print per subset")
    ap.add_argument("--md", action="store_true", help="also emit a markdown table for the paper")
    ap.add_argument("--seed", type=int, default=2001)
    args = ap.parse_args()

    rows, bad = load(args.judged)
    if not rows:
        sys.exit("no gradeable rows")
    rng = random.Random(args.seed)
    subsets = sorted({r["subset"] for r in rows})
    if args.baseline:
        if args.baseline not in subsets:
            sys.exit(f"--baseline {args.baseline!r} not in {subsets}")
        subsets = [args.baseline] + [s for s in subsets if s != args.baseline]
    by = {s: [r for r in rows if r["subset"] == s] for s in subsets}

    section("Coverage")
    print(f"graded rows: {len(rows)}" + (f"   ungradeable/failed: {bad}" if bad else ""))
    for s in subsets:
        print(f"  {s:<20} {len(by[s])}")
    meta = Counter((r.get("model", "?"), r.get("rubric", "?")) for r in rows)
    for (m, v), n in meta.items():
        print(f"judge: {m}  rubric: {v}  ({n} rows)")
    if len(meta) > 1:
        print("WARNING: rows were produced by more than one judge or rubric version")

    section("Mean score per criterion  [95% CI]")
    print(f"{'':<24}" + "".join(f"{s[:20]:>22}" for s in subsets))
    for c in CRITERIA:
        cells = []
        for s in subsets:
            m, lo, hi = mean_ci([r["scores"][c] for r in by[s]])
            cells.append(f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{c:<24}" + "".join(f"{c_:>22}" for c_ in cells))

    section("Score distribution (% of items, all criteria pooled)")
    print(f"{'':<24}" + "".join(f"{v:>8}" for v in SCALE))
    for s in subsets:
        vals = [r["scores"][c] for r in by[s] for c in CRITERIA]
        n = len(vals)
        cnt = Counter(vals)
        print(f"{s:<24}" + "".join(f"{100*cnt[v]/n:>7.1f}%" for v in SCALE))

    section(f"Defect rate under the validated screen (any criterion <= -1)")
    print(f"{'subset':<20}{'flagged':>10}{'apparent [95% CI]':>26}{'corrected*':>14}")
    for s in subsets:
        k, n = sum(flagged(r) for r in by[s]), len(by[s])
        lo, hi = wilson(k, n)
        corr = corrected(k / n, args.sens, args.spec)
        cs = "<= 0 **" if corr is None else f"{100*corr:.1f}%"
        print(f"{s:<20}{f'{k}/{n}':>10}{f'{100*k/n:.1f}% [{100*lo:.1f}-{100*hi:.1f}]':>26}{cs:>14}")
    print(f"\n* Rogan-Gladen correction using sensitivity={args.sens:.2f}, specificity={args.spec:.2f}")
    print("  from the 60-item clinician audit. Those come from 10 rejected items, so they carry")
    print("  wide uncertainty of their own that is NOT propagated here: treat corrected rates as")
    print("  indicative. The apparent rate is what the judge saw and is the defensible number.")
    print("** apparent rate is at or below the screen's false-positive floor (1 - specificity),")
    print("   so the data cannot distinguish the true defect rate from zero.")

    if args.baseline:
        section(f"Difference vs {args.baseline} (positive = more defects than baseline)")
        base = [flagged(r) for r in by[args.baseline]]
        for s in subsets[1:]:
            other = [flagged(r) for r in by[s]]
            d = statistics.fmean(other) - statistics.fmean(base)
            ci = boot_diff(other, base, rng)
            note = "" if ci is None or (ci[0] < 0 < ci[1]) else "   <- CI excludes 0"
            print(f"  {s:<20}{100*d:+6.1f} pp  [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]{note}")

    section("Themes in the judge's notes (heuristic substring match, items may match several)")
    print(f"{'theme':<24}" + "".join(f"{s[:20]:>22}" for s in subsets))
    for theme, keys in THEMES.items():
        cells = []
        for s in subsets:
            k = sum(any(x in (r.get("notes") or "").lower() for x in keys) for r in by[s])
            lo, hi = wilson(k, len(by[s]))
            cells.append(f"{k:>3} ({100*k/len(by[s]):.0f}% [{100*lo:.0f}-{100*hi:.0f}])")
        print(f"{theme:<24}" + "".join(f"{c:>22}" for c in cells))
    print("\nSubstring matching over free text: read these as pointers, not measurements.")

    if args.worst:
        section(f"Worst {args.worst} items per subset")
        for s in subsets:
            for r in sorted(by[s], key=lambda r: sum(r["scores"].values()))[:args.worst]:
                lows = ", ".join(f"{c}={v:+d}" for c, v in r["scores"].items() if v <= -1)
                print(f"  {r['id']:<28} {lows or 'no criterion <= -1'}")
                print(f"       {(r.get('notes') or '')[:150]}")

    if args.md:
        section("Markdown, for the response")
        print("| Component | n | Mean score* | Flagged by screen | 95% CI |")
        print("|---|---|---|---|---|")
        for s in subsets:
            allv = [r["scores"][c] for r in by[s] for c in CRITERIA]
            k, n = sum(flagged(r) for r in by[s]), len(by[s])
            lo, hi = wilson(k, n)
            print(f"| {s} | {n} | {statistics.fmean(allv):+.2f} | {100*k/n:.1f}% | [{100*lo:.1f}–{100*hi:.1f}%] |")
        print("\n*mean over all five criteria, -2 to +2.")


def selftest():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(20, 200)
    assert lo < 0.10 < hi
    assert corrected(0.90, 0.90, 0.72) is not None
    assert corrected(0.20, 0.90, 0.72) is None          # below the FP floor -> undefined, not negative
    p = corrected(0.35, 0.90, 0.72)
    assert abs(p - (0.35 + 0.72 - 1) / 0.62) < 1e-9
    assert corrected(0.5, 0.5, 0.5) is None             # a coin flip carries no information
    m, lo, hi = mean_ci([1, 1, 1, 1])
    assert m == 1 and lo == hi == 1
    assert flagged({"scores": {c: 0 for c in CRITERIA}}) is False
    assert flagged({"scores": {**{c: 2 for c in CRITERIA}, "Safety": -1}}) is True
    d = boot_diff([True] * 50, [False] * 50, random.Random(0), reps=200)
    assert d[0] > 0.9
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()