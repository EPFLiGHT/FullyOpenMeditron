#!/usr/bin/env python3
"""Analyse the clinician audit CSV exported by the annotator.

    python analyze_audit.py meditronfo_audit_2026-07-25.csv

Prints the numbers for the meta-review rebuttal: defect rate per subset with CIs,
human-vs-LLM agreement per criterion, and how well the LLM judge stands in for the
clinician's accept/reject decision. Stdlib only, no pandas.
"""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter

CRITERIA = [
    ("Clinical Soundness", "clinical_soundness"),
    ("Safety", "safety"),
    ("Contextual Fit", "contextual_fit"),
    ("Appropriate Confidence", "appropriate_confidence"),
    ("Clarity & Completeness", "clarity_completeness"),
]
FLAGS = ["factual_error", "unsafe_recommandation", "ambiguous_question", "hallucination"]
FLAG_ON = {"error_found", "unsafe_recommandation", "ambiguous", "hallucination"}
SCALE = [-2, -1, 0, 1, 2]
BOOTSTRAP = 2000


def num(s):
    s = (s or "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:  # -sig: the export writes a BOM for Excel
        return list(csv.DictReader(f))


def merge(rows, path):
    """Swap in LLM scores from a re-judged run. src_id is the corpus line, stable across runs;
    sample_id is only the annotator's display order, so do not join on it."""
    with open(path, encoding="utf-8") as f:
        judged = {}
        for line in f:
            if line.strip():
                j = json.loads(line)
                judged[j["id"]] = j
    hit = 0
    for r in rows:
        j = judged.get(r.get("src_id"))
        if not j or "scores" not in j:
            continue
        for _, col in CRITERIA:
            r["llm_" + col] = str(j["scores"][name_of(col)])
        r["llm_model"] = j.get("model", "")
        r["llm_notes"] = j.get("notes", "")
        r["llm_rubric"] = j.get("rubric", "")
        r["llm_question_defect"] = "1" if j.get("question_defect") else ""
        hit += 1
    miss = len(rows) - hit
    print(f"merged {hit}/{len(rows)} rows from {path}"
          f"{f' ({miss} not found or ungraded, keeping their original LLM scores)' if miss else ''}")
    rubrics = {r.get("llm_rubric", "") for r in rows}
    if len(rubrics) > 1:
        print(f"WARNING: mixed rubric versions in one analysis: {sorted(rubrics)}")


def name_of(col):
    return next(n for n, c in CRITERIA if c == col)


def wilson(k, n, z=1.96):
    """Wilson score interval. Normal approximation is useless at n=20."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa_w(pairs, cats=SCALE):
    """Quadratic-weighted Cohen's kappa. None if either rater never varies."""
    n = len(pairs)
    if not n:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1
    ma = [sum(row) for row in obs]
    mb = [sum(col) for col in zip(*obs)]
    w = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    num_ = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ma[i] * mb[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num_ / den


def kappa_ci(pairs, rng, reps=BOOTSTRAP):
    """Percentile bootstrap. At n=60 the point estimate alone is not reportable."""
    vals = []
    for _ in range(reps):
        k = kappa_w(rng.choices(pairs, k=len(pairs)))
        if k is not None:
            vals.append(k)
    if len(vals) < reps // 2:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:4.1f}%  [{100 * lo:.0f}–{100 * hi:.0f}%]" if n else "n/a"


def human(row, col):
    return num(row.get("human_" + col))


def llm(row, col):
    return num(row.get("llm_" + col))


def scored(row):
    return all(human(row, c) is not None for _, c in CRITERIA)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV exported from the annotator")
    ap.add_argument("--flag-threshold", type=int, default=-1,
                    help="a criterion score <= this counts as a defect (default -1)")
    ap.add_argument("--judged", help="judged.jsonl from a re-run; replaces the llm_* columns, "
                                     "joined on src_id. The clinician columns are never touched.")
    ap.add_argument("--seed", type=int, default=2001)
    args = ap.parse_args()

    rows = read(args.csv)
    if args.judged:
        merge(rows, args.judged)
    rng = random.Random(args.seed)
    done = [r for r in rows if scored(r) and r.get("verdict")]
    subsets = sorted({r["subset"] for r in rows})
    thr = args.flag_threshold

    section("Coverage")
    print(f"items in file:            {len(rows)}")
    print(f"fully reviewed:           {len(done)}")
    if len(done) < len(rows):
        print(f"incomplete, excluded:     {len(rows) - len(done)}")
    revealed = sum(1 for r in done if (r.get("llm_revealed_at") or "").strip())
    print(f"LLM scores revealed after scoring: {revealed}/{len(done)}"
          f"  (the UI blocks reveal before scoring, so these are unanchored)")
    if not done:
        sys.exit("nothing fully reviewed yet")

    section("Clinician verdict")
    for sub in subsets:
        s = [r for r in done if r["subset"] == sub]
        rej = sum(1 for r in s if r["verdict"] == "reject")
        print(f"{sub:<18} rejected {pct(rej, len(s))}")
    rej = sum(1 for r in done if r["verdict"] == "reject")
    print(f"{'ALL':<18} rejected {pct(rej, len(done))}")

    section("Concerns raised (clinician)")
    for flag in FLAGS:
        line = []
        for sub in subsets:
            s = [r for r in done if r["subset"] == sub]
            k = sum(1 for r in s if (r.get(flag) or "") in FLAG_ON)
            line.append(f"{sub} {k}/{len(s)}")
        tot = sum(1 for r in done if (r.get(flag) or "") in FLAG_ON)
        print(f"{flag:<24} {'  ·  '.join(line)}   |  all {pct(tot, len(done))}")

    section("Mean scores per criterion (clinician / LLM)")
    print(f"{'':<24}" + "".join(f"{s[:16]:>19}" for s in subsets) + f"{'ALL':>19}")
    for name, col in CRITERIA:
        cells = []
        for sub in subsets + [None]:
            s = [r for r in done if sub is None or r["subset"] == sub]
            h = [human(r, col) for r in s]
            m = [llm(r, col) for r in s if llm(r, col) is not None]
            cells.append(f"{statistics.fmean(h):+.2f} / {statistics.fmean(m):+.2f}" if m else f"{statistics.fmean(h):+.2f} /   —")
        print(f"{name:<24}" + "".join(f"{c:>19}" for c in cells))

    section("Agreement, clinician vs LLM (quadratic-weighted kappa)")
    print(f"{'criterion':<24}{'n':>4}{'exact':>8}{'±1':>7}{'bias':>8}{'kappa_w':>9}   95% CI")
    pooled = []
    for name, col in CRITERIA:
        pairs = [(human(r, col), llm(r, col)) for r in done if llm(r, col) is not None]
        if not pairs:
            print(f"{name:<24}   no LLM scores")
            continue
        pooled += pairs
        n = len(pairs)
        exact = sum(1 for a, b in pairs if a == b)
        near = sum(1 for a, b in pairs if abs(a - b) <= 1)
        bias = statistics.fmean(b - a for a, b in pairs)  # >0 means the LLM scores higher than the clinician
        k = kappa_w(pairs)
        ci = kappa_ci(pairs, rng) if k is not None else None
        ks = "flat" if k is None else f"{k:+.2f}"
        cis = "" if not ci else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        print(f"{name:<24}{n:>4}{100*exact/n:>7.0f}%{100*near/n:>6.0f}%{bias:>+8.2f}{ks:>9}   {cis}")
    if pooled:
        k = kappa_w(pooled)
        ci = kappa_ci(pooled, rng)
        n = len(pooled)
        exact = sum(1 for a, b in pooled if a == b)
        near = sum(1 for a, b in pooled if abs(a - b) <= 1)
        bias = statistics.fmean(b - a for a, b in pooled)
        print(f"{'POOLED (not independent)':<24}{n:>4}{100*exact/n:>7.0f}%{100*near/n:>6.0f}%{bias:>+8.2f}"
              f"{k:>+9.2f}   [{ci[0]:+.2f}, {ci[1]:+.2f}]" if k is not None else "")

    section(f"LLM judge as a screen for the clinician's reject (defect = any criterion <= {thr})")
    for label, pred in [("any <= -1 OR ill-posed", lambda r: min(llm(r, c) for _, c in CRITERIA) <= -1
                                                   or r.get("llm_question_defect") == "1"),
                        ("ill-posed question only", lambda r: r.get("llm_question_defect") == "1"),
                        ("any criterion <= -1", lambda r: min(llm(r, c) for _, c in CRITERIA) <= -1),
                        ("any criterion <= -2", lambda r: min(llm(r, c) for _, c in CRITERIA) <= -2),
                        ("mean criterion < 0", lambda r: statistics.fmean([llm(r, c) for _, c in CRITERIA]) < 0)]:
        s = [r for r in done if all(llm(r, c) is not None for _, c in CRITERIA)]
        if not s:
            print("no LLM scores to compare")
            break
        tp = sum(1 for r in s if pred(r) and r["verdict"] == "reject")
        fp = sum(1 for r in s if pred(r) and r["verdict"] != "reject")
        fn = sum(1 for r in s if not pred(r) and r["verdict"] == "reject")
        tn = len(s) - tp - fp - fn
        sens = f"{100*tp/(tp+fn):.0f}%" if tp + fn else "n/a"
        spec = f"{100*tn/(tn+fp):.0f}%" if tn + fp else "n/a"
        print(f"{label:<22} TP={tp:<3} FP={fp:<3} FN={fn:<3} TN={tn:<3} "
              f"sensitivity={sens:<5} specificity={spec}")

    section("Items the clinician rejected (for the appendix)")
    for r in done:
        if r["verdict"] != "reject":
            continue
        flags = [f for f in FLAGS if (r.get(f) or "") in FLAG_ON]
        low = [f"{name}={human(r, col)}" for name, col in CRITERIA if human(r, col) <= thr]
        print(f"  #{r['sample_id']:<4} {r.get('src_id', ''):<28} {r['subset']}")
        print(f"       clinician: {', '.join(low) or 'no criterion <= threshold'}"
              f"{'  flags: ' + ', '.join(flags) if flags else ''}")
        if (r.get("comments") or "").strip():
            print(f"       comment:   {r['comments'].strip()[:160]}")
        if (r.get("llm_notes") or "").strip():
            print(f"       llm:       {r['llm_notes'].strip()[:160]}")

    section("Disagreements worth reading (|delta| >= 2 on any criterion)")
    for r in done:
        gaps = [(name, human(r, col), llm(r, col)) for name, col in CRITERIA
                if llm(r, col) is not None and abs(human(r, col) - llm(r, col)) >= 2]
        if gaps:
            g = ", ".join(f"{n}: MD {h:+d} vs LLM {m:+d}" for n, h, m in gaps)
            print(f"  #{r['sample_id']:<4} {r['subset']:<18} {g}")


def selftest():
    assert kappa_w([(2, 2), (1, 1), (-2, -2), (0, 0)]) == 1.0
    assert kappa_w([(2, 2), (2, 2)]) is None                      # no variance -> undefined
    assert kappa_w([(2, -2), (-2, 2)]) < 0                        # systematic reversal
    k = kappa_w([(2, 2), (1, 2), (0, -1), (-2, -2), (1, 1), (-1, 0)])
    assert 0.5 < k < 1.0, k
    lo, hi = wilson(1, 20)
    assert lo < 0.05 < hi and hi > 0.2, (lo, hi)                   # 1/20 is compatible with 20%+
    assert wilson(0, 0) == (0.0, 0.0)
    assert name_of("safety") == "Safety"
    assert num(" 3 ") == 3 and num("") is None and num("x") is None
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()