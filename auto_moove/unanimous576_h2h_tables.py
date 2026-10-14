#!/usr/bin/env python3
# auto_moove/unanimous576_h2h_tables.py

"""
Numbers for the MeditronFO head-to-heads (against MedGemma, and against
each base for iclr_clean_reproduction.tex) of
tech_reports/judge_eval_unanimous576.tex, from the auto_moove.py outputs in
auto_moove/results/unanimous576/ (login node, no GPU).

.venv/bin/python auto_moove/unanimous576_h2h_tables.py

Each question counts once. unanimous-576 has 576 pairs over 516 questions; the
first run generated each duplicate separately but cached only the last copy,
which every later run then reused. Keeping the LAST occurrence therefore gives
every judge the same answers.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # login node: 48 BLAS threads fail to spawn

import glob, json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "unanimous576"
SIMILAR = (0.8, 1.25)        # word ratio treated as "similar length"
SCORE = {"1": 1.0, "12": 0.5, "2": 0.0}
rng = np.random.default_rng(0)


def load(path):
    last = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        last[r["prompt"]] = r
    return last


def words(s):
    return len((s or "").split())


def ci(x, n_boot=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), (float("nan"),) * 2
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return x.mean(), tuple(np.percentile(boots, [2.5, 97.5]))


def fmt(x):
    m, (lo, hi) = ci(x)
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] (n={len(x)})"


def label(v):
    return (v or ["?"])[0] if isinstance(v, list) else (v or "?")


def three_way(p, threshold):
    """auto_moove_check.three_way_vote on the order-averaged probabilities."""
    if p["prob_equal_mean"] >= max(p["prob_left_mean"], p["prob_right_mean"]) - threshold:
        return "12"
    return "1" if p["prob_left_mean"] >= p["prob_right_mean"] else "2"


def robustness(v):
    """Win rate under other tie rules and answer orders (iclr_clean_reproduction.tex)."""
    print("  robustness:")
    for t in (0.1, 0.3):
        votes = [three_way(x, t) for x in v]
        print(f"    Equal threshold {t}:        {fmt([SCORE[a] for a in votes])}, ties {votes.count('12')}")
    d2e = ["12" if x["winner_ab"] != x["winner_ba"] else x["averaged_judge_vote"] for x in v]
    print(f"    order flip -> Equal:        {fmt([SCORE[a] for a in d2e])}, ties {d2e.count('12')}")
    print(f"    ties dropped:               {fmt([SCORE[x['judge_vote']] for x in v if x['judge_vote'] != '12'])}")
    print(f"    AB only (model 1 first):    {fmt([SCORE[x['winner_ab']] for x in v])}")
    print(f"    BA only (model 2 first):    {fmt([SCORE[x['winner_ba']] for x in v])}")
    pick = np.random.default_rng(42).integers(0, 2, len(v))
    print(f"    random order per question:  "
          f"{fmt([SCORE[x['winner_ba'] if p else x['winner_ab']] for x, p in zip(v, pick)])}")


def main():
    runs = {}
    for f in sorted(glob.glob(str(RESULTS / "*_winner_logprob_bidir.jsonl"))):
        d = load(f)
        r0 = next(iter(d.values()))
        runs[(Path(r0["model_1"].removesuffix("+sys")).name,
              Path(r0["model_2"].removesuffix("+sys")).name, Path(r0["judge"]).name)] = d

    for (m1, m2, judge), d in runs.items():
        dec = [r for r in d.values() if r["judge_vote"] in SCORE]
        s = [SCORE[r["judge_vote"]] for r in dec]
        ratio = [words(r["model_1_answer"]) / max(1, words(r["model_2_answer"])) for r in dec]
        v = [r["verdict"] for r in dec]
        print(f"\n== {m1} vs {m2}, judge {judge}")
        print(f"questions {len(d)}, judged {len(dec)}: {m1} wins "
              f"{sum(x == 1 for x in s)}, {m2} {sum(x == 0 for x in s)}, ties {sum(x == .5 for x in s)}")
        print(f"  {m1} win rate (ties half):  {fmt(s)}")
        print(f"  both orders agree:          {100 * np.mean([x['orientation_consistent'] for x in v]):.1f}%")
        cons = [SCORE[x["winner_ab"]] for x in v if x["orientation_consistent"]]
        print(f"  win rate, consistent pairs: {fmt(cons)}")
        print(f"  mean P({m1} better):        {100 * np.mean([x['prob_left_mean'] for x in v]):.1f}%")
        print(f"  median words: {m1} {np.median([words(r['model_1_answer']) for r in d.values()]):.0f}, "
              f"{m2} {np.median([words(r['model_2_answer']) for r in d.values()]):.0f}")
        lo, hi = SIMILAR
        for name, keep in (("similar length", lambda q: lo <= q <= hi),
                           (f"{m1} >25% longer", lambda q: q > hi),
                           (f"{m1} >20% shorter", lambda q: q < lo)):
            print(f"  {name:<28}{fmt([x for x, q in zip(s, ratio) if keep(q)])}")
        for key in ("care_level", "source_set"):
            groups = {}
            for r, x in zip(dec, s):
                groups.setdefault(label(r["moove_meta"].get(key)), []).append(x)
            for g, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                if len(xs) >= 30:
                    print(f"  {key}={g:<40}{fmt(xs)}")
        robustness(v)

    # the two judges on the same answers
    pairs = sorted({(m1, m2) for m1, m2, _ in runs})
    for m1, m2 in pairs:
        js = sorted(j for a1, a2, j in runs if (a1, a2) == (m1, m2))
        if len(js) < 2:
            continue
        a, b = runs[(m1, m2, js[0])], runs[(m1, m2, js[1])]
        both = [p for p in a if a[p]["judge_vote"] in SCORE and b[p]["judge_vote"] in SCORE]
        agree = np.mean([a[p]["judge_vote"] == b[p]["judge_vote"] for p in both])
        print(f"\n{m1} vs {m2}: {js[0]} and {js[1]} agree on {100 * agree:.1f}% of {len(both)} questions")


if __name__ == "__main__":
    main()
