#!/usr/bin/env python3
# tech_reports/figures/iclr_pref.py

"""
Per-criterion preference results for iclr_clean_reproduction.tex (Table
tab:pref, Figure fig:iclr_pref): the pairs of Table tab:radar and the
Apertus-8B ablations, judged with --rubric moove_pref -- the rubric of the
criterion validation (Table tab:judge_validation, right) -- instead of Likert
scores. Runs: axolotl_config/teacher_ablation/launch_automoove.sh.

.venv/bin/python tech_reports/figures/iclr_pref.py

Writes iclr_pref.json, iclr_pref.pdf and iclr_pref_wtl.pdf next to this file and prints the table
rows. Win rate: MeditronFO's wins plus half the ties, in %, on one criterion;
an AB/BA disagreement is a tie. "Mean": the per-question mean of the five.
Intervals resample the questions (B = 2000); a difference from the full-corpus
model resamples them once for both.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # login node: 48 BLAS threads fail to spawn

import glob, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iclr_clean_figures as F      # colours, rcParams, point(), rows_axis(), save()

HERE = Path(__file__).resolve().parent
AM, TAG = F.AM, F.TAG
MOOVE = ("clinical_soundness", "safety", "contextual_fit", "appropriate_confidence", "clarity_completeness")
SCORE = {"1": 1.0, "12": 0.5, "2": 0.0}          # model 1 = MeditronFO
B = 2000
rng = np.random.default_rng(0)

A8 = "apertus_8b_meditronFO_gptoss_v1"
# (key, label, MeditronFO run name, base run name), base pairs in the order of tab:radar
PAIRS = [("apertus70b", "Apertus-70B", "apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-Instruct-2509"),
         ("olmo32b", "OLMo-2-32B", "olmo_2_32b_meditronFO_gptoss_v1", "OLMo-2-0325-32B-SFT"),
         ("euro22b", "EuroLLM-22B", "euro_llm_22b_meditronFO_gptoss_v1", "EuroLLM-22B-Instruct-2512"),
         ("apertus8b", "Apertus-8B", A8, "Apertus-8B-Instruct-2509"),
         ("euro9b", "EuroLLM-9B", "euro_llm_9b_meditronFO_gptoss_v1", "EuroLLM-9B-Instruct"),
         ("gemma27b", "Gemma-3-27B", "gemma_3_27b_meditronFO_gptoss_v1", "gemma-3-27b-it")]
ABLATIONS = [("no_curated_qa", "− Curated QA"), ("no_synthetic_qa", "− Synthetic Curated QA"),
             ("no_guidelines_qa", "− Guidelines QA"), ("no_synthetic_moove", "− Synthetic vignettes")]


def load(m1, m2, judge="gemma-4-31B-it"):
    """{question: {criterion: MeditronFO score 1 / 0.5 / 0}}, questions with every criterion parsed."""
    files = sorted(glob.glob(str(AM / f"*_auto_moove_{m1}{TAG}_{m2}{TAG}_{judge}_moove_pref_bidir.jsonl")))
    if not files:
        return None
    last = {}
    for line in open(files[-1], encoding="utf-8"):   # a requeued job's latest file is complete
        r = json.loads(line)
        last[r["prompt"]] = r
    return {q: {c: SCORE[r[f"pref_{c}"]] for c in MOOVE} for q, r in last.items()
            if all(r.get(f"pref_{c}") in SCORE for c in MOOVE)}


def interval(samples, pt):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(pt), float(lo), float(hi)]


def summarize(s):
    ids = list(s)
    idx = rng.integers(0, len(ids), (B, len(ids)))
    rec = {"n": len(ids)}
    for c in MOOVE + ("mean",):
        v = 100 * np.array([np.mean(list(s[i].values())) if c == "mean" else s[i][c] for i in ids])
        rec[c] = {"rate": interval(v[idx].mean(1), v.mean())}
        if c != "mean":
            x = np.array([s[i][c] for i in ids])
            rec[c] |= {"win": int((x == 1).sum()), "tie": int((x == 0.5).sum()), "loss": int((x == 0).sum())}
    return rec


def main():
    res, runs = {}, {}
    for k, _, m1, m2 in PAIRS:
        runs[k] = load(m1, m2)
    for k, _ in ABLATIONS:
        runs[k] = load(f"{A8}_{k}", "Apertus-8B-Instruct-2509")
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        print("no moove_pref run yet for:", ", ".join(missing))
    for k, v in runs.items():
        if v is not None:
            res[k] = summarize(v)
    # each ablation minus the full-corpus model on the mean, paired over shared questions
    if runs.get("apertus8b"):
        for k, _ in ABLATIONS:
            if runs.get(k) is None:
                continue
            ids = sorted(set(runs[k]) & set(runs["apertus8b"]))
            d = 100 * np.array([np.mean(list(runs[k][i].values())) - np.mean(list(runs["apertus8b"][i].values()))
                                for i in ids])
            idx = rng.integers(0, len(ids), (B, len(ids)))
            res[k]["mean_vs_full"] = interval(d[idx].mean(1), d.mean())
    # the Apertus-70B ablations, against the first 70B run;
    # last, so the draws above do not move. Added 26 September
    full70 = runs.get("apertus70b")
    for k, _ in ABLATIONS:
        r70 = load(f"apertus_70b_meditronFO_gptoss_v1_{k}", "Apertus-70B-Instruct-2509")
        if r70 is None or full70 is None:
            continue
        res[f"{k}_70b"] = summarize(r70)
        ids = sorted(set(r70) & set(full70))
        d = 100 * np.array([np.mean(list(r70[i].values())) - np.mean(list(full70[i].values())) for i in ids])
        idx = rng.integers(0, len(ids), (B, len(ids)))
        res[f"{k}_70b"]["mean_vs_full"] = interval(d[idx].mean(1), d.mean())
    json.dump(res, open(HERE / "iclr_pref.json", "w"), indent=1)
    table(res)
    figure(res, "iclr_pref_paper", F.PAPER, ablations=False)
    figure_combined(res)


def cell(v):
    return f"{v[0]:.1f} {{\\scriptsize[{v[1]:.1f}--{v[2]:.1f}]}}"


def table(res):
    print("\n% --- tab:pref rows ---")
    rows = [(k, lab) for k, lab, *_ in PAIRS] + [(k, "Apertus-8B " + lab.replace("−", "$-$")) for k, lab in ABLATIONS]
    for i, (k, lab) in enumerate(rows):
        if i == len(PAIRS):
            print("\\midrule")
        if k not in res:
            print(f"{lab} & \\multicolumn{{6}}{{c}}{{\\tbd}} \\\\")
            continue
        r = res[k]
        print(f"{lab} & " + " & ".join(cell(r[c]["rate"]) for c in MOOVE + ("mean",)) + f" & {r['n']} \\\\")
    print("% win / tie / loss per criterion")
    for k, lab in rows:
        if k in res:
            print(f"% {lab}: " + "; ".join(f"{c} {res[k][c]['win']}/{res[k][c]['tie']}/{res[k][c]['loss']}"
                                          for c in MOOVE))
    for k, lab in ABLATIONS:
        if "mean_vs_full" in res.get(k, {}):
            v = res[k]["mean_vs_full"]
            print(f"% {lab} mean minus full: {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]")


def figure(res, name="iclr_pref", out=None, ablations=True):
    """ablations=False: the base pairs only (the paper's Figure 6; its ablations are in their own table)."""
    crit = [("clinical_soundness", "Clinical\nsoundness"), ("safety", "Safety\n"),
            ("contextual_fit", "Contextual\nfit"), ("appropriate_confidence", "Appropriate\nconfidence"),
            ("clarity_completeness", "Clarity &\ncompleteness"), ("mean", "Mean of\nthe five")]
    rows = [(k, lab) for k, lab, *_ in PAIRS] + ([(k, "8B " + lab) for k, lab in ABLATIONS] if ablations else [])
    rows = [(k, lab) for k, lab in rows if k in res]
    if not rows:
        return
    nb = sum(k in res for k, *_ in PAIRS)
    ys = [i if i < nb else i + 0.6 for i in range(len(rows))]     # a gap before the ablations
    vals = [x for k, _ in rows for c, _ in crit for x in res[k][c]["rate"][1:]]
    lo, hi = 5 * np.floor(min(vals) / 5) - 1, 5 * np.ceil(max(vals) / 5) + 1
    fig, axes = plt.subplots(1, len(crit), figsize=(7.0, 0.8 + 0.27 * len(rows)), sharey=True, sharex=True)
    for ax, (c, title) in zip(axes, crit):
        ax.axvline(50, color=F.MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
        for y, (k, _) in zip(ys, rows):
            v = res[k][c]["rate"]
            # colour by the side of parity the interval supports; grey where it contains 50
            col = F.C1 if v[1] > 50 else F.C2 if v[2] < 50 else F.BASE
            F.point(ax, v, y, col, ms=4)
        ax.set_yticks(ys, [r[1] for r in rows])
        ax.set_ylim(ys[-1] + 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(lo, hi)
        ax.set_xticks(range(int(lo) + 10 - int(lo) % 10, int(hi) + 1, 10))
        ax.set_title(title, loc="left", color=F.INK, fontweight="bold" if c == "mean" else None)
        ax.tick_params(axis="x", labelsize=6.5)
        if nb < len(rows):
            ax.axhline(nb - 0.2, color=F.GRID, lw=0.8)
    # no axis label: the caption says what the x axis is
    fig.tight_layout(w_pad=0.6)
    F.save(fig, name, out or F.OUT)


def figure_wtl(res, name="iclr_pref_wtl", out=None, ablations=True):
    """figure() as win / tie / loss bars: the share of questions MeditronFO wins, ties or loses on each
    criterion; "Mean of the five": the five shares averaged, so win + tie / 2 is the mean win rate."""
    crit = [("clinical_soundness", "Clinical\nsoundness"), ("safety", "Safety\n"),
            ("contextual_fit", "Contextual\nfit"), ("appropriate_confidence", "Appropriate\nconfidence"),
            ("clarity_completeness", "Clarity &\ncompleteness"), ("mean", "Mean of\nthe five")]
    rows = [(k, lab) for k, lab, *_ in PAIRS] + ([(k, "8B " + lab) for k, lab in ABLATIONS] if ablations else [])
    rows = [(k, lab) for k, lab in rows if k in res]
    if not rows:
        return
    nb = sum(k in res for k, *_ in PAIRS)
    ys = [i if i < nb else i + 0.6 for i in range(len(rows))]
    share = lambda r, c: np.array([r[c]["win"], r[c]["tie"], r[c]["loss"]]) / r["n"] * 100
    # win, tie, loss in the colours of the paper's win / tie / loss figure
    colours = ((F.C1, "white"), (F.C3, F.INK), (F.C2, "white"))
    fig, axes = plt.subplots(1, len(crit), figsize=(7.0, 1.0 + 0.27 * len(rows)), sharey=True, sharex=True)
    for ax, (c, title) in zip(axes, crit):
        for y, (k, _) in zip(ys, rows):
            v = np.mean([share(res[k], m) for m in MOOVE], 0) if c == "mean" else share(res[k], c)
            left = 0
            for x, (col, tc) in zip(v, colours):
                ax.barh(y, x, left=left, height=0.7, color=col, edgecolor="white", lw=0.6, zorder=2)
                if x >= 13:
                    ax.text(left + x / 2, y, f"{x:.0f}", ha="center", va="center", color=tc, fontsize=6)
                left += x
        ax.axvline(50, color=F.INK, lw=0.6, ls=(0, (2, 2)), zorder=3, alpha=0.5)
        ax.set_yticks(ys, [r[1] for r in rows])
        ax.set_ylim(ys[-1] + 0.5, -0.5)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_title(title, loc="left", color=F.INK, fontweight="bold" if c == "mean" else None)
        ax.tick_params(axis="x", labelsize=6.5)
        if nb < len(rows):
            ax.axhline(nb - 0.2, color=F.GRID, lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col, _ in colours]
    F.legend_top(fig, (handles, ["MeditronFO wins", "tie", "base wins"]), x=0.55, ncol=3)
    # no axis label: the caption says what the x axis is
    fig.tight_layout(w_pad=0.6)
    F.save(fig, name, out or F.OUT)


def figure_combined(res, name="iclr_automoove_combined", out=None):
    """The paper's AutoMOOVE figure: the overall three-judge majority (iclr_clean_figures.wtl_majority)
    and the five Gemma-4 criteria (figure_wtl, without the mean) as one row of win / tie / loss panels
    over the base pairs. The Overall panel runs on below the others with the two MedGemma-27B pairs.
    No legend: the caption names the colours."""
    crit = [("overall", "Overall\n(majority of 3)"), ("clinical_soundness", "Clinical\nsoundness"),
            ("safety", "Safety\n"), ("contextual_fit", "Contextual\nfit"),
            ("appropriate_confidence", "Appropriate\nconfidence"), ("clarity_completeness", "Clarity &\ncompleteness")]
    base = [(k, lab) for k, lab, *_ in PAIRS if k in res]
    medg = [("gemma27b_medgemma", "Gemma-3-27B"), ("apertus70b_medgemma", "Apertus-70B")]
    wtl = F.CI["wtl"]
    gap = 0.8                     # rows between the base pairs and the MedGemma pairs, for their label
    y_med = [len(base) + gap + i for i in range(len(medg))]
    n_all = y_med[-1] + 1         # rows of the Overall panel

    def share(k, c):
        r = wtl[k]["majority_flip"] if c == "overall" else res[k][c]
        return np.array([r["win"], r["tie"], r["loss"]]) / (r["win"] + r["tie"] + r["loss"]) * 100

    colours = ((F.C1, "white"), (F.C3, F.INK), (F.C2, "white"))
    fig = plt.figure(figsize=(7.0, 0.75 + 0.23 * n_all))
    # hspace 0: the criterion panels end after the base pairs, the Overall panel spans both rows
    gs = fig.add_gridspec(2, len(crit), width_ratios=[1.35] + [1] * (len(crit) - 1),
                          height_ratios=[len(base), n_all - len(base)], hspace=0, wspace=0.22)

    def bars(ax, rows, ys, c, title, blocks):
        for y, (k, _) in zip(ys, rows):
            left = 0
            for x, (col, tc) in zip(share(k, c), colours):
                ax.barh(y, x, left=left, height=0.7, color=col, edgecolor="white", lw=0.6, zorder=2)
                if x >= 13:
                    ax.text(left + x / 2, y, f"{x:.0f}", ha="center", va="center", color=tc, fontsize=6)
                left += x
        for block in blocks:          # the 50% line per block of rows, not across the gap
            ax.plot([50, 50], [block[0] - 0.5, block[-1] + 0.5], color=F.INK, lw=0.6, ls=(0, (2, 2)), zorder=3,
                    alpha=0.5)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(title, loc="left", color=F.INK, fontsize=7, fontweight="bold" if c == "overall" else None)

    ax = fig.add_subplot(gs[:, 0])
    ys = list(range(len(base))) + y_med
    bars(ax, base + medg, ys, "overall", crit[0][1], [ys[:len(base)], y_med])
    ax.set_ylim(n_all - 0.5, -0.5)
    ax.set_yticks(ys, [lab for _, lab in base + medg])
    ax.text(0, len(base) - 0.5 + gap / 2, "vs MedGemma-27B", fontsize=6.5, color=F.INK2, style="italic",
            ha="left", va="center")
    for j, (c, title) in enumerate(crit[1:], 1):
        ax = fig.add_subplot(gs[0, j])
        bars(ax, base, range(len(base)), c, title, [range(len(base))])
        ax.set_ylim(len(base) - 0.5, -0.5)
        ax.tick_params(axis="y", labelleft=False)
    # no axis label: the caption says what the x axis is
    F.save(fig, name, out or F.PAPER)


if __name__ == "__main__":
    main()
