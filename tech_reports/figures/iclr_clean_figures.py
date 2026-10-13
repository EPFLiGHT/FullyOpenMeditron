#!/usr/bin/env python3
# tech_reports/figures/iclr_clean_figures.py

"""
Draft figures for tech_reports/iclr_clean_reproduction.tex, one per table that
reads better as a chart. Values and 95% bootstrap intervals come from
iclr_clean_ci.json (iclr_clean_ci.py, run it first); the length figure also
reads the per-question judge outputs of the Gemma-3-27B-MeditronFO (gpt-oss v1)
against MedGemma-27B run.

.venv/bin/python tech_reports/figures/iclr_clean_ci.py
.venv/bin/python tech_reports/figures/iclr_clean_figures.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # login node: 48 BLAS threads fail to spawn

import glob, json, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pymupdf
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CI = json.load(open(OUT / "iclr_clean_ci.json"))
AM = ROOT / "auto_moove/results/unanimous576"      # AutoMOOVE runs, as in iclr_clean_ci.py
TAG = ""
# log-prob judges against the physicians with an answer-order flip counted as a tie, the rule of the win
# rates (25 September). False: both orders averaged, the figures before that.
ORDER_FLIP_TIES = True

# Figure colours, fixed order, never cycled: dark blue and light purple carry the two main
# series (dark yellow is too light next to text), dark yellow and light blue the third and fourth
C1, C2, C3, C4 = "#0C43A0", "#AE3156", "#EBA547", "#68AFE7"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#9a9993"
BAND = "#F3DCE3"             # MedGemma interval, a light step of C2

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def save(fig, name, out=OUT):
    # Tectonic's xdvipdfmx cannot parse the cross-reference table of some matplotlib
    # PDFs ("Cannot parse cross-reference stream"); a PyMuPDF rewrite loads every time
    tmp = out / f".{name}.tmp.pdf"
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    with pymupdf.open(tmp) as doc:
        doc.save(out / f"{name}.pdf", garbage=3, deflate=True)
    tmp.unlink()


def rows_axis(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def point(ax, v, y, c, hollow=False, label=None, marker="o", ms=5):
    """(mean, lo, hi) as a dot on its interval."""
    m, lo, hi = v
    ax.plot([lo, hi], [y, y], color=BASE if hollow else c, lw=1.3, solid_capstyle="round", zorder=2)
    if hollow:
        ax.plot(m, y, marker, ms=ms, mfc="white", mec=BASE, mew=1.3, label=label, zorder=3)
    else:
        ax.plot(m, y, marker, ms=ms, color=c, mec="white", mew=1, label=label, zorder=3)


def fmt_ci(v, nd=1, sign=True):
    s = "+" if sign else ""
    return f"{v[0]:{s}.{nd}f} [{v[1]:{s}.{nd}f}, {v[2]:{s}.{nd}f}]".replace("-", "−")


def legend_top(fig, ax_or_handles, x=0.55, ncol=2, **kw):
    h, l = ax_or_handles if isinstance(ax_or_handles, tuple) else ax_or_handles.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(x, 1.0), ncol=ncol, handletextpad=0.2, **kw)


# --- Table tab:judge_validation -----------------------------------------------------------
def judge_validation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
    judges = [("winner_logprob_bidir_gemma4_31b", "Gemma-4-31B log-prob"),
              ("winner_generate_bidir_gemma4_31b", "Gemma-4-31B generated"),
              ("winner_logprob_bidir_qwen38_27b", "Qwen3.8-27B log-prob"),
              ("winner_generate_bidir_qwen38_27b", "Qwen3.8-27B generated"),
              ("winner_logprob_bidir_qwen3_30b", "Qwen3-30B-A3B log-prob"),
              ("winner_generate_bidir_qwen3_30b", "Qwen3-30B-A3B generated"),
              ("winner_logprob_bidir_museglimmer_30b", "Muse-Glimmer-30B log-prob"),
              ("winner_generate_bidir_museglimmer_30b", "Muse-Glimmer-30B generated"),
              ("winner_logprob_bidir_nemotron35_30b", "Nemotron-3.5-Lightning log-prob"),
              ("winner_generate_bidir_nemotron35_30b", "Nemotron-3.5-Lightning generated")]
    jo = CI["judge_overall"]
    for key, off, c, lab in (("ties", -0.15, C1, "with ties (n = 576)"), ("noties", 0.15, C2, "no ties")):
        for i, (tag, _) in enumerate(judges):
            point(a, jo[tag][key]["kappa"], i + off, c, label=lab if i == 0 else None)
    rows_axis(a, [j[1] for j in judges])
    for sep in (5.5,):
        a.axhline(sep, color=GRID, lw=0.8)
    a.set_xlim(0.35, 0.8)
    a.set_xlabel(r"Cohen's $\kappa$ with physicians (95% CI)")
    a.set_title("Overall verdict", loc="left", color=INK)
    a.legend(loc="lower center", bbox_to_anchor=(0.45, 1.08), ncol=1, handletextpad=0.2)

    crit = ["Clinical soundness", "Safety", "Contextual fit", "Approp. confidence", "Clarity & complet."]
    keys = ["clinical_soundness", "safety", "contextual_fit", "appropriate_confidence", "clarity_completeness"]
    series = [("judge, unanimous pairs", C1, "unanimous"), ("judge, per review", C2, "review"),
              ("physician vs physician", C3, "hh")]
    for k, (lab, c, s) in enumerate(series):
        for i, key in enumerate(keys):
            point(b, CI["judge_validation"][key][s], i + (k - 1) * 0.22, c, label=lab if i == 0 else None)
    rows_axis(b, crit)
    b.set_xlim(0.1, 0.75)
    b.set_xlabel(r"$\kappa$ per MOOVE criterion (Gemma-4-31B, moove_pref; 95% CI)")
    b.set_title("Per criterion", loc="left", color=INK)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(w_pad=2)
    save(fig, "iclr_judge_validation")


PAPER = ROOT / "papers" / "sept_25th_iclr_paper" / "figures"


def judge_validation_paper():
    """judge_validation for the paper's Figure 3: both panels side by side at the ICLR text width
    (5.5 in), so it is not scaled down, with shorter labels. Judges without data in iclr_clean_ci.json
    are skipped."""
    judges = [("winner_logprob_bidir_gemma4_31b", "Gemma-4 log-prob"),
              ("winner_generate_bidir_gemma4_31b", "Gemma-4 generated"),
              ("winner_logprob_bidir_qwen38_27b", "Qwen3.8 log-prob"),
              ("winner_generate_bidir_qwen38_27b", "Qwen3.8 generated"),
              ("winner_logprob_bidir_museglimmer_30b", "Muse-Glimmer log-prob"),
              ("winner_generate_bidir_museglimmer_30b", "Muse-Glimmer generated"),
              ("winner_logprob_bidir_nemotron35_30b", "Nemotron-3.5 log-prob"),
              ("winner_generate_bidir_nemotron35_30b", "Nemotron-3.5 generated"),
              # the majority of Gemma-4, Qwen3.8 and Muse-Glimmer, the vote of the main results
              ("winner_logprob_majority", "Majority log-prob"),
              ("winner_generate_majority", "Majority generated")]
    jo = {**CI["judge_overall"], **CI.get("judge_majority", {})}
    if ORDER_FLIP_TIES:
        jo.update(CI["judge_overall_flip"])
    judges = [j for j in judges if j[0] in jo]
    with plt.rc_context({"font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 6.8,
                         "legend.fontsize": 6.5}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [1, 1]})
        for key, off, c, lab in (("ties", -0.15, C1, "with ties"), ("noties", 0.15, C2, "no ties")):
            for i, (tag, _) in enumerate(judges):
                point(a, jo[tag][key]["kappa"], i + off, c, label=lab if i == 0 else None, ms=4)
        rows_axis(a, [j[1] for j in judges])
        n_early = sum(1 for t, _ in judges if "museglimmer" not in t and "nemotron" not in t)
        a.axhline(n_early - 0.5, color=GRID, lw=0.8)      # below: the two judges added on 23 September
        n_single = sum(1 for t, _ in judges if "majority" not in t)
        a.axhline(n_single - 0.5, color=GRID, lw=0.8)     # below: the three-judge majority
        a.set_xlim(0.35, 0.8)
        a.set_xlabel(r"$\kappa$, overall verdict")
        a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)

        crit = ["Clinical soundness", "Safety", "Contextual fit", "Approp. confidence", "Clarity & complet."]
        keys = ["clinical_soundness", "safety", "contextual_fit", "appropriate_confidence", "clarity_completeness"]
        series = [("Gemma-4, unanimous pairs", C1, "unanimous"), ("Gemma-4, per review", C2, "review"),
                  ("physician vs physician", C3, "hh")]
        for k, (lab, c, s_) in enumerate(series):
            for i, key in enumerate(keys):
                point(b, CI["judge_validation"][key][s_], i + (k - 1) * 0.22, c, label=lab if i == 0 else None, ms=4)
        rows_axis(b, crit)
        b.set_xlim(0.1, 0.75)
        b.set_xlabel(r"$\kappa$ per MOOVE criterion")
        b.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
        fig.tight_layout(w_pad=1.0)
        save(fig, "judge_validation_paper", PAPER)


# --- Table tab:judge_robustness -----------------------------------------------------------
def judge_robustness():
    settings = [("default", "Both orders, threshold 0 (default)"), ("thr0.1", "Equal threshold 0.1"),
                ("thr0.3", "Equal threshold 0.3"), ("flip_equal", "Order flip counts as Equal"),
                ("ties_dropped", "Ties dropped"), ("ab", "AB only (MeditronFO first)"),
                ("ba", "BA only (MedGemma first)"), ("random", "Random order per question")]
    if ORDER_FLIP_TIES:     # the flip row first, as the default (tab:wtl); ties dropped under the same rule
        settings = [("flip_equal", "Order flip counts as Equal (default)"), ("default", "Both orders averaged"),
                    ("thr0.1", "Equal threshold 0.1"), ("thr0.3", "Equal threshold 0.3"),
                    ("flip_ties_dropped", "Ties dropped"), ("ab", "AB only (MeditronFO first)"),
                    ("ba", "BA only (MedGemma first)"), ("random", "Random order per question")]
    models = [("apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-MeditronFO"),
              ("gemma_3_27b_meditronFO_gptoss_v1", "Gemma-3-27B-MeditronFO")]
    # the two judges of the paper, then the two added on 23 September (thinking off)
    judges = [("gemma-4-31B-it", "Gemma-4-31B", C1, "o"), ("Qwen3.8-27B", "Qwen3.8-27B", C2, "s"),
              ("Muse-Glimmer-30B", "Muse-Glimmer-30B", C3, "D"),
              ("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "Nemotron-3.5-Lightning", C4, "^")]
    offs = (-0.3, -0.1, 0.1, 0.3)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    for ax, (key, model) in zip(axes, models):
        ax.axvline(50, color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
        for (judge, name, c, mk), off in zip(judges, offs):
            rec = CI["robustness"][f"{key}|{judge}"]
            for i, (s_, _) in enumerate(settings):
                point(ax, rec[s_], i + off, c, marker=mk, ms=4.5, label=name if i == 0 else None)
        for sep in (0.5, 4.5):
            ax.axhline(sep, color=GRID, lw=0.8)
        rows_axis(ax, [s_[1] for s_ in settings])
        ax.set_xlim(10, 100)
        ax.set_title(f"{model}\nagainst MedGemma-27B", loc="left", color=INK)
        ax.set_xlabel("MeditronFO win rate, % (ties half; 95% CI)")
    legend_top(fig, axes[0], x=0.6, ncol=4, title="judge:", title_fontsize=7.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "iclr_judge_robustness")


# --- Table tab:judge_robustness_humans ----------------------------------------------------
def judge_robustness_humans(out=OUT, metric="kappa", legend="top"):
    """tab:judge_robustness_humans: each judge's Cohen's kappa (or, metric="f1", macro-F1) with the physicians'
    unanimous verdict under the tie rules and answer orders of tab:judge_robustness, 576 pairs. Equal thresholds
    exist only for log-prob scoring; for generated verdicts the default already counts an order flip as Equal.
    legend="right": the judges as a one-column list right of the panels, and a shorter figure (the paper's Figure 7)."""
    lo, hi = (0, 0.85) if metric == "kappa" else (0.15, 0.95)
    settings = [("default", "Both orders (default)"), ("thr0.1", "Equal threshold 0.1"),
                ("thr0.3", "Equal threshold 0.3"), ("flip_equal", "Order flip counts as Equal"),
                ("ties_dropped", "Ties dropped"), ("ab", "AB only"), ("ba", "BA only")]
    if ORDER_FLIP_TIES:     # the flip row first, as the default; averaging is then a log-prob variant
        settings = [("flip_equal", "Order flip counts as Equal (default)"), ("default", "Both orders averaged"),
                    ("thr0.1", "Equal threshold 0.1"), ("thr0.3", "Equal threshold 0.3"),
                    ("ties_dropped", "Ties dropped"), ("ab", "AB only"), ("ba", "BA only")]
    # the colours and markers of iclr_judge_robustness for its four judges
    judges = [("gemma4_31b", "Gemma-4-31B", C1, "o"), ("qwen38_27b", "Qwen3.8-27B", C2, "s"),
              ("qwen3_30b", "Qwen3-30B-A3B", INK2, "v"), ("gpt56sol_none", "GPT-5.6 Sol", MUTED, "P"),
              ("museglimmer_30b", "Muse-Glimmer-30B", C3, "D"), ("nemotron35_30b", "Nemotron-3.5-Lightning", C4, "^"),
              ("majority", "Majority (Gemma-4, Qwen3.8, Muse)", INK, "X")]
    rh = CI["robustness_humans"]
    key = lambda scoring, slug: f"winner_{scoring}_" + ("majority" if slug == "majority" else f"bidir_{slug}")
    right = legend == "right"
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 3.6) if right else (7.0, 3.9), sharey=True)
    handles = {}
    for ax, scoring, title in zip(axes, ("logprob", "generate"), ("Log-prob scoring", "Generated verdict")):
        present = [j for j in judges if key(scoring, j[0]) in rh]
        offs = np.linspace(-0.3, 0.3, len(present))
        for (slug, name, c, mk), off in zip(present, offs):
            rec = rh[key(scoring, slug)]
            for i, (s_, _) in enumerate(settings):
                # generated verdicts: their default already counts a flip as a tie, and there is nothing to average
                s_ = ({"flip_equal": "default", "default": None}.get(s_, s_)
                      if ORDER_FLIP_TIES and scoring == "generate" else s_)
                if ORDER_FLIP_TIES and scoring == "logprob" and s_ == "ties_dropped":
                    s_ = "flip_ties_dropped"      # the pairs both called a winner under the default rule
                if s_ in rec:
                    point(ax, rec[s_][metric], i + off, c, marker=mk, ms=4.5)
            handles[name] = ax.plot([], [], mk, color=c, mec="white", mew=1, ms=4.5)[0]
        if scoring == "generate":
            notes = (((1, "log-prob only"), (2, "log-prob only"), (3, "log-prob only")) if ORDER_FLIP_TIES
                     else ((1, "log-prob only"), (2, "log-prob only"), (3, "as the default row")))
            for i, note in notes:
                ax.text((lo + hi) / 2, i, note, ha="center", va="center", fontsize=7, color=MUTED)
        for sep in (0.5, 4.5):
            ax.axhline(sep, color=GRID, lw=0.8)
        # right legend: the paper's narrow version, labels on two lines to leave the panels the width
        wrap = {"Both orders (default)": "Both orders\n(default)", "Equal threshold 0.1": "Equal\nthreshold 0.1",
                "Equal threshold 0.3": "Equal\nthreshold 0.3", "Order flip counts as Equal": "Order flip\ncounts as Equal",
                "Order flip counts as Equal (default)": "Order flip counts\nas Equal (default)",
                "Both orders averaged": "Both orders\naveraged",
                "Ties dropped": "Ties\ndropped"} if right else {}
        rows_axis(ax, [wrap.get(s_[1], s_[1]) for s_ in settings])
        ax.set_xlim(lo, hi)
        ax.set_title(title, loc="left", color=INK)
    names = [j[1] for j in judges if j[1] in handles]
    if right:
        short = {"Nemotron-3.5-Lightning": "Nemotron-3.5-\nLightning",
                 "Majority (Gemma-4, Qwen3.8, Muse)": "Majority (Gemma-4,\nQwen3.8, Muse)"}
        fig.legend([handles[n] for n in names], [short.get(n, n) for n in names],
                   loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, handletextpad=0.2, labelspacing=0.8,
                   title="judge", title_fontsize=7.5, alignment="left")
    else:
        legend_top(fig, ([handles[n] for n in names], names), x=0.6, ncol=3, title="judge:", title_fontsize=7.5)
    xlabel = r"Cohen's $\kappa$" if metric == "kappa" else "Macro-F1 (Model 1 / Model 2 / Equal)"
    fig.supxlabel(xlabel + " with the physicians (95% CI)", x=0.55 if right else 0.6, fontsize=8, color=INK2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "iclr_judge_robustness_humans" + ("" if metric == "kappa" else "_f1"), out)


# --- Table tab:medbench -------------------------------------------------------------------
def medbench():
    rows = [("apertus70b_gptoss", "Apertus-70B"), ("olmo32b_ft", "OLMo-2-32B"), ("euro22b_ft", "EuroLLM-22B"), ("apertus8b_gptoss", "Apertus-8B"),
            ("euro9b_ft", "EuroLLM-9B"), ("gemma27b_ft", "Gemma-3-27B")]
    mb = CI["medbench"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    fig.subplots_adjust(wspace=0.55)
    for ax, metric, title, xl in zip(axes, ("mcqa", "hb"),
                                     ("Multiple-choice average (4 tasks)", "HealthBench full, Gemma-4-31B grader"),
                                     ((42, 69), (15, 60))):
        ref = mb["medgemma"][metric]
        ax.axvspan(ref[1], ref[2], color=BAND, lw=0, zorder=0)
        ax.axvline(ref[0], color=C2, lw=1, ls=(0, (3, 2)), zorder=1)
        ax.text(ref[0], -0.75, "MedGemma-27B ", color=INK2, fontsize=7, ha="right", va="bottom")
        for i, (k, _) in enumerate(rows):
            r = mb[k][metric]
            ax.plot([r["base"][0], r["ft"][0]], [i, i], color=C1, lw=1, alpha=0.35, zorder=1)
            point(ax, r["base"], i, BASE, hollow=True, label="base" if i == 0 else None)
            point(ax, r["ft"], i, C1, label="+ MeditronFO" if i == 0 else None)
            # the MCQA gain is the draw of Table tab:medbench, so figure and table print the same interval
            delta = CI["tables"]["medbench_gain"][k] if metric == "mcqa" else r["delta"]
            ax.text(1.02, i, "Δ " + fmt_ci(delta), va="center", fontsize=6.5, color=INK2,
                    transform=ax.get_yaxis_transform())
        for sep in (2.5, 4.5):
            ax.axhline(sep, color=GRID, lw=0.8)
        rows_axis(ax, [r[1] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -1.1)
        ax.set_xlim(*xl)
        ax.set_xlabel("accuracy / score, % (95% CI)")
        ax.set_title(title, loc="left", color=INK, pad=10)
    legend_top(fig, axes[0], x=0.5)
    save(fig, "iclr_medbench")
    shutil.copy(OUT / "iclr_medbench.pdf", PAPER)   # the tech report and the paper


# --- Table tab:general --------------------------------------------------------------------
def general():
    rows = [("apertus70b_gptoss", "Apertus-70B"), ("olmo32b_ft", "OLMo-2-32B"), ("euro22b_ft", "EuroLLM-22B"), ("apertus8b_gptoss", "Apertus-8B"),
            ("euro9b_ft", "EuroLLM-9B"), ("gemma27b_ft", "Gemma-3-27B")]
    g = CI["general"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.55, hspace=0.55)
    for ax, metric, title in zip(axes.flat, ("mmlu_pro", "ifeval", "arc", "avg"),
                                 ("MMLU-Pro", "IFEval (prompt-level strict)", "ARC-Challenge (acc_norm)",
                                  "General average (3 benchmarks)")):
        ref = g["medgemma"][metric]["ft"]
        ax.axvspan(ref[1], ref[2], color=BAND, lw=0, zorder=0)
        ax.axvline(ref[0], color=C2, lw=1, ls=(0, (3, 2)), zorder=1)
        ax.text(ref[0], -0.75, "MedGemma-27B ", color=INK2, fontsize=7, ha="right", va="bottom")
        for i, (k, _) in enumerate(rows):
            r = g[k][metric]
            ax.plot([r["base"][0], r["ft"][0]], [i, i], color=C1, lw=1, alpha=0.35, zorder=1)
            point(ax, r["base"], i, BASE, hollow=True, label="base" if i == 0 else None)
            point(ax, r["ft"], i, C1, label="+ MeditronFO" if i == 0 else None)
            ax.text(1.02, i, "Δ " + fmt_ci(r["delta"]), va="center", fontsize=6.5, color=INK2,
                    transform=ax.get_yaxis_transform())
        for sep in (2.5, 4.5):
            ax.axhline(sep, color=GRID, lw=0.8)
        rows_axis(ax, [r[1] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -1.1)
        lo = min(min(g[k][metric]["base"][1], g[k][metric]["ft"][1]) for k, _ in rows + [("medgemma", "")])
        hi = max(max(g[k][metric]["base"][2], g[k][metric]["ft"][2]) for k, _ in rows + [("medgemma", "")])
        ax.set_xlim(5 * np.floor(lo / 5) - 1, 5 * np.ceil(hi / 5) + 1)
        ax.set_xlabel("accuracy, % (95% CI)")
        ax.set_title(title, loc="left", color=INK, pad=10)
    legend_top(fig, axes[0, 0], x=0.5)
    save(fig, "iclr_general")


# --- Table tab:radar ----------------------------------------------------------------------
def criteria():
    crit = [("clinical_soundness", "Clinical"), ("safety", "Safety"), ("contextual_fit", "Context"),
            ("appropriate_confidence", "Confidence"), ("clarity_completeness", "Clarity")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
    for ax, (key, name) in zip(axes, (("apertus70b", "Apertus-70B"), ("gemma27b", "Gemma-3-27B"))):
        rec = CI["criteria"][key]
        for i, (c, _) in enumerate(crit):
            r = rec[c]
            col = C1 if r["delta"][0] > 0 else C2
            ax.plot([r["base"][0], r["ft"][0]], [i, i], color=col, lw=1, alpha=0.35, zorder=1)
            point(ax, r["base"], i, BASE, hollow=True, label="base" if i == 0 else None)
            point(ax, r["ft"], i, col, label="+ MeditronFO" if i == 0 else None)
            ax.text(2.02, i, "Δ " + fmt_ci(r["delta"], 2), va="center", fontsize=6.5, color=INK2)
        rows_axis(ax, [c[1] for c in crit])
        ax.set_xlim(0.3, 2.0)
        ax.set_xlabel("mean MOOVE criterion score, −2 to 2 (95% CI)")
        ax.set_title(f"{name}   (mean Δ {fmt_ci(rec['mean_delta'], 2)})", loc="left", color=INK)
    legend_top(fig, axes[0], x=0.5)
    fig.tight_layout(w_pad=6)
    save(fig, "iclr_criteria")


# --- Table tab:wtl (paper Figure 3, as a forest plot) --------------------------------------
# every MeditronFO model against its base, in the order of tab:medbench
BASE_PAIRS = [("apertus70b", "Apertus-70B", "Apertus-70B-Instruct"),
              ("olmo32b", "OLMo-2-32B", "OLMo-2-32B-SFT"), ("euro22b", "EuroLLM-22B", "EuroLLM-22B-Instruct"),
              ("apertus8b", "Apertus-8B", "Apertus-8B-Instruct"), ("euro9b", "EuroLLM-9B", "EuroLLM-9B-Instruct"),
              ("gemma27b", "Gemma-3-27B", "Gemma-3-27B-IT")]


def wtl():
    rows = BASE_PAIRS + [("gemma27b_medgemma", "Gemma-3-27B", "MedGemma-27B"),
                         ("apertus70b_medgemma", "Apertus-70B", "MedGemma-27B")]
    judges = [("gemma-4-31B-it", "Gemma-4-31B judge", C1, "o"), ("Qwen3.8-27B", "Qwen3.8-27B judge", C2, "s"),
              ("Muse-Glimmer-30B", "Muse-Glimmer-30B judge", C3, "D"),
              ("majority", "majority of the three", C4, "^")]
    w = CI["wtl"]
    nb = len(BASE_PAIRS)
    ys = list(range(nb)) + [nb + 1.0, nb + 2.1]           # a gap between the two blocks
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.axvline(50, color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.text(50, -0.95, "parity ", color=INK2, fontsize=7, ha="right", va="bottom")
    ax.text(1.02, -0.95, "win / tie / loss   win rate", fontsize=6.5, color=MUTED, va="bottom",
            transform=ax.get_yaxis_transform())
    for y, (k, ft, _) in zip(ys, rows):
        present = [j for j in judges if j[0] in w[k]]
        offs = {1: (0,), 2: (-0.19, 0.19), 3: (-0.27, 0, 0.27), 4: (-0.3, -0.1, 0.1, 0.3)}[len(present)]
        for (judge, lab, c, mk), off in zip(present, offs):
            r = w[k][judge]
            point(ax, r["rate"], y + off, c, marker=mk, ms=6.5 if mk == "^" else 5,
                  label=lab if k == "gemma27b_medgemma" else None)
            ax.text(1.02, y + off, f"{r['win']} / {r['tie']} / {r['loss']}   {r['rate'][0]:.1f}",
                    va="center", fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys, [f"{ft}  vs  {base}" for _, ft, base in rows])
    ax.set_ylim(ys[-1] + 0.5, -1.1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.axhline(nb - 0.1, color=GRID, lw=0.8)
    for y, t in ((-0.55, "MeditronFO against its base"), (nb + 0.35, "MeditronFO against MedGemma-27B")):
        ax.text(0.01, y, t, fontsize=7, color=MUTED, style="italic", va="center", zorder=4,
                transform=ax.get_yaxis_transform(), ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    lo = min(r["rate"][1] for v in w.values() for r in v.values())
    ax.set_xlim(min(38, 5 * np.floor(lo / 5) - 2), 90)
    # no axis label: the caption says what the x axis is
    legend_top(fig, ax, x=0.45, ncol=4)
    save(fig, "iclr_wtl")


# --- Table tab:radar (paper Figure 4, as forest plots of the paired differences) ----------


def wtl_majority(key="majority_flip", name="iclr_wtl_majority", paper=False):
    """The paper's original stacked win / tie / loss bars, with each judge's answer-order flip counted
    as a tie: the majority of the three judges (majority_flip in iclr_clean_ci.json), or one judge
    ("<judge>|flip"). No legend and no axis label: the caption gives the colours and the scale.
    paper=True: the paper's Figure 5, at the ICLR text width (5.5 in) so it is not scaled down."""
    if paper:
        with plt.rc_context({"font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}):
            return _wtl_majority(key, name, 4.25, 2.8, 6.5, PAPER)   # labels add ~1.25 in
    return _wtl_majority(key, name, 7.0, 3.6, 7.5, OUT)


def _wtl_majority(key, name, width, height, fs, out):
    w = CI["wtl"]
    win_c, tie_c, loss_c = C1, C3, C2        # brand colours, ties in the orange of the other figures
    blocks = [("MeditronFO against its base", [(k, ft, base) for k, ft, base in BASE_PAIRS]),
              ("MeditronFO against MedGemma-27B",
               [("gemma27b_medgemma", "Gemma-3-27B", "MedGemma-27B"),
                ("apertus70b_medgemma", "Apertus-70B", "MedGemma-27B")])]
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                             gridspec_kw={"height_ratios": [len(b[1]) for b in blocks], "hspace": 0.3})
    for ax, (title, rows) in zip(axes, blocks):
        for y, (k, ft, base) in enumerate(rows):
            r = w[k][key]
            n = r["win"] + r["tie"] + r["loss"]
            left = 0
            for cnt, c, tc in ((r["win"], win_c, "white"), (r["tie"], tie_c, INK), (r["loss"], loss_c, "white")):
                v = 100 * cnt / n
                ax.barh(y, v, left=left, height=0.62, color=c, edgecolor="white", lw=0.8, zorder=2)
                if v >= 6:
                    ax.text(left + v / 2, y, f"{v:.1f}%", ha="center", va="center", color=tc, fontsize=fs - 0.5)
                left += v
            ax.text(101.5, y, base, va="center", ha="left", fontsize=fs, color=INK)
        ax.set_yticks(range(len(rows)), [f"{ft}-MeditronFO" for _, ft, _ in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, -0.62, title, fontsize=fs - 0.5, color=MUTED, style="italic", va="bottom", ha="left",
                transform=ax.get_yaxis_transform())
    save(fig, name, out)


def criteria_forest():
    crit = [("clinical_soundness", "Clinical\nsoundness"), ("safety", "Safety\n"),
            ("contextual_fit", "Contextual\nfit"), ("appropriate_confidence", "Appropriate\nconfidence"),
            ("clarity_completeness", "Clarity &\ncompleteness"), ("mean_delta", "Mean of\nthe five")]
    rows = [(k, ft) for k, ft, _ in BASE_PAIRS]
    # Gemma-4 (circles) and, a little below, Qwen3.8 (squares) on the same answers
    judges = [(CI["criteria"], "o", -0.17, "Gemma-4-31B judge"), (CI["criteria_q38"], "s", 0.17, "Qwen3.8-27B judge")]
    lim = max(abs(x) for cr, *_ in judges for k, _ in rows for c, _ in crit
              for x in (cr[k][c] if c == "mean_delta" else cr[k][c]["delta"])[1:])
    lim = np.ceil(lim * 4) / 4
    fig, axes = plt.subplots(1, len(crit), figsize=(7.0, 3.0), sharey=True, sharex=True)
    for ax, (c, title) in zip(axes, crit):
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        for i, (k, _) in enumerate(rows):
            for cr, mk, off, _ in judges:
                v = cr[k][c] if c == "mean_delta" else cr[k][c]["delta"]
                # colour by the sign the interval supports; grey where it contains zero
                col = C1 if v[1] > 0 else C2 if v[2] < 0 else BASE
                point(ax, v, i + off, col, marker=mk, ms=4)
        rows_axis(ax, [r[1] for r in rows])
        ax.set_xlim(-lim, lim)
        ax.set_title(title, loc="left", color=INK, fontweight="bold" if c == "mean_delta" else None)
        ax.tick_params(axis="x", labelsize=6.5)
    for sep in (2.5, 4.5):
        for ax in axes:
            ax.axhline(sep, color=GRID, lw=0.8)
    fig.supxlabel("Δ mean MOOVE criterion score, MeditronFO − base (−2 to 2 scale; paired 95% CI, 516 questions)",
                  fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker=mk, ls="", color=INK2, ms=4.5, label=lab) for _, mk, _, lab in judges]
    fig.tight_layout(w_pad=0.6)
    legend_top(fig, (handles, [h.get_label() for h in handles]), x=0.5)
    save(fig, "iclr_criteria_forest")


# --- Table tab:ablations ------------------------------------------------------------------
def ablations():
    configs = [("no_curated_qa", "− Curated QA"), ("no_synthetic_qa", "− Synthetic Curated QA"),
               ("no_guidelines_qa", "− Guidelines QA"), ("no_synthetic_moove", "− Synthetic vignettes")]
    panels = [("mcqa", "MCQA avg"), ("hb", "HealthBench"), ("automoove", "AutoMOOVE win rate"),
              ("likert", "Δ Likert")]
    ab = CI["ablations"]
    if ab[configs[0][0]].get("likert") is None:     # no Likert run of the ablations
        panels = panels[:3]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9), sharey=True)
    for ax, (k, title) in zip(axes, panels):
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        lim = 0
        for i, (c, _) in enumerate(configs):
            v = ab[c][k]
            point(ax, v, i, C1)
            lim = max(lim, abs(v[1]), abs(v[2]))
        rows_axis(ax, [c[1] for c in configs])
        ax.set_xlim(-1.1 * lim, 1.1 * lim)
        ax.set_title(title, loc="left", color=INK)
    fig.supxlabel("change against the full-corpus model, paired 95% CI (points; score for Δ Likert)",
                  fontsize=8, color=INK2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "iclr_ablations")


def ablations_teacher():
    """The paper's ablation figure: the corpus-component and second-teacher ablations in one, drawn as
    teacher() with each model's own score and 95% interval. Both blocks: the base, the two teachers (at
    70B their first runs; the second runs are in the appendix) and the gpt-oss corpus with one component
    removed. HB hard and professional stay in the appendix (teacher())."""
    a, t, a70 = CI["ablations_abs"], CI["teacher"], CI["ablations70_abs"]
    components = (("no_curated_qa", "− Curated QA"), ("no_synthetic_qa", "− Synthetic Curated QA"),
                  ("no_guidelines_qa", "− Guidelines QA"), ("no_synthetic_moove", "− Synthetic vignettes"))
    tv = lambda k, m: t[k]["full_gemma4" if m == "hb" else m] if m != "automoove" else a.get(k, {}).get(m)
    # (label, colour, hollow, value(metric)); the ablation values come from ablations_abs, the rest from teacher
    blocks = [
        ("Apertus-8B", [("Apertus-8B-Instruct", BASE, True, lambda m: a["base"].get(m)),
                        ("gpt-oss-120b teacher", C1, False, lambda m: a["full"].get(m)),
                        ("Qwen3.8-27B teacher", C2, False, lambda m: tv("apertus8b_qwen", m))]
         + [(lab, C4, False, lambda m, c=c: a[c].get(m)) for c, lab in components]),
        ("Apertus-70B", [("Apertus-70B-Instruct", BASE, True, lambda m: tv("apertus70b", m)),
                         ("gpt-oss-120b teacher", C1, False, lambda m: tv("apertus70b_gptoss", m)),
                         ("Qwen3.8-27B teacher", C2, False, lambda m: tv("apertus70b_qwen", m))]
         + [(lab, C4, False, lambda m, c=c: a70[c].get(m)) for c, lab in components]),
    ]
    metrics = [("mcqa", "MCQA avg"), ("hb", "HB full"), ("ifeval", "IFEval")]
    fig = plt.figure(figsize=(7.0, 3.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[3, 1.25], wspace=0.55)
    left = outer[0].subgridspec(2, len(metrics), height_ratios=[len(b[1]) for b in blocks], hspace=0.12,
                                wspace=0.12)
    axes = np.empty((2, len(metrics)), dtype=object)
    for row in range(2):
        for col in range(len(metrics)):
            axes[row, col] = fig.add_subplot(left[row, col], sharey=axes[row, 0] if col else None,
                                             sharex=axes[0, col] if row else None)
    for col, (m, name) in enumerate(metrics):
        xs = []
        for row, (size, rows) in enumerate(blocks):
            ax = axes[row, col]
            for i, (_, c, hollow, val) in enumerate(rows):
                v = val(m)
                point(ax, v, i, c, hollow=hollow)
                xs += v[1:]
            rows_axis(ax, [r[0] for r in rows])
            ax.axhline(2.5, color=GRID, lw=0.8)
            if col == 0:
                ax.set_ylabel(size, color=INK, fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == 0:
                ax.set_title(name, loc="left", color=INK)
                ax.tick_params(axis="x", labelbottom=False)
        pad = (max(xs) - min(xs)) * 0.08
        axes[0, col].set_xlim(min(xs) - pad, max(xs) + pad)
        axes[1, col].set_xlabel("% (95% CI)")
    # right: the HealthBench full gain of each finetune over its base, raw and after HealthBench
    # Professional's length adjustment (healthbench_breakdown_figures.length_fig, panel b)
    D = json.load(open(OUT / "healthbench_breakdown.json"))["splits"]["full"]["diffs"]
    hb_rows = [(f"{k}_ft-{k}", f"{lab}-MFO") for k, lab, _ in BASE_PAIRS] + [("medgemma-gemma27b", "MedGemma-27B")]
    ax = fig.add_subplot(outer[1])
    for i, (pk, _) in enumerate(hb_rows):
        point(ax, D[pk]["overall"], i - 0.15, C1, label="raw" if i == 0 else None)
        point(ax, D[pk]["score_len_adj"], i + 0.15, C2, label="length-adjusted" if i == 0 else None,
              marker="s", ms=4)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(len(hb_rows) - 1.5, color=GRID, lw=0.8)
    rows_axis(ax, [r[1] for r in hb_rows])
    ax.set_ylim(len(hb_rows) - 0.5, -1.4)      # room above the first row for the legend
    ax.set_xlim(-13, 18)
    ax.set_title("HB full gain over base", loc="left", color=INK)
    ax.set_xlabel("points (95% CI)")
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.2, fontsize=6.5)
    save(fig, "iclr_ablations_teacher", PAPER)


# --- Table tab:teacher --------------------------------------------------------------------
def teacher(pro=True, out=OUT):
    """pro=False drops the HB professional panel (the paper's copy)."""
    t = CI["teacher"]
    metrics = [("mcqa", "MCQA avg"), ("full_gemma4", "HB full"), ("hard_gemma4", "HB hard"),
               ("professional_gemma4", "HB professional"), ("ifeval", "IFEval")]
    if not pro:
        metrics = [m for m in metrics if m[0] != "professional_gemma4"]
    # size: [(key, colour, y offset, hollow, marker, legend label)]
    layout = {
        "8B": [("apertus8b", BASE, -0.25, True, "o", "base"), ("apertus8b_gptoss", C1, -0.08, False, "o", "gpt-oss-120b"),
               ("apertus8b_qwen", C2, 0.08, False, "o", "Qwen3.8-27B")],
        "70B": [("apertus70b", BASE, -0.3, True, "o", "base"),
                ("apertus70b_gptoss", C1, -0.18, False, "o", "gpt-oss-120b"),
                ("apertus70b_gptoss_rep", C1, -0.08, False, "o", None),
                ("apertus70b_qwen", C2, 0.04, False, "o", "Qwen3.8-27B"),
                ("apertus70b_qwen_rep", C2, 0.14, False, "o", None)],
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.1), sharey=True)
    for ax, (m, name) in zip(axes, metrics):
        xs = []
        for i, (size, pts) in enumerate(layout.items()):
            for key, c, off, hollow, mk, lab in pts:
                v = t[key][m]
                point(ax, v, i + off, c, hollow=hollow, marker=mk, ms=4.5 if mk == "D" else 5,
                      label=lab if (m == "mcqa" and size == "70B") else None)
                xs += v[1:]
        rows_axis(ax, list(layout))
        ax.set_ylim(1.5, -0.5)
        ax.set_title(name, loc="left", color=INK)
        pad = (max(xs) - min(xs)) * 0.08
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_xlabel("% (95% CI)")
    legend_top(fig, axes[0], x=0.5, ncol=3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "iclr_teacher", out)


# --- Table tab:length ---------------------------------------------------------------------
def length():
    f = sorted(glob.glob(str(AM / f"*gemma_3_27b_meditronFO_gptoss_v1{TAG}_medgemma-27b-text-it{TAG}_gemma-4-31B-it"
                                  "_winner_logprob_bidir.jsonl")))[-1]
    last = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        last[r["prompt"]] = r          # last copy of each question, as in unanimous576_h2h_tables.py
    d = [r for r in last.values() if r["judge_vote"] in ("1", "12", "2")]
    words = lambda s: len((s or "").split())
    ratio = np.array([words(r["model_1_answer"]) / max(1, words(r["model_2_answer"])) for r in d])
    margin = np.array([r["verdict"]["prob_left_mean"] - r["verdict"]["prob_right_mean"] for r in d])
    score = np.array([{"1": 1.0, "12": 0.5, "2": 0.0}[r["judge_vote"]] for r in d])
    lr = np.log2(ratio)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a.axhline(0, color=MUTED, lw=1)
    a.axvline(0, color=MUTED, lw=1)
    a.scatter(lr, margin, s=9, color=C1, alpha=0.55, lw=0)
    a.set_xlim(-2.5, 2.5)
    a.set_xlabel("log₂ word ratio, MeditronFO / MedGemma")
    a.set_ylabel("P(MeditronFO) − P(MedGemma)")
    a.set_title(f"Preference margin against length (n = {len(d)}, r = {fmt_ci(CI['length']['r'], 2, False)})",
                loc="left", color=INK)

    rng = np.random.default_rng(0)
    bins = [("MeditronFO >20% shorter", ratio < 0.8), ("similar (0.8–1.25×)", (ratio >= 0.8) & (ratio <= 1.25)),
            ("MeditronFO >25% longer", ratio > 1.25)]
    b.axvline(50, color=MUTED, lw=1, ls=(0, (3, 2)))
    for i, (lab, m) in enumerate(bins):
        x = score[m]
        boots = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
        point(b, (100 * x.mean(), lo, hi), i, C1)
        b.text(hi + 1, i, f"{100 * x.mean():.1f}  (n = {len(x)})", va="center", fontsize=7, color=INK2)
    rows_axis(b, [x[0] for x in bins])
    b.set_xlim(25, 85)
    b.set_xlabel("MeditronFO win rate, % (ties half; 95% CI)")
    b.set_title("Win rate by length bin", loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    save(fig, "iclr_length")


if __name__ == "__main__":
    judge_validation()                  # also draws judge_validation_paper
    judge_robustness()                  # copy iclr_judge_robustness.pdf into the paper's figures
    judge_robustness_humans(PAPER, legend="right")
    judge_robustness_humans(PAPER, metric="f1")
    medbench()                          # copy iclr_medbench.pdf into the paper's figures
    for judge, slug in (("gemma-4-31B-it", "gemma4"), ("Qwen3.8-27B", "qwen38"), ("Muse-Glimmer-30B", "muse")):
        wtl_majority(f"{judge}|flip", f"iclr_wtl_{slug}", paper=True)
    teacher(pro=False, out=PAPER)
    ablations_teacher()
