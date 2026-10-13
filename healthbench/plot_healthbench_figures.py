"""The paper's Figure 1: open models on HealthBench full (Gemma-4-31B grader) against parameter count,
coloured by openness, with the Pareto frontiers of the open-weight and the fully-open models and an arrow
from each base to its MeditronFO finetune. Intervals are in the appendix.

Scores are read from healthbench/results/<run>/results_<model>_<subset>.json.

    .venv/bin/python healthbench/plot_healthbench_figures.py [--results full_gemma4_31b]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

REPO = Path(__file__).resolve().parents[1]

# Figure colours: closed = light purple, open weights = dark yellow, fully open = dark blue.
# Dark yellow is too light for text, so its labels use the band text's dark step
CLOSED, OPENW, FULLY = "#AE3156", "#EBA547", "#0C43A0"
BAND = {CLOSED: "#F3DCE3", OPENW: "#FBEBD4", FULLY: "#DCE6F4"}
BAND_TEXT = {CLOSED: "#79123F", OPENW: "#8A5A10", FULLY: "#0C43A0"}
INK, MUTED = "#1a1a1a", "#6b6b66"
LABEL = {CLOSED: CLOSED, OPENW: BAND_TEXT[OPENW], FULLY: FULLY}   # colour of the "best ..." line labels

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
})

# run name in healthbench_summary.csv -> (display name, params in B, openness, role)
# role: base = general instruct model, fo = our finetune, med = earlier medical model
# openness follows the paper's appendix table tab:fully_open_comparison1 (EuroLLM and OLMo-2 are fully open)
MODELS = {
    "apertus8b_instruct":           ("Apertus-8B-Instruct", 8, FULLY, "base"),
    "meditronFO_gptoss_8b":         ("Apertus-8B-MeditronFO", 8, FULLY, "fo"),
    "base_euro9b_instruct":         ("EuroLLM-9B-Instruct", 9, FULLY, "base"),
    "meditronFO_gptoss_euro9b":     ("EuroLLM-9B-MeditronFO", 9, FULLY, "fo"),
    "base_euro22b_instruct":        ("EuroLLM-22B-Instruct", 22, FULLY, "base"),
    "meditronFO_gptoss_euro22b":    ("EuroLLM-22B-MeditronFO", 22, FULLY, "fo"),
    "base_gemma3_27b_it":           ("Gemma-3-27B-IT", 27, OPENW, "base"),
    "meditronFO_gptoss_gemma3_27b": ("Gemma-3-27B-MeditronFO", 27, OPENW, "fo"),
    "base_medgemma27b":             ("MedGemma-27B", 27, OPENW, "med"),
    "base_olmo32b_sft":             ("OLMo-2-32B-SFT", 32, FULLY, "base"),
    "meditronFO_gptoss_olmo32b":    ("OLMo-2-32B-MeditronFO", 32, FULLY, "fo"),
    "apertus70b_instruct":          ("Apertus-70B-Instruct", 70, FULLY, "base"),
    "meditronFO_gptoss_70b_16n":    ("Apertus-70B-MeditronFO", 70, FULLY, "fo"),
    "ref_meditron3_70b":            ("Llama-3.1-Meditron-3-70B", 70, OPENW, "med"),
    "ref_aloe_beta_70b":            ("Llama-3.1-Aloe-Beta-70B", 70, OPENW, "med"),
}
FINETUNE_OF = {
    "meditronFO_gptoss_8b": "apertus8b_instruct",
    "meditronFO_gptoss_euro9b": "base_euro9b_instruct",
    "meditronFO_gptoss_euro22b": "base_euro22b_instruct",
    "meditronFO_gptoss_gemma3_27b": "base_gemma3_27b_it",
    "meditronFO_gptoss_olmo32b": "base_olmo32b_sft",
    "meditronFO_gptoss_70b_16n": "apertus70b_instruct",
}
# Figure 1d only (launch.sh openweights): more earlier medical models, and general open-weight models,
# the Gemma-3 and Llama-3.1 instruct models of the medical models' families and gpt-oss-120b (our teacher;
# a mixture of experts, placed at its 117B total parameters). role gen = general open-weight model.
EXTRA = {
    "base_gemma3_4b_it":            ("Gemma-3-4B-IT", 4, OPENW, "gen"),
    "ref_medgemma4b":               ("MedGemma-4B", 4, OPENW, "med"),
    "base_llama31_8b_instruct":     ("Llama-3.1-8B-Instruct", 8, OPENW, "gen"),
    "ref_aloe_beta_8b":             ("Llama-3.1-Aloe-Beta-8B", 8, OPENW, "med"),
    "ref_meditron3_8b":             ("Llama-3.1-Meditron-3-8B", 8, OPENW, "med"),
    "base_llama31_70b_instruct":    ("Llama-3.1-70B-Instruct", 70, OPENW, "gen"),
    "ref_gptoss_120b":              ("gpt-oss-120b", 117, OPENW, "gen"),
}
# medical model -> general instruct model of the same family. Only Meditron-3 was trained from that
# instruct model; Aloe-Beta starts from the Llama-3.1 pretrained model and MedGemma from Gemma-3
# pretrained ones, so figure 1d draws these as links, not base -> finetune arrows.
SAME_FAMILY = {
    "ref_medgemma4b": "base_gemma3_4b_it",
    "ref_aloe_beta_8b": "base_llama31_8b_instruct",
    "ref_meditron3_8b": "base_llama31_8b_instruct",
    "base_medgemma27b": "base_gemma3_27b_it",
    "ref_aloe_beta_70b": "base_llama31_70b_instruct",
    "ref_meditron3_70b": "base_llama31_70b_instruct",
}


def hb_examples(results_dir):
    """{run: (score, per-conversation scores)} from results_<run>_<subset>.json. The score is the
    file's own (the clipped mean), not healthbench_summary.csv's last row, which for professional is
    the length-adjusted score."""
    out = {}
    for run in [*MODELS, *EXTRA]:
        # results_<run>_<subset>.json exactly: a bare glob also caught meditronFO_gptoss_8b's
        # ablations (results_meditronFO_gptoss_8b_no_synthetic_qa_full.json), and the answers
        # (results_<run>_full.completions.json)
        p = next((q for q in results_dir.glob(f"results_{run}_*.json")
                  if q.stem[len(f"results_{run}_"):].isalnum()), None)
        if p is not None:
            d = json.load(open(p))
            out[run] = (100 * d["score"], np.array([e["score"] for e in d["metadata"]["example_level_metadata"]]))
    return out


def load_scores(results_dir, n_boot=2000):
    """Scores (%) and 95% percentile-bootstrap intervals over the conversations (2000 resamples,
    seed 0, clipped to [0, 1] as simple-evals does), the same intervals as
    tech_reports/figures/iclr_clean_ci.py."""
    rng = np.random.default_rng(0)
    scores, ci = {}, {}
    for run, (score, ex) in hb_examples(results_dir).items():   # EXTRA last: MODELS keep their resamples
        scores[run] = score
        boot = np.clip(ex[rng.integers(0, len(ex), (n_boot, len(ex)))].mean(axis=1), 0, 1)
        ci[run] = tuple(100 * np.percentile(boot, [2.5, 97.5]))
    return scores, ci


# medbench runs (eval_results/<dir>) behind figure 1f: those of tech_reports/figures/iclr_clean_ci.py,
# and, for the EXTRA models, runs of 23 September (the May runs of the Gemma-3-4B pair had a
# 500-question PubMedQA; the task now has 1000). The Gemma family runs without a chat template.
MEDBENCH = {
    "apertus70b_instruct": "eval-gen_apertus-70b-instruct-2509_3478465",
    "meditronFO_gptoss_70b_16n": "eval-gen_apertus_70b_meditronfo_gptoss_v1_16n_3465533",
    "apertus8b_instruct": "eval-gen_apertus-8b-instruct-2509_3478462",
    "meditronFO_gptoss_8b": "eval-gen_apertus_8b_meditronfo_gptoss_v1_3464365",
    "base_olmo32b_sft": "eval-gen_olmo-2-0325-32b-sft_3485004",
    "meditronFO_gptoss_olmo32b": "eval-gen_olmo_2_32b_meditronfo_gptoss_v1_3467269",
    "base_euro22b_instruct": "eval-gen_eurollm-22b-instruct-2512_3484596",
    "meditronFO_gptoss_euro22b": "eval-gen_euro_llm_22b_meditronfo_gptoss_v1_3467274",
    "base_euro9b_instruct": "eval-gen_eurollm-9b-instruct_3484597",
    "meditronFO_gptoss_euro9b": "eval-gen_euro_llm_9b_meditronfo_gptoss_v1_3467276",
    "base_gemma3_27b_it": "eval-gen_gemma-3-27b-it_3484598",
    "meditronFO_gptoss_gemma3_27b": "eval-gen_gemma_3_27b_meditronfo_gptoss_v1_3474465",
    "base_medgemma27b": "eval-gen_medgemma-27b-text-it_3489833",
    "base_gemma3_4b_it": "*_3498976",
    "ref_medgemma4b": "*_3499959",
    "base_llama31_8b_instruct": "*_3495854",
    "base_llama31_70b_instruct": "*_3495855",
    "ref_aloe_beta_8b": "*_3498619",
    "ref_aloe_beta_70b": "*_3499960",
}
MEDBENCH_TASKS = ("medmcqa_g", "medqa_g", "pubmedqa_g", "medxpertqa_g")


def medbench_examples(run):
    """{task: per-question 0/1} of a run's medbench, from the samples rescored with the fixed answer
    filter (scripts/rescore_medbench.py) or, for runs after the fix, the plain samples; the latest
    results file of the run. None if the run has no medbench."""
    d = next(iter(sorted((REPO / "eval_results").glob(MEDBENCH.get(run, "-")))), None)
    if d is None:
        return None
    res = sorted(p for p in d.glob("*/results_*.json") if not p.stem.endswith("_rescored"))
    if not res:
        return None
    ts = res[-1].stem[len("results_"):]
    out = {}
    for t in MEDBENCH_TASKS:
        f = next(d.glob(f"*/samples_{t}_{ts}_rescored.jsonl"), None) or next(d.glob(f"*/samples_{t}_{ts}.jsonl"))
        out[t] = np.array([json.loads(l)["exact_match"] for l in open(f)], dtype=float)  # one filter per task
    return out


def load_medavg(hb_dir, n_boot=2000):
    """Figure 1f: mean of the four medbench accuracies and HealthBench full (%), with a 95% bootstrap
    interval that resamples every task's items independently."""
    rng = np.random.default_rng(0)
    hb = hb_examples(hb_dir)
    scores, ci = {}, {}
    for run in [*MODELS, *EXTRA]:
        mb = medbench_examples(run)
        if mb is None or run not in hb:
            continue
        parts = [*mb.values()]
        scores[run] = 100 * (sum(x.mean() for x in parts) + hb[run][0] / 100) / 5
        boot = sum(x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1) for x in parts)
        ex = hb[run][1]
        boot = boot + np.clip(ex[rng.integers(0, len(ex), (n_boot, len(ex)))].mean(axis=1), 0, 1)
        ci[run] = tuple(100 * np.percentile(boot / 5, [2.5, 97.5]))
    return scores, ci


def errbar(ax, x, run, scores, ci, color, **kw):
    lo, hi = ci[run]
    y = scores[run]
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=1.2, capsize=3,
                capthick=1.2, zorder=kw.pop("zorder", 4.5), **kw)


# ---------------------------------------------------------------- Figure 1: timeline

def label(ax, x, y, name, size, value, dx, dy, ha="center", bold=True):
    """Name (bold), size (small italic) and score stacked, offset (dx, dy) points from the point."""
    kw = dict(xycoords="data", textcoords="offset points", ha=ha, va="bottom", color=INK)
    ax.annotate(f"{value:.1f}", (x, y), xytext=(dx, dy), fontsize=10, **kw)
    off = dy + 11
    if size:
        ax.annotate(size, (x, y), xytext=(dx, off), fontsize=6.5, style="italic", **kw)
        off += 7
    ax.annotate(name, (x, y), xytext=(dx, off), fontsize=9.5,
                fontweight="bold" if bold else "normal", **kw)


def fig_timeline(scores, ci, out):
    s = scores
    closed = [("GPT-4o", "", date(2023, 11, 1), 32.0, (0, 4)),
              ("o1", "", date(2024, 9, 12), 42.0, (0, 4)),
              ("o3", "", date(2025, 4, 16), 60.0, (-22, 4)),
              ("GPT-5 thinking", "", date(2025, 8, 7), 62.6, (0, 4))]
    this = date(2026, 5, 1)
    openw = [("Llama-3.1-Meditron-3", "70B", date(2024, 8, 1), s["ref_meditron3_70b"], (-40, -2)),
             ("Llama-3.1-Aloe-Beta", "70B", date(2024, 12, 1), s["ref_aloe_beta_70b"], (0, -56)),
             ("MedGemma", "27B", date(2025, 5, 20), s["base_medgemma27b"], (-14, -58)),
             ("Gemma-MeditronFO", "27B", this, s["meditronFO_gptoss_gemma3_27b"], None)]
    apertus = [("Apertus", "70B", date(2025, 9, 2), s["apertus70b_instruct"], (10, 9)),
               ("Apertus-MeditronFO", "70B", this, s["meditronFO_gptoss_70b_16n"], None)]
    apertus8 = [("Apertus", "8B", date(2025, 9, 2), s["apertus8b_instruct"], (-18, -58)),
                ("Apertus-MeditronFO", "8B", this, s["meditronFO_gptoss_8b"], None)]

    fig, ax = plt.subplots(figsize=(11.1, 5.1))
    x0, x1 = date(2023, 8, 1), date(2026, 9, 1)
    for a, b, c, txt in [(x0, date(2024, 6, 15), CLOSED, "Large Proprietary\nModels"),
                         (date(2024, 6, 15), date(2025, 7, 1), OPENW, "Open Weights\nModels"),
                         (date(2025, 7, 1), x1, FULLY, "Fully Open\nModels")]:
        ax.axvspan(a, b, color=BAND[c], zorder=0, lw=0)
        mid = a + (b - a) / 2
        ax.text(mid, 3, txt, ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                color=BAND_TEXT[c], zorder=1)

    for pts, color in [(closed, CLOSED), (openw, OPENW), (apertus, FULLY), (apertus8, FULLY)]:
        ax.plot([p[2] for p in pts], [p[3] for p in pts], "-o", color=color, lw=2.2, ms=6.5, zorder=4)
        for name, size, x, y, off in pts:
            if off is not None:
                label(ax, x, y, name, size, y, *off)

    for x, run, c in [(date(2024, 8, 1), "ref_meditron3_70b", OPENW), (date(2024, 12, 1), "ref_aloe_beta_70b", OPENW),
                      (date(2025, 5, 20), "base_medgemma27b", OPENW), (this, "meditronFO_gptoss_gemma3_27b", OPENW),
                      (date(2025, 9, 2), "apertus70b_instruct", FULLY), (this, "meditronFO_gptoss_70b_16n", FULLY),
                      (date(2025, 9, 2), "apertus8b_instruct", FULLY), (this, "meditronFO_gptoss_8b", FULLY)]:
        errbar(ax, x, run, scores, ci, c)

    # right-hand "This work" column: labels ordered by score, spread apart if they collide
    ours = sorted([openw[-1], apertus[-1], apertus8[-1]], key=lambda p: -p[3])
    ys, prev = [], None
    for p in ours:
        y = p[3] if prev is None else min(p[3], prev - 8.2)
        ys.append(y)
        prev = y
    for p, y in zip(ours, ys):
        ax.annotate(p[0], (this, y), xytext=(14, 4), textcoords="offset points", fontsize=9.5,
                    fontweight="bold", va="bottom", color=INK)
        ax.annotate(p[1], (this, y), xytext=(14, -3), textcoords="offset points", fontsize=6.5,
                    style="italic", va="bottom", color=INK)
        ax.annotate(f"{p[3]:.1f}", (this, y), xytext=(14, -13), textcoords="offset points",
                    fontsize=10, va="bottom", color=INK)
    lo, hi = min(p[3] for p in ours), max(p[3] for p in ours)
    ax.plot([this - (x1 - x0) * 0.004] * 2, [lo - 1.5, hi + 1.5], color=MUTED, lw=1.2, zorder=3)
    for yy in (lo - 1.5, hi + 1.5):
        ax.plot([this - (x1 - x0) * 0.004, this - (x1 - x0) * 0.001], [yy, yy], color=MUTED, lw=1.2)
    ax.text(this, max(hi + 7, 64), "This work", ha="center", fontsize=13, fontweight="bold", color=INK)

    best = [("best closed", max(p[3] for p in closed), CLOSED),
            ("best open-weights", max(p[3] for p in openw), OPENW),
            ("best fully open", max(p[3] for p in apertus + apertus8), FULLY)]
    for txt, y, c in best:
        ax.axhline(y, color=c, ls=":", lw=1.3, alpha=0.6, zorder=2)
        ax.text(date(2023, 8, 12), y + 0.8, txt, color=LABEL[c], fontsize=11, fontweight="bold", va="bottom")

    ax.set_xlim(x0, x1)
    ax.set_ylim(-2, 88)
    ax.set_yticks(range(0, 71, 10))
    ax.set_ylabel("Performance on HealthBench", fontsize=15, fontweight="bold")
    ticks = sorted({p[2] for p in closed + openw + apertus + apertus8} - {date(2025, 4, 16)})
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # year strip under the month ticks
    trans = ax.get_xaxis_transform()
    ybar = -0.105
    ax.plot([x0, x1], [ybar, ybar], color=INK, lw=0.9, transform=trans, clip_on=False)
    bounds = [x0] + [date(y, 1, 1) for y in (2024, 2025, 2026)] + [x1]
    ax.plot(bounds, [ybar] * len(bounds), "o", color=INK, ms=3.5, transform=trans, clip_on=False)
    for a, b, yr in zip(bounds[:-1], bounds[1:], (2023, 2024, 2025, 2026)):
        ax.text(a + (b - a) / 2, ybar - 0.035, str(yr), transform=trans, ha="center", va="top", fontsize=12)
    ax.set_xlabel("Release date", fontsize=15, fontweight="bold", labelpad=34)

    handles = [Line2D([], [], color=c, lw=2.5) for c in (CLOSED, OPENW, FULLY)]
    ax.legend(handles, ["Closed, API access only", "Semi-Open, Open-Weights", "Fully-Open"],
              loc="upper left", fontsize=11.5, framealpha=0.9, edgecolor="none", bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1b: size vs score

def frontier(points):
    """Upper-left Pareto frontier: (size, score) points that no smaller-or-equal model beats."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out, best = [], float("-inf")
    for x, y in pts:
        if y > best:
            out.append((x, y))
            best = y
    return out


def draw_frontier(ax, pts, color, ls, lw=1.6, alpha=0.8, label=None, zorder=1, straight=False):
    """Pareto frontier of pts as a monotone cubic (PCHIP) in log size, through every Pareto-optimal
    model, without overshoot; straight: straight segments between them. Returns the frontier points."""
    front = frontier(pts)
    if len(front) > 1 and straight:
        fx, fy = zip(*front)
        ax.plot(fx, fy, ls, color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)
    elif len(front) > 1:
        fx, fy = zip(*front)
        lx = np.log(fx)
        grid = np.linspace(lx[0], lx[-1], 200)
        ax.plot(np.exp(grid), PchipInterpolator(lx, fy)(grid), ls, color=color, lw=lw, alpha=alpha,
                zorder=zorder, label=label)
    return front


def plotted_models(scores):
    """The models of figures 1b-1d: MODELS plus the EXTRA models that have a score. With any of them,
    the Meditron-3 models are left out. Returns (models, extra)."""
    models = {r: m for r, m in {**MODELS, **EXTRA}.items() if r in scores}
    extra = any(m[3] == "gen" for m in models.values())
    if extra:
        models = {r: m for r, m in models.items() if not r.startswith("ref_meditron3")}
    return models, extra


def size_axis(ax, models, xlim):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(sorted({m[1] for m in models.values()}))
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}B"))


def gptoss_name(run, name):
    return name + "\n(MoE, 5.1B active)" if run == "ref_gptoss_120b" else name


# label offsets in points for figure 1b, tuned by eye
OFF_B = {
    "apertus8b_instruct": (0, -12, "center"),
    "meditronFO_gptoss_8b": (0, 12, "center"),
    "base_euro9b_instruct": (6, -6, "left"),
    "meditronFO_gptoss_euro9b": (6, -3, "left"),
    "base_euro22b_instruct": (-6, -9, "right"),
    "meditronFO_gptoss_euro22b": (-6, 3, "right"),
    "base_gemma3_27b_it": (-7, -4, "right"),
    "meditronFO_gptoss_gemma3_27b": (-7, 5, "right"),
    "base_medgemma27b": (7, -9, "left"),
    "base_olmo32b_sft": (6, -4, "left"),
    "meditronFO_gptoss_olmo32b": (6, -8, "left"),
    "apertus70b_instruct": (-7, -4, "right"),
    "meditronFO_gptoss_70b_16n": (-7, 2, "right"),
    "ref_meditron3_70b": (-7, -4, "right"),
    "ref_aloe_beta_70b": (-7, -4, "right"),
}
# with the EXTRA models, as in figure 1d: the Llama-3.1 families sit left of Apertus at 8B and 70B
OFF_B_EXTRA = {
    "base_gemma3_4b_it": (7, 3, "left"), "ref_medgemma4b": (7, 0, "left"),
    "base_llama31_8b_instruct": (-7, -4, "right"), "ref_aloe_beta_8b": (-7, -9, "right"),
    "meditronFO_gptoss_8b": (-7, -7, "right"), "base_gemma3_27b_it": (-9, -13, "right"), "apertus8b_instruct": (7, -4, "left"),
    "base_olmo32b_sft": (-7, -4, "right"),
    "apertus70b_instruct": (7, -4, "left"), "meditronFO_gptoss_70b_16n": (7, 4, "left"),
    "ref_aloe_beta_70b": (-7, -4, "right"), "base_llama31_70b_instruct": (7, -4, "left"),
    "ref_gptoss_120b": (0, -17, "center"),
}


def fig_size(scores, ci, out):
    models, extra = plotted_models(scores)
    nudge = {"base_medgemma27b": 1.05, **(NUDGE_D_EXTRA if extra else {})}  # MedGemma: level with Gemma-3-27B-IT
    off = {**OFF_B, **OFF_B_EXTRA} if extra else OFF_B
    xof = lambda r: models[r][1] * nudge.get(r, 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.6 if extra else 5.4))
    xmin, xmax = (3.4, 160) if extra else (6.5, 95)

    for ft, base in FINETUNE_OF.items():
        x = MODELS[ft][1]
        ax.annotate("", xy=(x, scores[ft] - 0.9), xytext=(x, scores[base] + 0.9),
                    arrowprops=dict(arrowstyle="-|>", color=MODELS[ft][2], lw=1.3, alpha=0.55,
                                    shrinkA=0, shrinkB=0, mutation_scale=10), zorder=2)

    # frontiers with and without our finetunes; "before" = the models that existed before MeditronFO
    pts = lambda keep: [(models[r][1], scores[r]) for r in models if keep(models[r])]
    for keep, c, ls, lw, lab in [
            (lambda m: m[3] != "fo", MUTED, ":", 1.4, "Pareto frontier, open models before MeditronFO"),
            (lambda m: m[3] != "fo" and m[2] == FULLY, FULLY, ":", 1.4,
             "Pareto frontier, fully open before MeditronFO"),
            (lambda m: True, MUTED, "--", 1.6, "Pareto frontier, open models with MeditronFO"),
            (lambda m: m[2] == FULLY, FULLY, "-", 1.6, "Pareto frontier, fully open with MeditronFO")]:
        draw_frontier(ax, pts(keep), c, ls, lw=lw, label=lab)

    marker = {"base": "o", "gen": "o", "fo": "o", "med": "s"}
    for run, (name, _size, color, role) in models.items():
        size, y = xof(run), scores[run]
        face = color if role == "fo" else "white"
        errbar(ax, size, run, scores, ci, color, alpha=0.8)
        ax.scatter(size, y, s=62 if role == "fo" else 50, marker="D" if run == "ref_gptoss_120b" else marker[role],
                   facecolor=face, edgecolor=color, linewidth=1.8, zorder=5)
        dx, dy, ha = off.get(run, OFF_D_DEFAULT)
        ax.annotate(gptoss_name(run, name), (size, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=7.5 if extra else 8, color=INK,
                    fontweight="bold" if role == "fo" else "normal", zorder=6)

    gpt5 = 62.6
    ax.axhline(gpt5, color=CLOSED, ls=":", lw=1.3, zorder=1)
    ax.text(xmin * 1.03, gpt5 + 0.6, "GPT-5 thinking, 62.6 (OpenAI-reported, OpenAI grader)",
            color=LABEL[CLOSED], fontsize=8.5, va="bottom")

    size_axis(ax, models, (xmin, xmax))
    ax.set_ylim(*((15, 66) if extra else (14, 67)))
    ax.set_xlabel("Parameters (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("HealthBench full (%)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    h = [Line2D([], [], marker="o", ls="", mfc=FULLY, mec=FULLY, ms=7),
         Line2D([], [], marker="o", ls="", mfc="white", mec=FULLY, mew=1.8, ms=7),
         Line2D([], [], marker="o", ls="", mfc=OPENW, mec=OPENW, ms=7),
         Line2D([], [], marker="o", ls="", mfc="white", mec=OPENW, mew=1.8, ms=7),
         Line2D([], [], marker="s", ls="", mfc="white", mec=OPENW, mew=1.8, ms=7)]
    labs = ["MeditronFO finetune, fully open", "Base model, fully open",
            "MeditronFO finetune, open weights",
            "General model, open weights" if extra else "Base model, open weights", "Earlier medical model"]
    if extra:   # both legends below the axes: the plane has no free corner
        fh, fl = ax.get_legend_handles_labels()
        ax.legend(h + fh, labs + fl, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8,
                  framealpha=0.95, edgecolor="none")
    else:
        leg1 = ax.legend(h, labs, loc="lower left", bbox_to_anchor=(0.26, 0.01), fontsize=8,
                         framealpha=0.95, edgecolor="none")
        ax.add_artist(leg1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.925), fontsize=7.5, framealpha=0.95,
                  edgecolor="none")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1c: nebulas

SLATE = "#68AFE7"   # fully-open bases, before finetuning: brand light blue, so base -> finetune reads light -> dark


def nebula(ax, pts, color, xlim, ylim, pad=0.065, alpha=0.30):
    """Soft cloud around a group of models: its convex hull, padded by `pad` (fraction of the axes)
    and faded out at the edge, plus a glow on each model. Drawn in axes-fraction space so the padding
    looks the same along the log size axis and the score axis."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.path import Path as MPath
    from scipy.spatial import ConvexHull
    lx0, lx1 = np.log10(xlim)
    aspect = 0.62                                   # axes height / width, keeps the padding round
    to_u = lambda x: (np.log10(x) - lx0) / (lx1 - lx0)
    to_v = lambda y: (np.asarray(y) - ylim[0]) / (ylim[1] - ylim[0]) * aspect
    P = np.column_stack([to_u(np.array([p[0] for p in pts])), to_v([p[1] for p in pts])])
    hull = P[ConvexHull(P).vertices]
    u = np.linspace(0, 1, 500)
    v = np.linspace(0, aspect, 320)
    U, V = np.meshgrid(u, v)
    G = np.column_stack([U.ravel(), V.ravel()])
    # distance to the hull: 0 inside, else distance to the nearest edge
    d = np.full(len(G), np.inf)
    for a_, b_ in zip(hull, np.roll(hull, -1, axis=0)):
        ab = b_ - a_
        t = np.clip(((G - a_) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(G - (a_ + t[:, None] * ab), axis=1))
    d[MPath(hull).contains_points(G)] = 0
    d = d.reshape(U.shape)
    shell = np.clip(1 - d / pad, 0, 1) ** 1.6
    glow = np.max([np.exp(-((U - pu) ** 2 + (V - pv) ** 2) / (2 * (pad * 0.55) ** 2)) for pu, pv in P], axis=0)
    field = np.clip(0.55 * shell + 0.6 * glow, 0, 1)
    rgb = to_rgb(color)
    cmap = LinearSegmentedColormap.from_list("neb", [(*rgb, 0), (*rgb, alpha)])
    X = 10 ** (lx0 + U * (lx1 - lx0))
    Y = ylim[0] + V / aspect * (ylim[1] - ylim[0])
    # an image in axes coordinates (the grid is already in axes-fraction space): smooth, no mesh seams
    ax.imshow(field, cmap=cmap, vmin=0, vmax=1, origin="lower", extent=(0, 1, 0, 1), transform=ax.transAxes,
              aspect="auto", interpolation="bilinear", zorder=1)
    ax.contour(X, Y, field, levels=[0.12], colors=[color], linewidths=0.8, alpha=0.4, zorder=1.5)


# label offsets for figure 1c
OFF_C = dict(OFF_B, base_euro22b_instruct=(-6, -8, "right"), base_olmo32b_sft=(6, -5, "left"))


def fig_nebula(scores, ci, out):
    """With the EXTRA models, as in figure 1d: the medical and general open-weight models (Gemma-3-27B-IT
    included) form one open-weight region, medical models drawn as squares."""
    models, extra = plotted_models(scores)
    nudge = {"base_medgemma27b": 1.05, **(NUDGE_D_EXTRA if extra else {})}
    off = {**OFF_C, **OFF_B_EXTRA} if extra else OFF_C
    xof = lambda r: models[r][1] * nudge.get(r, 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    xlim, ylim = ((3.4, 160), (15, 66)) if extra else ((6.5, 95), (12, 67))
    fo_fully = [r for r, m in models.items() if m[3] == "fo" and m[2] == FULLY]
    base_fully = [r for r, m in models.items() if m[3] == "base" and m[2] == FULLY]
    if extra:
        openw = [r for r, m in models.items() if m[3] in ("med", "gen") or r == "base_gemma3_27b_it"]
        openw_label = "Open-weight models (squares: medical)"
    else:
        openw = [r for r, m in models.items() if m[3] == "med"]
        openw_label = "Earlier open-weight medical models"
    groups = [(base_fully, SLATE, "Fully-open bases, before finetuning"),
              (openw, OPENW, openw_label),
              (fo_fully, FULLY, "MeditronFO, fully open (this work)")]
    for runs, c, _ in groups:
        nebula(ax, [(xof(r), scores[r]) for r in runs], c, xlim, ylim)

    for ft, base in FINETUNE_OF.items():
        x = models[ft][1]
        ax.annotate("", xy=(x, scores[ft] - 0.9), xytext=(x, scores[base] + 0.9),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, alpha=0.5,
                                    shrinkA=0, shrinkB=0, mutation_scale=9), zorder=2)

    colour = {**{r: SLATE for r in base_fully}, **{r: FULLY for r in fo_fully}, **{r: OPENW for r in openw},
              "base_gemma3_27b_it": OPENW, "meditronFO_gptoss_gemma3_27b": OPENW}
    for run, (name, _size, _c, role) in models.items():
        c = colour[run]
        size, y = xof(run), scores[run]
        errbar(ax, size, run, scores, ci, c, alpha=0.85)
        ax.scatter(size, y, s=58 if role == "fo" else 46,
                   marker="s" if role == "med" else "D" if run == "ref_gptoss_120b" else "o",
                   facecolor=c if role == "fo" else "white", edgecolor=c, linewidth=1.7, zorder=5)
        dx, dy, ha = off.get(run, OFF_D_DEFAULT)
        ax.annotate(gptoss_name(run, name), (size, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=7.5, color=INK, fontweight="bold" if role == "fo" else "normal",
                    zorder=6)

    gpt5 = 62.6
    ax.axhline(gpt5, color=CLOSED, ls=":", lw=1.3, zorder=1)
    ax.text(xlim[0] * 1.03, gpt5 + 0.6, "GPT-5 thinking, 62.6 (OpenAI-reported, OpenAI grader)",
            color=LABEL[CLOSED], fontsize=8.5, va="bottom")

    size_axis(ax, models, xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Parameters (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("HealthBench full (%)", fontsize=12, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    from matplotlib.patches import Patch
    h = [Patch(facecolor=c, alpha=0.35, edgecolor=c) for _, c, _ in groups[::-1]]
    h.append(Line2D([], [], marker="o", ls="", mfc=OPENW, mec=OPENW, ms=7))
    labs = [g[2] for g in groups[::-1]]
    if extra:
        ax.legend(h, labs + ["Gemma-3-27B-MeditronFO (open weights, this work)"], loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, framealpha=0.95, edgecolor="none")
    else:
        ax.legend(h, labs + ["Gemma-3-27B and its finetune (open weights)"], loc="upper left",
                  bbox_to_anchor=(0.0, 0.93), fontsize=8, framealpha=0.95, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1d: ellipses around frontiers

def ellipse(ax, pts, color, xlim, ylim, pad=0.045, alpha=0.13):
    """Oblong ellipse around pts, drawn in display-isotropic axes space so it stays an ellipse on the
    log size axis: long axis along the group's principal direction, padded by `pad` (axes widths)."""
    from matplotlib.patches import Polygon
    bb = ax.get_window_extent()
    k = bb.height / bb.width
    lx0, lx1 = np.log10(xlim)
    P = np.column_stack([(np.log10([p[0] for p in pts]) - lx0) / (lx1 - lx0),
                         (np.array([p[1] for p in pts]) - ylim[0]) / (ylim[1] - ylim[0]) * k])
    c = P.mean(axis=0)
    if len(P) > 1:
        _, _, vt = np.linalg.svd(P - c)
        e1 = vt[0]
    else:
        e1 = np.array([1.0, 0.0])
    e2 = np.array([-e1[1], e1[0]])
    a = np.abs((P - c) @ e1).max() + pad
    if len(P) == 1:                                   # a one-model group: a small circle
        a = b = pad * 0.6
    else:
        b = np.abs((P - c) @ e2).max() * 1.15 + pad
    t = np.linspace(0, 2 * np.pi, 200)
    E = c + np.outer(a * np.cos(t), e1) + np.outer(b * np.sin(t), e2)
    E[:, 1] /= k
    ax.add_patch(Polygon(E, closed=True, transform=ax.transAxes, facecolor=color, alpha=alpha,
                         edgecolor="none", zorder=1))
    ax.add_patch(Polygon(E, closed=True, transform=ax.transAxes, facecolor="none", edgecolor=color,
                         lw=1.4, alpha=0.85, zorder=1.2))



# label offsets for figure 1d, tuned by eye; models not listed get OFF_D_DEFAULT
OFF_D = dict(OFF_C, base_euro22b_instruct=(7, -7, "left"), base_medgemma27b=(9, -4, "left"),
             meditronFO_gptoss_gemma3_27b=(-8, 5, "right"), base_gemma3_27b_it=(-8, -5, "right"))
OFF_D_DEFAULT = (-7, -4, "right")
# with the EXTRA models: the 4B pair is labelled above and below (the axis clips the left), the Llama-3.1
# families sit left of Apertus at 8B and 70B, so their labels go left and Apertus-70B's go right
OFF_D_EXTRA = {
    "base_gemma3_4b_it": (0, 10, "center"), "ref_medgemma4b": (7, 0, "left"),
    "ref_aloe_beta_8b": (-7, -9, "right"), "base_llama31_70b_instruct": (7, -4, "left"),
    "meditronFO_gptoss_8b": (-7, 5, "right"), "apertus8b_instruct": (7, -4, "left"),
    "base_olmo32b_sft": (-7, -4, "right"),
    "apertus70b_instruct": (7, -4, "left"), "meditronFO_gptoss_70b_16n": (7, 4, "left"),
    "ref_gptoss_120b": (0, -17, "center"),
}
# size-axis nudges for models that share a size, as factors
NUDGE_D = {"base_medgemma27b": 1.06, "meditronFO_gptoss_gemma3_27b": 0.94, "base_gemma3_27b_it": 0.94}
NUDGE_D_EXTRA = {r: 0.88 for r in ("base_llama31_8b_instruct", "ref_aloe_beta_8b", "ref_meditron3_8b",
                                   "base_llama31_70b_instruct", "ref_aloe_beta_70b", "ref_meditron3_70b")}


# label offsets for figures 1d-1f, where the scores move the models into one another
OFF_E = {"base_gemma3_4b_it": (0, -12, "center"), "ref_medgemma4b": (0, -12, "center"),
         "ref_aloe_beta_8b": (-7, 6, "right"), "base_euro22b_instruct": (0, -12, "center"),
         "meditronFO_gptoss_8b": (7, -8, "left")}


OFF_PRO = {"base_gemma3_4b_it": (0, -12, "center"), "meditronFO_gptoss_8b": (-7, 8, "right"),
           "ref_aloe_beta_8b": (-7, 5, "right"), "base_llama31_8b_instruct": (-7, -7, "right")}


OFF_AVG = {"apertus8b_instruct": (-7, -4, "right"), "meditronFO_gptoss_8b": (7, 7, "left"),
           "ref_medgemma4b": (0, 10, "center"), "ref_aloe_beta_8b": (0, 10, "center")}


GLOW = ((15, 0.08), (8, 0.14))   # (width, alpha) of the halo strokes, widest first


def glow(ax, front, color):
    """A soft halo along the frontier: wide translucent strokes under the frontier line."""
    if len(front) < 2:
        return
    fx, fy = zip(*front)
    for lw, a in GLOW:
        ax.plot(fx, fy, "-", color=color, lw=lw, alpha=a, solid_capstyle="round", solid_joinstyle="round",
                zorder=1.5)


def fade(ax, front, color, xlim, ylim, depth=9.0, alpha=0.22):
    """A tint under the frontier that fades out `depth` score points below it, as an axes-space image."""
    if len(front) < 2:
        return
    from matplotlib.colors import to_rgb
    fx, fy = zip(*front)
    nx, ny = 800, 400
    lx = np.linspace(*np.log10(xlim), nx)
    ys = np.linspace(*ylim, ny)[:, None]
    top = np.interp(lx, np.log10(fx), fy, left=np.nan, right=np.nan)[None, :]
    lo, hi = np.log10(fx[0]), np.log10(fx[-1])   # taper both ends over a tenth of the frontier's span
    taper = np.clip(np.minimum(lx - lo, hi - lx) / (0.1 * (hi - lo)), 0, 1)[None, :]
    with np.errstate(invalid="ignore"):
        a = np.clip(1 - (top - ys) / depth, 0, 1) ** 1.6 * alpha * taper
        a[(ys > top) | np.isnan(top)] = 0
    img = np.zeros((ny, nx, 4))
    img[..., :3] = to_rgb(color)
    img[..., 3] = np.nan_to_num(a)
    ax.imshow(img, extent=(0, 1, 0, 1), transform=ax.transAxes, origin="lower", aspect="auto",
              interpolation="bilinear", zorder=1)


def fig_ellipses(scores, ci, out, ylabel="HealthBench full (%)", ylim=None, gpt5=62.6, offsets=None,
                 gpt5_right=False, compact=False, ellipses=True, shine=None, drop=()):
    """Like figure 1c, but each group's ellipse circles only its Pareto frontier (the models no
    smaller model of the group beats), joined by its frontier curve; every model is drawn. The models
    of EXTRA that have a score join it: then the earlier medical and the general open-weight models
    (Gemma-3-27B-IT included) form one open-weight group with one frontier, medical models drawn as
    squares and linked to the general model of their family, and the Meditron-3 models are left out.
    compact: the paper's version, shorter, with only the three groups in a legend inside the axes.
    ellipses=False: no ellipses, the frontiers drawn as straight segments.
    shine ("glow" / "fade", with ellipses=False): a soft halo along each frontier, or a tint under it.
    drop: runs left out of the figure (and of their group's frontier)."""
    models, extra = plotted_models(scores)
    models = {r: m for r, m in models.items() if r not in drop}
    nudge = {**NUDGE_D, **NUDGE_D_EXTRA} if extra else NUDGE_D
    off = {**OFF_D, **(OFF_D_EXTRA if extra else {}), **(offsets or {})}   # offsets: per figure, by eye
    fig, ax = plt.subplots(figsize=(8.4, 4.4 if compact else 5.6))
    xlim = ((3.4, 160) if max(m[1] for m in models.values()) > 100 else (3.4, 118)) if extra else (6.5, 95)
    if ylim is None:   # the data's range, the GPT-5 line included
        lo = min(ci[r][0] for r in models) - 3
        hi = max([ci[r][1] for r in models] + ([gpt5 + 2] if gpt5 else [])) + 2
        ylim = (5 * np.floor(lo / 5), 5 * np.ceil(hi / 5))
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(sorted({m[1] for m in models.values()}))
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}B"))
    ax.set_xlabel("Parameters (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()   # the ellipses need the final axes shape

    openw = ([r for r, m in models.items() if m[3] in ("med", "gen") or r == "base_gemma3_27b_it"], OPENW,
             "Open-weight models (squares: medical)")
    if not extra:
        openw = ([r for r, m in models.items() if m[3] == "med"], OPENW, "Earlier open-weight medical models")
    groups = [([r for r, m in models.items() if m[3] == "base" and m[2] == FULLY], SLATE,
               "Fully-open bases, before finetuning"),
              openw,
              ([r for r, m in models.items() if m[3] == "fo" and m[2] == FULLY], FULLY,
               "MeditronFO, fully open (this work)")]
    xof = lambda r: models[r][1] * nudge.get(r, 1)
    shown = {}
    for runs, c, _ in groups:
        front = draw_frontier(ax, [(xof(r), scores[r]) for r in runs], c, "-", lw=1.8, alpha=0.9,
                              zorder=3, straight=not ellipses)
        keep = [r for r in runs if (xof(r), scores[r]) in front]
        if ellipses:
            ellipse(ax, [(xof(r), scores[r]) for r in keep], c, xlim, ylim)
        elif shine == "glow":
            glow(ax, front, c)
        elif shine == "fade":
            fade(ax, front, c, xlim, ylim)
        shown.update({r: c for r in keep})
    for runs, c, _ in groups:                        # every model is drawn; only the frontier is circled
        shown.update({r: c for r in runs if r not in shown})
    for r in ("base_gemma3_27b_it", "meditronFO_gptoss_gemma3_27b"):   # without EXTRA / always: in no group
        if r in models:
            shown.setdefault(r, OPENW)

    for ft, base in FINETUNE_OF.items():
        if ft not in models or base not in models:
            continue
        x = xof(ft)
        ax.annotate("", xy=(x, scores[ft] - 0.9), xytext=(xof(base), scores[base] + 0.9),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, alpha=0.5,
                                    shrinkA=0, shrinkB=0, mutation_scale=9), zorder=2)
    if extra:
        for med, g in SAME_FAMILY.items():
            if med in models and g in models:
                ax.plot([xof(g), xof(med)], [scores[g], scores[med]], ls=(0, (2, 2)), color=BAND_TEXT[OPENW],
                        lw=0.9, alpha=0.55, zorder=2)
    for run, c in shown.items():
        name, _size, _c, role = models[run]
        size, y = xof(run), scores[run]
        errbar(ax, size, run, scores, ci, c, alpha=0.85)
        ax.scatter(size, y, s=58 if role == "fo" else 46,
                   marker={"med": "s"}.get(role, "D" if run == "ref_gptoss_120b" else "o"),
                   facecolor=c if role == "fo" else "white", edgecolor=c, linewidth=1.7, zorder=5)
        dx, dy, ha = off.get(run, OFF_D_DEFAULT)
        if run == "ref_gptoss_120b":
            name += "\n(MoE, 5.1B active)"
        ax.annotate(name, (size, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7.5 if extra else 8, color=INK, fontweight="bold" if role == "fo" else "normal",
                    zorder=6)

    if gpt5:
        ax.axhline(gpt5, color=CLOSED, ls=":", lw=1.3, zorder=1)
        # gpt5_right: the label at the right end, under the line, where the left end has models
        ax.text(xlim[1] * 0.97 if gpt5_right else xlim[0] * 1.03, gpt5 + (-0.6 if gpt5_right else 0.6),
                f"GPT-5 thinking, {gpt5} (OpenAI-reported, OpenAI grader)", color=LABEL[CLOSED], fontsize=8.5,
                va="top" if gpt5_right else "bottom", ha="right" if gpt5_right else "left")

    from matplotlib.patches import Patch
    h = [Patch(facecolor=to_alpha(c, 0.13), edgecolor=c, lw=1.4) if ellipses
         else tuple(Line2D([], [], color=c, lw=lw * 0.6, alpha=a, solid_capstyle="round") for lw, a in GLOW)
         + (Line2D([], [], color=c, lw=1.8),) if shine == "glow"
         else Line2D([], [], color=c, lw=1.8) for _, c, _ in groups[::-1]]
    h.append(Line2D([], [], marker="o", ls="", mfc=OPENW, mec=OPENW, ms=7))
    labs = [g[2] for g in groups[::-1]]
    if extra:
        labs.append("Gemma-3-27B-MeditronFO (open weights, this work)")
        h.append(Line2D([], [], ls=(0, (2, 2)), color=BAND_TEXT[OPENW], lw=0.9, alpha=0.7))
        labs.append("Medical model to general model of its family")
    else:
        labs.append("Gemma-3-27B and its finetune (open weights)")
    if compact:   # the three groups only, in the empty lower right of the plane
        ax.legend(h[:3], labs[:3], loc="lower right", fontsize=8, framealpha=0.95, edgecolor="none")
    elif extra:   # below the axes: the plane has no free corner
        ax.legend(h, labs, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8,
                  framealpha=0.95, edgecolor="none")
    else:
        ax.legend(h, labs, loc="lower left", bbox_to_anchor=(0.26, 0.01), fontsize=8, framealpha=0.95,
                  edgecolor="none")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# label offsets for the paper's Figure 1, tuned by eye
OFF_PAPER = {
    "base_gemma3_4b_it": (-4, 12, "left"), "ref_medgemma4b": (0, -13, "center"),
    "base_llama31_8b_instruct": (-9, -3, "right"), "ref_aloe_beta_8b": (3, -13, "center"),
    "meditronFO_gptoss_8b": (-9, 4, "right"), "apertus8b_instruct": (9, -4, "left"),
    "base_euro9b_instruct": (9, -5, "left"), "meditronFO_gptoss_euro9b": (10, -1, "left"),
    "meditronFO_gptoss_euro22b": (0, 13, "center"), "base_euro22b_instruct": (9, -7, "left"),
    "meditronFO_gptoss_gemma3_27b": (-8, 10, "right"), "base_gemma3_27b_it": (-6, -11, "right"),
    "base_medgemma27b": (10, -2, "left"),
    "meditronFO_gptoss_olmo32b": (10, -3, "left"), "base_olmo32b_sft": (-9, -3, "right"),
    "meditronFO_gptoss_70b_16n": (10, 5, "left"), "apertus70b_instruct": (9, -2, "left"),
    "ref_aloe_beta_70b": (-9, -1, "right"), "base_llama31_70b_instruct": (6, -12, "right"),
    "ref_gptoss_120b": (0, -18, "center"),
}


def fig_paper(scores, out, gpt5=None):   # gpt5=62.6 draws the OpenAI-graded GPT-5 line (not comparable, App. H.1)
    """The paper's Figure 1: filled markers coloured by openness, circles before and stars after
    finetuning (MeditronFO and the earlier medical models), grey arrows from each base to its
    MeditronFO finetune, and the Pareto frontiers of the open-weight and the fully-open models as
    straight segments. No intervals: they are in the appendix."""
    models, _ = plotted_models(scores)
    nudge = {**NUDGE_D_EXTRA, "base_medgemma27b": 1.08}
    xof = lambda r: models[r][1] * nudge.get(r, 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    xlim, ylim = (3.1, 200), (15, 65)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(sorted({m[1] for m in models.values()}))
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}B"))
    ax.tick_params(labelsize=10)
    ax.set_xlabel("Parameters (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("HealthBench full (%)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    for c in (OPENW, FULLY):
        draw_frontier(ax, [(xof(r), scores[r]) for r, m in models.items() if m[2] == c], c, "-", lw=2,
                      alpha=0.9, zorder=3, straight=True)
    for ft, base in FINETUNE_OF.items():
        if scores[ft] - scores[base] > 2:   # Gemma-3-27B gains too little for an arrow
            ax.annotate("", xy=(xof(ft), scores[ft] - 1.1), xytext=(xof(base), scores[base] + 1.1),
                        arrowprops=dict(arrowstyle="-|>", color="#9a9a9a", lw=1.4, shrinkA=0, shrinkB=0,
                                        mutation_scale=11), zorder=2)
    for run, (name, _size, c, role) in models.items():
        star = role in ("fo", "med")
        ax.scatter(xof(run), scores[run], s=200 if star else 75, marker="*" if star else "o", color=c,
                   edgecolor="white", linewidth=0.6, zorder=5)
        dx, dy, ha = OFF_PAPER.get(run, OFF_D_DEFAULT)
        ax.annotate(gptoss_name(run, name), (xof(run), scores[run]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=INK,
                    fontweight="bold" if role == "fo" else "normal", zorder=6)

    if gpt5 is not None:
        ax.axhline(gpt5, color=CLOSED, lw=1.5, zorder=1)
        ax.text(xlim[0] * 1.03, gpt5 + 0.5, f"GPT-5 thinking, {gpt5} (OpenAI-reported, OpenAI grader)",
                color=CLOSED, fontsize=9.5, va="bottom")

    dot = lambda c, m="o", ms=8: Line2D([], [], marker=m, ls="", mfc=c, mec=c, ms=ms)
    h = [dot(FULLY), dot(OPENW), dot(INK), dot(INK, "*", 11), Line2D([], [], color=FULLY, lw=2),
         Line2D([], [], marker="$\\uparrow$", ls="", color="#9a9a9a", ms=10)]
    labs = ["Fully open", "Open weight", "Before fine-tuning", "After fine-tuning",
            "Best-performing frontier", "Fine-tuning gain"]
    ax.legend(h, labs, loc="lower right", fontsize=8, framealpha=0.95, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_alpha(color, a):
    from matplotlib.colors import to_rgb
    return (*to_rgb(color), a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="full_gemma4_31b", help="directory under healthbench/results")
    a = ap.parse_args()
    scores, ci = load_scores(REPO / "healthbench" / "results" / a.results)
    fig_paper(scores, REPO / "papers" / "sept_25th_iclr_paper" / "figures" / "healthbench_size_paper.pdf")
    for run, (name, size, *_rest) in sorted(MODELS.items(), key=lambda kv: -scores[kv[0]]):
        print(f"{name:28s} {size:>3}B  {scores[run]:5.1f}  [{ci[run][0]:.1f}, {ci[run][1]:.1f}]")


if __name__ == "__main__":
    main()
