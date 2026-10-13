#!/usr/bin/env python3
# tech_reports/figures/healthbench_breakdown_figures.py

"""
Figures and table bodies for tech_reports/healthbench_breakdown.tex, from
healthbench_breakdown.json (healthbench_breakdown_ci.py, run it first). Table bodies go to
tech_reports/figures/hb_tables/*.tex and are \\input by the report, so no number is typed by hand.

.venv/bin/python tech_reports/figures/healthbench_breakdown_ci.py
.venv/bin/python tech_reports/figures/healthbench_breakdown_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from iclr_clean_figures import (C1, C2, C3, INK, INK2, MUTED, GRID, BASE, OUT, PAPER, save, point,
                                rows_axis)

HB = json.load(open(OUT / "healthbench_breakdown.json"))
TAB = OUT / "hb_tables"
TAB.mkdir(exist_ok=True)

NAME = {"apertus70b": "Apertus-70B-Instruct", "apertus8b": "Apertus-8B-Instruct",
        "olmo32b": "OLMo-2-32B-SFT", "euro22b": "EuroLLM-22B-Instruct",
        "euro9b": "EuroLLM-9B-Instruct", "gemma27b": "Gemma-3-27B-IT", "medgemma": "MedGemma-27B"}
SHORT = {"apertus70b": "Apertus-70B", "apertus8b": "Apertus-8B", "olmo32b": "OLMo-2-32B",
         "euro22b": "EuroLLM-22B", "euro9b": "EuroLLM-9B", "gemma27b": "Gemma-3-27B"}
BASES = list(SHORT)
ABL = {"no_curated_qa": "no Curated QA", "no_synthetic_qa": "no Synthetic QA",
       "no_guidelines_qa": "no Guidelines QA", "no_synthetic_moove": "no synthetic vignettes"}

THEMES = [("theme:emergency_referrals", "Emergency referrals"), ("theme:context_seeking", "Context seeking"),
          ("theme:global_health", "Global health"), ("theme:health_data_tasks", "Health data tasks"),
          ("theme:communication", "Communication"), ("theme:hedging", "Hedging"),
          ("theme:complex_responses", "Complex responses")]
AXES = [("axis:accuracy", "Accuracy"), ("axis:completeness", "Completeness"),
        ("axis:communication_quality", "Communication quality"),
        ("axis:context_awareness", "Context awareness"), ("axis:instruction_following", "Instruction following")]
LEVELS = [("level:example", "Example rubrics"), ("level:cluster", "Consensus rubrics")]
PRO = [("type:good_faith", "Good faith"), ("type:red_teaming", "Red teaming"),
       ("difficulty:typical", "Typical"), ("difficulty:difficult", "Difficult"),
       ("use_case:consult", "Consult"), ("use_case:research", "Research"), ("use_case:writing", "Writing")]
OVERALL = [("overall", "Overall")]

# the rows of the difference heatmaps: finetune - base, then the two reference rows
HEAT_ROWS = [(f"{b}_ft-{b}", f"{SHORT[b]}") for b in BASES] + [
    ("medgemma-gemma27b", "MedGemma-27B vs Gemma-3-27B"),
    ("apertus70b_ft_rep-apertus70b_ft", "Apertus-70B-MFO, 2nd run vs 1st"),
]
# column labels of the paper's Figure 5, wrapped to fit a flat label under a narrow cell
PAPER_COL = {"theme:emergency_referrals": "Emergency\nreferrals", "theme:context_seeking": "Context\nseeking",
             "theme:global_health": "Global\nhealth", "theme:health_data_tasks": "Health\ndata tasks",
             "theme:communication": "Commu-\nnication", "theme:complex_responses": "Complex\nresponses",
             "axis:completeness": "Complete-\nness", "axis:communication_quality": "Commun.\nquality",
             "axis:context_awareness": "Context\nawareness", "axis:instruction_following": "Instruction\nfollowing"}
NEUTRAL = "#f2f1ee"
DIV = LinearSegmentedColormap.from_list("hbdiv", [C2, NEUTRAL, C1])


def S(split):
    return HB["splits"][split]


def sig(v):
    return v[1] > 0 or v[2] < 0


# --- figures ------------------------------------------------------------------------------
def heatmap(split, groups, name, rows=HEAT_ROWS, vmax=20, title=None, ref_rows=2, width=7.0, fs=6.8,
            out=OUT, row_h=0.05, title_fs=8, crop=False, wspace=0.04, pad=1.5, show_n=True, rot=40):
    """Paired difference (points) per row and slice; a cell whose 95% interval holds 0 is drawn
    pale with grey text. pad: inches of height beyond the rows; rot=0 sets the column labels flat."""
    D = S(split)["diffs"]
    n = S(split)["n"]
    widths = [len(g[1]) for g in groups]
    fig, axes = plt.subplots(1, len(groups), figsize=(width, row_h * fs * len(rows) + pad),
                             gridspec_kw={"width_ratios": widths, "wspace": wspace})
    axes = np.atleast_1d(axes)
    for gi, (ax, (gname, cols)) in enumerate(zip(axes, groups)):
        img = np.ones((len(rows), len(cols), 3))
        for i, (pk, _) in enumerate(rows):
            for j, (m, _) in enumerate(cols):
                v = D[pk][m]
                c = np.array(to_rgb(DIV(0.5 + 0.5 * np.clip(v[0] / vmax, -1, 1))))
                if not sig(v):
                    c = 0.3 * c + 0.7 * np.array(to_rgb(NEUTRAL))
                img[i, j] = c
                strong = sig(v) and abs(v[0]) > 0.55 * vmax
                ax.text(j, i, (f"{v[0]:+.1f}" if round(v[0], 1) else "0.0").replace("-", "−"), ha="center", va="center", fontsize=fs,
                        color="white" if strong else (INK if sig(v) else MUTED),
                        fontweight="bold" if sig(v) else "normal")
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(cols)), [f"{c[1]}\n(n={n[c[0]]})" if show_n else c[1] for c in cols],
                      rotation=rot, ha="right" if rot else "center", rotation_mode="anchor", fontsize=fs)
        ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(len(rows) - ref_rows - 0.5, color=INK2, lw=0.8)
        if len(cols) > 1:                      # a one-column group is named by its tick label
            ax.set_title(gname, loc="left", color=INK, fontsize=title_fs)
        if gi == 0:
            ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=fs + 0.7)
        else:
            ax.set_yticks([])
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8.5, color=INK)
    save(fig, name, out)
    if crop:
        crop_to_ink(out / f"{name}.pdf")


def crop_to_ink(path, margin=2):
    """Trim a PDF page to its drawn content plus margin points: the tight bounding box of rotated tick
    labels (rotation_mode="anchor") leaves blank space under them."""
    import pymupdf
    doc = pymupdf.open(path)
    page = doc[0]
    pm = page.get_pixmap(dpi=144)
    ink = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3].min(axis=2) < 250
    ys, xs = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    k = 72 / 144
    r = pymupdf.Rect(xs[0] * k - margin, ys[0] * k - margin, (xs[-1] + 1) * k + margin, (ys[-1] + 1) * k + margin)
    page.set_cropbox(r & page.rect)
    doc.save(str(path) + ".tmp", garbage=3, deflate=True)
    doc.close()
    Path(str(path) + ".tmp").replace(path)


def axes_panels():
    """Scores per rubric axis and level, base against finetune, one panel per base (full)."""
    sc = S("full")["scores"]
    rows = OVERALL + AXES + LEVELS
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for ax, b in zip(axs.flat, BASES):
        for i, (m, _) in enumerate(rows):
            point(ax, sc[b][m], i - 0.15, BASE, hollow=True, label="base" if i == 0 else None)
            point(ax, sc[f"{b}_ft"][m], i + 0.15 if b != "gemma27b" else i, C1,
                  label="+ MeditronFO" if i == 0 else None)
            if b == "gemma27b":
                point(ax, sc["medgemma"][m], i + 0.22, C2, marker="D", ms=4,
                      label="MedGemma-27B" if i == 0 else None)
        rows_axis(ax, [r[1] for r in rows])
        ax.axhline(0.5, color=GRID, lw=0.8)
        ax.axhline(5.5, color=GRID, lw=0.8)
        ax.set_title(SHORT[b], loc="left", color=INK)
        ax.set_xlim(0, 90)
    fig.supxlabel("HealthBench full score (%, 95% CI)", fontsize=8, color=INK2)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[1, 2].get_legend_handles_labels()
    fig.legend(h1 + h2[2:], l1 + l2[2:], loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=3)
    fig.tight_layout()
    save(fig, "hb_axes")


def length_fig(width=7.2, name="hb_length", out=OUT, pro=True):
    """(a) mean answer length against score, base to finetune; (b, c) the finetune's gain raw and
    after the Professional length adjustment, full and professional. pro=False drops panel (c)."""
    fig, axs = plt.subplots(1, 3 if pro else 2, figsize=(width, 3.2 * width / 7.2 + 0.2),
                            gridspec_kw={"width_ratios": [1.6, 1, 1] if pro else [1.6, 1]})
    # label offsets (points) that keep the finetune labels clear of each other and the arrows
    off = {"apertus70b": (5, -2, "left"), "apertus8b": (5, -4, "left"), "euro22b": (-5, 4, "right"),
           "olmo32b": (4, -10, "left"), "euro9b": (5, -3, "left"), "gemma27b": (5, -1, "left")}
    a = axs[0]
    sc = S("full")["scores"]
    for b in BASES:
        x0, y0 = sc[b]["words"][0], sc[b]["overall"][0]
        x1, y1 = sc[f"{b}_ft"]["words"][0], sc[f"{b}_ft"]["overall"][0]
        a.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=BASE, lw=0.9,
                                                           shrinkA=4, shrinkB=4))
        a.plot(x0, y0, "o", ms=5, mfc="white", mec=BASE, mew=1.2, zorder=3)
        a.plot(x1, y1, "o", ms=5, color=C1, mec="white", mew=1, zorder=3)
        dx, dy, ha = off[b]
        a.annotate(SHORT[b], (x1, y1), xytext=(dx, dy), textcoords="offset points",
                   fontsize=plt.rcParams["font.size"] * 0.8,
                   color=INK2, ha=ha)
    a.plot(sc["medgemma"]["words"][0], sc["medgemma"]["overall"][0], "D", ms=4, color=C2, mec="white")
    a.annotate("MedGemma-27B", (sc["medgemma"]["words"][0], sc["medgemma"]["overall"][0]),
               xytext=(-2, -9), textcoords="offset points", fontsize=plt.rcParams["font.size"] * 0.8,
               color=INK2, ha="right")
    a.set_xticks([0, 250, 500, 750, 1000])
    a.set_xlabel("mean answer length (words)")
    a.set_ylabel("HealthBench full (%)")
    a.set_title("(a) Length and score, full", loc="left", color=INK)
    a.set_xlim(0, 1000)
    rows = [(f"{b}_ft-{b}", SHORT[b]) for b in BASES] + [("medgemma-gemma27b", "MedGemma vs Gemma-3")]
    panels = [(axs[1], "full", "(b) Gain, full")] + ([(axs[2], "professional", "(c) Gain, professional")] if pro else [])
    for ax, split, t in panels:
        D = S(split)["diffs"]
        for i, (pk, _) in enumerate(rows):
            point(ax, D[pk]["overall"], i - 0.15, C1, label="raw" if i == 0 else None)
            point(ax, D[pk]["score_len_adj"], i + 0.15, C2, label="length-adjusted" if i == 0 else None,
                  marker="s", ms=4)
        ax.axvline(0, color=MUTED, lw=0.8)
        rows_axis(ax, [r[1] for r in rows] if ax is axs[1] else [""] * len(rows))
        ax.set_title(t, loc="left", color=INK)
        ax.set_xlabel("difference (points, 95% CI)")
        ax.set_xlim(-12, 18)
    axs[1].legend(loc="lower center", bbox_to_anchor=(1.0 if pro else 0.5, 1.1), ncol=2)
    fig.tight_layout(w_pad=1.0)
    save(fig, name, out)


# --- tables -------------------------------------------------------------------------------
def num(x, nd=1):
    return f"{x:.{nd}f}"


def dcell(v, nd=1):
    x = f"{v[0]:+.{nd}f}" if round(v[0], nd) else f"{0:.{nd}f}"   # no "-0.0"
    return f"$\\mathbf{{{x}}}$" if sig(v) else f"${x}$"


def write(name, text):
    # this filesystem does not always truncate a file rewritten shorter: remove it first
    f = TAB / f"{name}.tex"
    f.unlink(missing_ok=True)
    f.write_text(text)


def score_table(split, cols, name, extra=None):
    """Base rows: score. Finetune rows: score and paired difference to the base (bold when the
    95% interval excludes 0). MedGemma: difference to Gemma-3-27B-IT."""
    sc, D, n = S(split)["scores"], S(split)["diffs"], S(split)["n"]
    head = " & ".join(f"\\textbf{{{c[1]}}}" for c in cols)
    lines = [f"\\begin{{tabular}}{{l{'c' * len(cols)}}}", "\\toprule", f"\\textbf{{Model}} & {head} \\\\",
             "\\textit{n} & " + " & ".join(str(n[c[0]]) for c in cols) + " \\\\", "\\midrule"]

    def ft_row(label, key, ref):
        cells = [f"{num(sc[key][c[0]][0])} {{\\scriptsize {dcell(D[f'{key}-{ref}'][c[0]])}}}" for c in cols]
        return f"{label} & " + " & ".join(cells) + " \\\\"

    for b in BASES:
        lines.append(f"{NAME[b]} & " + " & ".join(num(sc[b][c[0]][0]) for c in cols) + " \\\\")
        lines.append(ft_row("\\quad + MeditronFO", f"{b}_ft", b))
    lines.append("\\midrule")
    lines.append(ft_row("MedGemma-27B {\\scriptsize (vs Gemma-3)}", "medgemma", "gemma27b"))
    lines.append(ft_row("Apertus-70B-MFO, 2nd run {\\scriptsize (vs 1st)}", "apertus70b_ft_rep", "apertus70b_ft"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(name, "\n".join(lines) + "\n")


def penalty_table(split, name):
    sc, D = S(split)["scores"], S(split)["diffs"]
    cols = [("overall", 1), ("pos_only", 1), ("penalty_lost", 1), ("neg_any", 1), ("neg_rate", 1)]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "\\textbf{Model} & \\textbf{Score} & \\textbf{Without penalties} & \\textbf{Penalty points lost}"
             " & \\textbf{Answers with $\\geq$1 penalty} & \\textbf{Penalty criteria met} \\\\", "\\midrule"]
    for b in BASES:
        lines.append(f"{NAME[b]} & " + " & ".join(num(sc[b][c][0]) for c, _ in cols) + " \\\\")
        lines.append("\\quad + MeditronFO & " + " & ".join(
            f"{num(sc[b + '_ft'][c][0])} {{\\scriptsize {dcell(D[f'{b}_ft-{b}'][c])}}}" for c, _ in cols) + " \\\\")
    lines.append("\\midrule")
    lines.append("MedGemma-27B {\\scriptsize (vs Gemma-3)} & " + " & ".join(
        f"{num(sc['medgemma'][c][0])} {{\\scriptsize {dcell(D['medgemma-gemma27b'][c])}}}" for c, _ in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(name, "\n".join(lines) + "\n")


def length_table(name):
    lines = ["\\begin{tabular}{l" + "ccc" * 2 + "}", "\\toprule",
             "& \\multicolumn{3}{c}{\\textbf{Full}} & \\multicolumn{3}{c}{\\textbf{Hard}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "\\textbf{Model}" + " & Chars & Score & Adjusted" * 2 + " \\\\", "\\midrule"]

    def cells(key, ref):
        out = []
        for split in ("full", "hard"):
            sc, D = S(split)["scores"], S(split)["diffs"]
            for m, nd in (("chars", 0), ("overall", 1), ("score_len_adj", 1)):
                v = sc[key][m][0]
                c = f"{v:.{nd}f}"
                if ref:
                    c += f" {{\\scriptsize {dcell(D[f'{key}-{ref}'][m], nd)}}}"
                out.append(c)
        return " & ".join(out)

    for b in BASES:
        lines.append(f"{NAME[b]} & {cells(b, None)} \\\\")
        lines.append(f"\\quad + MeditronFO & {cells(b + '_ft', b)} \\\\")
    lines.append("\\midrule")
    lines.append(f"MedGemma-27B {{\\scriptsize (vs Gemma-3)}} & {cells('medgemma', 'gemma27b')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(name, "\n".join(lines) + "\n")


def breakdown_table(prefix, name, label_fn):
    """One row per physician-agreed category or consensus cluster (full): n, the Apertus-70B-MeditronFO
    score, and the paired difference of every finetune to its base."""
    sc, D, n = S("full")["scores"], S("full")["diffs"], S("full")["n"]
    order = [lab for _, lab in THEMES]
    ms = sorted((m for m in n if m.startswith(prefix)),
                key=lambda m: (order.index(label_fn(m)[0]), label_fn(m)[1]))
    pairs = [(f"{b}_ft-{b}", SHORT[b]) for b in BASES] + [("medgemma-gemma27b", "MedGemma")]
    lines = ["\\begin{tabular}{lr" + "c" * (1 + len(pairs)) + "}", "\\toprule",
             "\\textbf{Slice} & \\textit{n} & \\textbf{A-70B-MFO} & " + " & ".join(
                 f"\\textbf{{{p[1]}}}" for p in pairs) + " \\\\", "\\midrule"]
    group = None
    for m in ms:
        g, lab = label_fn(m)
        if g != group:
            if group is not None:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{{3 + len(pairs)}}}{{l}}{{\\textit{{{g}}}}} \\\\")
            group = g
        lines.append(f"\\quad {lab} & {n[m]} & {num(sc['apertus70b_ft'][m][0])} & "
                     + " & ".join(dcell(D[p][m]) for p, _ in pairs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(name, "\n".join(lines) + "\n")


THEME_OF = {t.split(":")[1]: lab for t, lab in THEMES}


def category_label(m):
    """physician_agreed_category:<cat> -> (theme, category), the theme from the cluster names."""
    cat = m.split(":", 1)[1]
    for c in S("full")["n"]:
        if c.startswith("cluster:"):
            rest = c.split(":", 1)[1]
            for t, lab in THEME_OF.items():
                if rest.startswith(t + "_" + cat + "_"):
                    return lab, cat.replace("-", " ")
    return "Other", cat


def cluster_label(m):
    rest = m.split(":", 1)[1]
    for t, lab in THEME_OF.items():
        if rest.startswith(t + "_"):
            rest = rest[len(t) + 1:]
            break
    cat, crit = rest.split("_", 1)
    return lab, f"{cat.replace('-', ' ')}: {crit.replace('_', ' ')}"


def ablation_table(split, name):
    """One tabular, two blocks (overall and themes; axes and levels), so it fits one \\resizebox."""
    sc, D = S(split)["scores"], S(split)["diffs"]
    blocks = [OVERALL + THEMES, AXES + LEVELS]
    w = max(len(b) for b in blocks)
    pad = lambda cells: cells + [""] * (w - len(cells))
    lines = [f"\\begin{{tabular}}{{l{'c' * w}}}", "\\toprule"]
    for bi, cols in enumerate(blocks):
        if bi:
            lines += ["\\midrule", "\\addlinespace"]
        lines += ["\\textbf{Apertus-8B} & " + " & ".join(pad([f"\\textbf{{{c[1]}}}" for c in cols])) + " \\\\",
                  "\\midrule",
                  "base & " + " & ".join(pad([num(sc["apertus8b"][c[0]][0]) for c in cols])) + " \\\\",
                  "MeditronFO (all four sets) & " + " & ".join(
                      pad([num(sc["apertus8b_ft"][c[0]][0]) for c in cols])) + " \\\\"]
        for k, lab in ABL.items():
            lines.append(f"\\quad {lab} & " + " & ".join(pad(
                [f"{num(sc[k][c[0]][0])} {{\\scriptsize {dcell(D[f'{k}-apertus8b_ft'][c[0]])}}}" for c in cols]))
                + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(name, "\n".join(lines) + "\n")


def main():
    theme_groups = [("Overall", OVERALL), ("Theme", THEMES), ("Rubric axis", AXES), ("Rubric level", LEVELS)]
    # the paper's copies, drawn at its text width (5.5 in) so they are not scaled down.
    # Figure 5 is kept short: no Overall column (Figures 1 and 3 have it) or rubric-level group, rows
    # named by base (the caption says finetune - base), flat two-line column labels without n
    paper_rows = HEAT_ROWS[:-2] + [("medgemma-gemma27b", "MedGemma-27B")]
    paper_groups = [(g, [(m, PAPER_COL.get(m, lab)) for m, lab in cols]) for g, cols in theme_groups[1:-1]]
    heatmap("full", paper_groups, "hb_heatmap_full_paper", rows=paper_rows, ref_rows=1, width=5.9, fs=5.0,
            out=PAPER, row_h=0.028, title_fs=6.2, crop=True, wspace=0.035, pad=0.55, show_n=False, rot=0)
    small = {k: v * 0.8 for k, v in plt.rcParams.items()
             if k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
                      "legend.fontsize")}
    with plt.rc_context(small):
        length_fig(width=5.5, name="hb_length_paper", out=PAPER, pro=False)
    length_table("length")                                                       # tab:hb_length
    breakdown_table("physician_agreed_category:", "categories", category_label)  # tab:hb_consensus
    print("figures and tables written to", OUT)

if __name__ == "__main__":
    main()
