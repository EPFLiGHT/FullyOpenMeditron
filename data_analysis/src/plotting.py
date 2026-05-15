"""
Publication-quality plotting for source<->synthetic distribution shift.

Two plot types:
  1. Paired bar chart per (pair, field): source vs synthetic proportions
     side by side, sorted by source proportion descending, with the
     divergence value + CI in the title.
  2. Heatmap: rows = pairs, cols = fields, cell = JSD (categorical) or
     normalized Wasserstein (ordinal). One summary figure for the paper.

Design
------
- No seaborn: one less dependency, matplotlib alone handles everything.
- Viridis-like palette but just 2 colors per plot (source = teal, synthetic = coral).
- 300 DPI PDF output for LaTeX. Tight layout, legible fonts (serif, 11pt).
- Bar charts show empty categories (0% proportion) so you can see full
  distribution shape, but only when there are <= 12 categories. More than that
  we filter to the union of non-zero categories to avoid a visual mess.
- Long category labels are rotated 45deg with right-alignment.
"""

import math
from typing import Dict, List, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_academic_style():
    """NeurIPS-ish style: serif body, clean axes, no unnecessary chartjunk."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
    })


# Colors — two tones that work in both light mode and print. Teal/coral
# read cleanly together and match common medical-paper palettes.
COLOR_SOURCE = "#1D9E75"      # teal
COLOR_SYNTHETIC = "#D85A30"   # coral
COLOR_NEUTRAL = "#888780"     # gray


# ---------------------------------------------------------------------------
# Paired bar chart (one field, one pair)
# ---------------------------------------------------------------------------

def _format_category_label(cat):
    """Clean up enum values for display: underscores -> spaces, sentence case."""
    s = str(cat).replace("_", " ")
    return s[0].upper() + s[1:] if s else s


def plot_paired_categorical(
    categories,
    props_source,
    props_synthetic,
    field_name,
    pair_name,
    jsd,
    jsd_lo,
    jsd_hi,
    n_source,
    n_synthetic,
    out_path,
    max_cats_to_show=12,
    min_shown_prop=0.005,
):
    """Side-by-side bar chart: source vs synthetic proportions for one field."""
    categories = list(categories)
    props_source = np.asarray(props_source)
    props_synthetic = np.asarray(props_synthetic)

    # Filter categories: keep only those with non-trivial proportion in at
    # least one arm, if the total category count is large.
    if len(categories) > max_cats_to_show:
        keep = [i for i, (a, b) in enumerate(zip(props_source, props_synthetic))
                if max(a, b) >= min_shown_prop]
        if not keep:
            keep = list(range(len(categories)))
        # Always keep at least max_cats_to_show so we don't over-filter
        if len(keep) < max_cats_to_show:
            # Pad with highest-proportion remaining cats
            remaining = [i for i in range(len(categories)) if i not in keep]
            remaining.sort(
                key=lambda i: -max(props_source[i], props_synthetic[i])
            )
            keep = keep + remaining[:max_cats_to_show - len(keep)]
        keep.sort()
        categories = [categories[i] for i in keep]
        props_source = props_source[keep]
        props_synthetic = props_synthetic[keep]

    # Sort by source proportion descending (makes the shape readable)
    order = np.argsort(-props_source)
    categories = [categories[i] for i in order]
    props_source = props_source[order]
    props_synthetic = props_synthetic[order]

    n_cats = len(categories)
    x = np.arange(n_cats)
    w = 0.4

    fig_w = max(5.0, n_cats * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))

    ax.bar(x - w / 2, props_source * 100, width=w,
           color=COLOR_SOURCE, edgecolor="black", linewidth=0.3,
           label="Source  (n = {})".format(n_source))
    ax.bar(x + w / 2, props_synthetic * 100, width=w,
           color=COLOR_SYNTHETIC, edgecolor="black", linewidth=0.3,
           label="Synthetic  (n = {})".format(n_synthetic))

    # title = "{}:  {}\nJSD = {:.3f}  [95% CI: {:.3f}, {:.3f}]".format(
    #     pair_name.upper(), _format_category_label(field_name),
    #     jsd, jsd_lo, jsd_hi,
    # )
    # ax.set_title(title)
    ax.set_ylabel("Percentage of records (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_category_label(c) for c in categories],
                       rotation=45, ha="right")
    ax.legend(frameon=True, loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ordinal comparison (difficulty): histograms
# ---------------------------------------------------------------------------

def plot_paired_ordinal(
    values_source,
    values_synthetic,
    field_name,
    pair_name,
    w1,
    w1_lo,
    w1_hi,
    mean_source,
    mean_synthetic,
    n_source,
    n_synthetic,
    out_path,
    bins=None,
):
    """Paired histogram for an ordinal field (difficulty 1-5)."""
    if bins is None:
        bins = np.arange(0.5, 6.0, 1.0)  # 1..5 integer bins

    values_source = np.asarray([v for v in values_source if v is not None])
    values_synthetic = np.asarray([v for v in values_synthetic if v is not None])

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    hs, _ = np.histogram(values_source, bins=bins, density=True)
    hs_syn, _ = np.histogram(values_synthetic, bins=bins, density=True)

    centers = (bins[:-1] + bins[1:]) / 2.0
    w = 0.4
    ax.bar(centers - w / 2, hs * 100, width=w,
           color=COLOR_SOURCE, edgecolor="black", linewidth=0.3,
           label="Source (mean={:.2f}, n={})".format(mean_source, n_source))
    ax.bar(centers + w / 2, hs_syn * 100, width=w,
           color=COLOR_SYNTHETIC, edgecolor="black", linewidth=0.3,
           label="Synthetic (mean={:.2f}, n={})".format(mean_synthetic, n_synthetic))

    # title = "{}:  {}\nWasserstein-1 = {:.3f}  [95% CI: {:.3f}, {:.3f}]".format(
    #     pair_name.upper(), _format_category_label(field_name),
    #     w1, w1_lo, w1_hi,
    # )
    # ax.set_title(title)
    ax.set_xlabel("Difficulty (1 = easy, 5 = expert)")
    ax.set_ylabel("Density (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(frameon=True, loc="best")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def plot_summary_heatmap(
    rows,              # list of pair names
    cols,              # list of field names
    matrix,            # 2-D np.ndarray, same shape as (rows, cols), NaN for missing
    out_path,
    title="Jensen-Shannon divergence: source vs synthetic",
    vmax=None,
):
    """Heatmap of divergence values across (pair, field)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if vmax is None:
        vmax = float(np.nanmax(matrix)) if np.any(~np.isnan(matrix)) else 1.0
        vmax = max(vmax, 0.1)  # don't collapse to zero

    fig, ax = plt.subplots(figsize=(max(5.0, len(cols) * 0.9),
                                    max(2.5, len(rows) * 0.7)))
    # Cool-to-warm: white -> red for increasing divergence
    im = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([_format_category_label(c) for c in cols],
                       rotation=30, ha="right")
    ax.set_yticklabels([r.upper() for r in rows])

    # Annotate cells
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt_color = "white" if v > vmax * 0.55 else "#333333"
                ax.text(j, i, "{:.2f}".format(v), ha="center", va="center",
                        color=txt_color, fontsize=9)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        color="#aaaaaa", fontsize=9)

    # ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("JSD (0 = identical, 1 = disjoint)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task-format shift table (rendered as a plot for the paper)
# ---------------------------------------------------------------------------

def plot_task_format_table(
    stats_per_pair,
    out_path,
):
    """One figure summarizing the per-pair task-format shift: MCQ rate,
    document rate, paired rate, token counts. Rendered as a small table plot.

    stats_per_pair: {pair_name: {"source": stats_dict, "synthetic": stats_dict}}
    """
    pairs = list(stats_per_pair.keys())
    metrics = [
        ("pct_paired", "% paired (A/B)", ".1f"),
        ("pct_mcq", "% MCQ", ".1f"),
        ("pct_document", "% document", ".1f"),
        ("pct_has_think_block", "% <think>", ".1f"),
        ("median_q_tokens", "Q tokens (med)", ".0f"),
        ("median_a_tokens", "A tokens (med)", ".0f"),
        ("median_full_tokens", "Full tokens (med)", ".0f"),
    ]

    col_labels = ["{}   {}".format(p.upper(), side)
                  for p in pairs for side in ("source", "synthetic")]
    row_labels = [m[1] for m in metrics]

    rows = []
    for key, _label, fmt in metrics:
        row = []
        for p in pairs:
            for side in ("source", "synthetic"):
                v = stats_per_pair[p][side].get(key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    row.append("—")
                else:
                    row.append(("{:" + fmt + "}").format(v))
        rows.append(row)

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(col_labels), 1 + 0.5 * len(row_labels)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Alternate row colors for readability
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            if i % 2 == 0:
                cell.set_facecolor("#f5f5f2")

    # ax.set_title("Task-format shift: source vs synthetic", pad=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)