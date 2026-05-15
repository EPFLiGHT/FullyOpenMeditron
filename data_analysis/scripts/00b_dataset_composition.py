#!/usr/bin/env python3
"""
00b_dataset_composition.py

Generate pie charts for paper: dataset composition and synthetic-vs-source split.
Produces four separate figures:
  - composition_by_group_records.pdf
  - composition_by_group_tokens.pdf
  - composition_syn_vs_src_records.pdf
  - composition_syn_vs_src_tokens.pdf

For the by-group plots, MOOVE source and Guidelines source are intentionally
excluded — see comments below for why.

No GPU, no sampling — iterates the included datasets once to count records
and tokens, using the same loaders as the rest of the pipeline.

Data paths are resolved by `load_data_paths(args)` (see src/config.py), with
priority: CLI flag > YAML config (--config) > built-in default.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Default config path is script-relative so the script runs correctly from
# anywhere (parent repo root, the analysis subdir, etc.).
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)

# from loader import (                                        # noqa: E402
#     load_moove_source, load_moove_synthetic,
#     load_meditron_source, load_meditron_synthetic,
#     load_guidelines_source, load_guidelines_synthetic,
# )
from loader import (                                        # noqa: E402
    load_moove_synthetic,
    load_meditron_source, load_meditron_synthetic,
    load_guidelines_synthetic,
)
from plotting import set_academic_style                     # noqa: E402
from config import load_data_paths                          # noqa: E402


# Styling — same palette family as the divergence figures, but with enough
# distinct hues for 6 slices. Teal/coral variants for source/synthetic,
# purple/amber for the third pair to stay visually distinct.
GROUP_COLORS = {
    "moove__source":          "#1D9E75",  # teal (source)
    "moove__synthetic":       "#5DCAA5",  # teal lighter
    "curated__source":        "#D85A30",  # coral (source)
    "curated__synthetic":     "#F0997B",  # coral lighter
    "guidelines__source":     "#534AB7",  # purple (source)
    "guidelines__synthetic":  "#AFA9EC",  # purple lighter
}

GROUP_LABELS = {
    # "moove__source":          "MOOVE source",
    "moove__synthetic":       "MOOVE synthetic",
    "curated__source":        "Curated QA source",
    "curated__synthetic":     "Curated QA synthetic",
    # "guidelines__source":     "Guidelines source",
    "guidelines__synthetic":  "Guidelines synthetic",
}

SPLIT_COLORS = {
    "source":    "#1D9E75",  # teal
    "synthetic": "#D85A30",  # coral
}


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def _count_records_and_tokens(records_iter):
    """Walk an iterator of Record and return (n_records, total_tokens).
    Tokens = whitespace-split count of full_text."""
    n = 0
    toks = 0
    for r in records_iter:
        n += 1
        if r.full_text:
            toks += len(r.full_text.split())
    return n, toks


def build_plan(args):
    paths = load_data_paths(args)

    # MOOVE source and Guidelines source are intentionally excluded from
    # the by-group composition plot. Re-enable by uncommenting the
    # corresponding entries below.

    # def _guidelines_src_stream():
    #     for f in sorted(paths["guidelines_dir"].glob("*.jsonl")):
    #         for r in load_guidelines_source(f):
    #             yield r

    return [
        # ("moove__source",          lambda: load_moove_source(paths["moove_source"])),
        ("moove__synthetic",       lambda: load_moove_synthetic(paths["moove_synthetic"])),
        ("curated__source",        lambda: load_meditron_source(paths["curated_source"])),
        ("curated__synthetic",     lambda: load_meditron_synthetic(paths["curated_synthetic"])),
        # ("guidelines__source",     _guidelines_src_stream),
        ("guidelines__synthetic",  lambda: load_guidelines_synthetic(paths["guidelines_synthetic"])),
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _human(n):
    """Format a count compactly: 1,234,567 -> '1.23M'."""
    if n >= 1_000_000:
        return "{:.2f}M".format(n / 1_000_000)
    if n >= 1_000:
        return "{:.1f}k".format(n / 1_000)
    return str(n)


def _autopct_with_count(total, values, min_pct_for_label=6.0):
    """Return a matplotlib autopct callable that renders 'X.X%\nN'.
    Returns empty string for slices below min_pct_for_label so labels
    don't overlap on thin wedges — the legend covers those counts."""
    idx = [0]
    def fmt(pct):
        v = values[idx[0]]
        idx[0] += 1
        if pct < min_pct_for_label:
            return ""
        return "{:.1f}%\n{}".format(pct, _human(v))
    return fmt


def _draw_pie(ax, labels, values, colors, title):
    total = sum(values)
    if total == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    autopct = _autopct_with_count(total, values)

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,             # legend handles labels — cleaner for 6 slices
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=autopct,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 20, "color": "black"},
    )

    # Percentage labels inside slices — white text, small, bold
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)
        at.set_fontweight("bold")

    # Legend with count + percentage suffixes
    legend_labels = []
    for l, v in zip(labels, values):
        pct = 100.0 * v / total if total else 0.0
        legend_labels.append("{}  ({}, {:.1f}%)".format(l, _human(v), pct))

    leg = ax.legend(wedges, legend_labels, loc="center left",
                    bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=16)
    for text in leg.get_texts():
        text.set_color("black")
        # text.set_fontweight("bold")


def plot_composition_by_group(counts_records, counts_tokens, out_dir):
    """Saves two separate figures: one for records, one for tokens."""
    # group_order = [
    #     "moove__source", "moove__synthetic",
    #     "curated__source", "curated__synthetic",
    #     "guidelines__source", "guidelines__synthetic",
    # ]
    group_order = [
        "moove__synthetic",
        "curated__source", "curated__synthetic",
        "guidelines__synthetic",
    ]
    labels = [GROUP_LABELS[g] for g in group_order]
    colors = [GROUP_COLORS[g] for g in group_order]

    vals_records = [counts_records[g] for g in group_order]
    vals_tokens = [counts_tokens[g] for g in group_order]

    # Plot 1: Records
    fig1, ax1 = plt.subplots(figsize=(8, 5.2))
    _draw_pie(ax1, labels, vals_records, colors, "")
    plt.title("Total records = {}".format(_human(sum(vals_records))), pad=10, fontsize=16)
    plt.tight_layout()
    out_a = out_dir / "composition_by_group_records.pdf"
    plt.savefig(out_a)
    plt.close(fig1)
    print("Wrote {}".format(out_a))

    # Plot 2: Tokens
    fig2, ax2 = plt.subplots(figsize=(8, 5.2))
    _draw_pie(ax2, labels, vals_tokens, colors, "")
    plt.title("Total tokens = {}".format(_human(sum(vals_tokens))), pad=10, fontsize=16)
    plt.tight_layout()
    out_b = out_dir / "composition_by_group_tokens.pdf"
    plt.savefig(out_b)
    plt.close(fig2)
    print("Wrote {}".format(out_b))


def plot_syn_vs_src(counts_records, counts_tokens, out_dir):
    """Saves two separate figures: one for records, one for tokens."""
    def _sum(d, split):
        return sum(v for k, v in d.items() if k.endswith("__" + split))

    rec_src = _sum(counts_records, "source")
    rec_syn = _sum(counts_records, "synthetic")
    tok_src = _sum(counts_tokens, "source")
    tok_syn = _sum(counts_tokens, "synthetic")

    labels = ["Source", "Synthetic"]
    colors = [SPLIT_COLORS["source"], SPLIT_COLORS["synthetic"]]

    # Plot 1: Records
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    _draw_pie(ax1, labels, [rec_src, rec_syn], colors, "")
    plt.title("By record count  (total = {})".format(_human(rec_src + rec_syn)), pad=10)
    plt.tight_layout()
    out_c = out_dir / "composition_syn_vs_src_records.pdf"
    plt.savefig(out_c)
    plt.close(fig1)
    print("Wrote {}".format(out_c))

    # Plot 2: Tokens
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    _draw_pie(ax2, labels, [tok_src, tok_syn], colors, "")
    plt.title("By token count  (total = {})".format(_human(tok_src + tok_syn)), pad=10)
    plt.tight_layout()
    out_d = out_dir / "composition_syn_vs_src_tokens.pdf"
    plt.savefig(out_d)
    plt.close(fig2)
    print("Wrote {}".format(out_d))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                   help="YAML config with data paths. CLI flags override config values. "
                        "Pass --config '' to skip and use built-in defaults.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--moove-source", type=str, default=None)
    p.add_argument("--moove-synthetic", type=str, default=None)
    p.add_argument("--meditron-source", type=str, default=None)
    p.add_argument("--meditron-synthetic", type=str, default=None)
    p.add_argument("--guidelines-dir", type=str, default=None)
    p.add_argument("--guidelines-synthetic", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs/composition")
    p.add_argument("--cache", type=str, default=None,
                   help="Optional path to cache counts JSON — on repeat runs, "
                   "load counts from here instead of re-walking 216k records.")
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_academic_style()

    cache_path = Path(args.cache) if args.cache else (out_dir / "counts.json")

    # Load counts from cache if available and not forcing rebuild
    counts_records = {}
    counts_tokens = {}
    if cache_path.exists() and not args.rebuild_cache:
        print("[cache] loading counts from {}".format(cache_path))
        data = json.loads(cache_path.read_text())
        counts_records = data.get("records", {})
        counts_tokens = data.get("tokens", {})

    # Figure out which groups still need counting
    plan = build_plan(args)
    expected_groups = {g for g, _ in plan}
    missing = expected_groups - set(counts_records.keys())

    if missing:
        print("[count] need to count {} group(s): {}".format(
            len(missing), sorted(missing)))
        for group_name, loader_fn in plan:
            if group_name in counts_records and group_name in counts_tokens:
                continue
            print("  counting {}...".format(group_name))
            n, toks = _count_records_and_tokens(loader_fn())
            counts_records[group_name] = n
            counts_tokens[group_name] = toks
            print("    -> {} records, {} tokens".format(_human(n), _human(toks)))

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "records": counts_records,
            "tokens": counts_tokens,
        }, indent=2))
        print("[cache] wrote {}".format(cache_path))

    # Print summary table
    print("\nGroup-level counts:")
    print("  {:<26} {:>12} {:>16}".format("group", "records", "tokens"))
    print("  " + "-" * 58)
    for g in ["moove__source", "moove__synthetic",
              "curated__source", "curated__synthetic",
              "guidelines__source", "guidelines__synthetic"]:
        print("  {:<26} {:>12} {:>16}".format(
            g, _human(counts_records.get(g, 0)), _human(counts_tokens.get(g, 0))))

    print("\nGenerating plots...")
    # Plotting functions now take the output directory and handle saving internally
    plot_composition_by_group(counts_records, counts_tokens, out_dir)
    plot_syn_vs_src(counts_records, counts_tokens, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()