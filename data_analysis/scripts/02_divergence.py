#!/usr/bin/env python3
"""
02_divergence.py
----------------
Stage 2: paired distribution comparison for the three source<->synthetic pairs.

Reads outputs/metadata/{group}.jsonl (produced by stage 1) and produces:
  - outputs/divergence/
      {pair}_divergences.csv     (field-level metrics with bootstrap CIs)
      {pair}_proportions.csv     (per-category proportions, source and synthetic)
      task_format.csv            (is_mcq / is_paired / token counts per pair/split)
      figures/
        {pair}_{field}.pdf       (paired bar charts)
        {pair}_difficulty.pdf    (ordinal histogram, QA pairs only)
        summary_heatmap.pdf      (JSD across all pairs and fields)
        task_format.pdf          (task-format shift table)
      report.md                  (auto-generated prose summary)

Only reads files — no GPU, no model. Fast to rerun.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from divergence import (  # noqa: E402
    FIELD_CATEGORIES, QA_FIELDS, DOC_FIELDS, ORDINAL_FIELDS,
    SHARED_FIELDS_FOR_GUIDELINES,
    GEOGRAPHY_BINS, bin_geography,
    categorical_divergence, ordinal_divergence,
    task_format_stats,
)
from plotting import (  # noqa: E402
    set_academic_style,
    plot_paired_categorical, plot_paired_ordinal,
    plot_summary_heatmap, plot_task_format_table,
)


# ---------------------------------------------------------------------------
# Loading tagged records
# ---------------------------------------------------------------------------

def load_tagged(path):
    """Read a stage-1 output JSONL. Returns list of dicts."""
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def extract_field_values(records, field, bin_geo=True):
    """Pull out the value of `field` from analysis_metadata across records.

    - For `geography`, bin to continent if bin_geo=True.
    - For `difficulty`, keep ints or None.
    - Records that failed parsing (_parse_ok=False) are dropped.
    - Missing field (because the record is document-schema but we're asking
      for a QA field) returns [] for that record.
    """
    values = []
    for r in records:
        meta = r.get("analysis_metadata") or {}
        if not meta.get("_parse_ok"):
            continue
        if field not in meta:
            continue
        v = meta[field]
        if field == "geography":
            if bin_geo:
                v = bin_geography(v)
        values.append(v)
    return values


# ---------------------------------------------------------------------------
# Pair-level computation
# ---------------------------------------------------------------------------

def compute_pair_divergences(pair, source_records, syn_records, n_boot):
    """Compute per-field divergences for one pair. Returns list of row dicts."""
    # Which fields do we compare? Depends on pair.
    if pair == "guidelines":
        # Source is documents, synthetic is QA — only 4 shared fields compared
        fields_to_compare = SHARED_FIELDS_FOR_GUIDELINES
    else:
        # MOOVE and Meditron: full QA schema
        fields_to_compare = QA_FIELDS

    rows = []

    for field in fields_to_compare:
        categories = FIELD_CATEGORIES[field]
        vals_s = extract_field_values(source_records, field, bin_geo=True)
        vals_y = extract_field_values(syn_records, field, bin_geo=True)
        if not vals_s or not vals_y:
            continue

        jsd, jsd_lo, jsd_hi, tv, tv_lo, tv_hi, props_s, props_y = (
            categorical_divergence(
                vals_s, vals_y, categories,
                n_iter=n_boot, seed_parts=(pair, field),
            )
        )
        rows.append({
            "pair": pair,
            "field": field,
            "metric": "jensen_shannon",
            "value": jsd,
            "ci_low": jsd_lo,
            "ci_high": jsd_hi,
            "n_source": len(vals_s),
            "n_synthetic": len(vals_y),
        })
        rows.append({
            "pair": pair,
            "field": field,
            "metric": "total_variation",
            "value": tv,
            "ci_low": tv_lo,
            "ci_high": tv_hi,
            "n_source": len(vals_s),
            "n_synthetic": len(vals_y),
        })
        # Attach the proportions for later plotting — pack as side effect
        rows[-2]["_props_source"] = props_s.tolist()
        rows[-2]["_props_synthetic"] = props_y.tolist()
        rows[-2]["_categories"] = list(categories)

    # Ordinal field (difficulty) — only for QA pairs (MOOVE, Meditron)
    if pair in ("moove", "curated"):
        vals_s = extract_field_values(source_records, "difficulty", bin_geo=False)
        vals_y = extract_field_values(syn_records, "difficulty", bin_geo=False)
        vals_s_int = [v for v in vals_s if v is not None]
        vals_y_int = [v for v in vals_y if v is not None]
        if vals_s_int and vals_y_int:
            w1, w1_lo, w1_hi, mean_s, mean_y, med_s, med_y = ordinal_divergence(
                vals_s_int, vals_y_int, n_iter=n_boot,
                seed_parts=(pair, "difficulty"),
            )
            rows.append({
                "pair": pair,
                "field": "difficulty",
                "metric": "wasserstein_1",
                "value": w1,
                "ci_low": w1_lo,
                "ci_high": w1_hi,
                "n_source": len(vals_s_int),
                "n_synthetic": len(vals_y_int),
                "_mean_source": mean_s,
                "_mean_synthetic": mean_y,
                "_median_source": med_s,
                "_median_synthetic": med_y,
                "_values_source": vals_s_int,
                "_values_synthetic": vals_y_int,
            })

    return rows


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_divergence_csv(rows, out_path):
    """Public CSV with just the metric rows (no attached props/values)."""
    fields = ["pair", "field", "metric", "value", "ci_low", "ci_high",
              "n_source", "n_synthetic"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields if k in r})


def write_proportions_csv(all_rows, out_path):
    """Per-category proportions for every (pair, field, category)."""
    fields = ["pair", "field", "category", "prop_source", "prop_synthetic"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            if r["metric"] != "jensen_shannon":
                continue
            cats = r.get("_categories") or []
            ps = r.get("_props_source") or []
            py = r.get("_props_synthetic") or []
            for c, a, b in zip(cats, ps, py):
                w.writerow({
                    "pair": r["pair"],
                    "field": r["field"],
                    "category": c,
                    "prop_source": "{:.6f}".format(a),
                    "prop_synthetic": "{:.6f}".format(b),
                })


def write_task_format_csv(stats_per_pair, out_path):
    rows = []
    for pair, sides in stats_per_pair.items():
        for split, stats in sides.items():
            row = {"pair": pair, "split": split}
            row.update(stats)
            rows.append(row)
    if not rows:
        return
    fields = ["pair", "split", "n", "pct_paired", "pct_mcq", "pct_document",
              "pct_has_think_block", "median_q_tokens", "median_a_tokens",
              "median_full_tokens"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _classify_jsd(v):
    """Rough interpretation labels for the paper summary."""
    if v < 0.05:
        return "near-identical"
    if v < 0.15:
        return "small shift"
    if v < 0.30:
        return "moderate shift"
    if v < 0.50:
        return "large shift"
    return "severe shift"


def write_report(all_rows, task_stats, out_path):
    lines = []
    lines.append("# Source vs synthetic distribution shift — stage 2 report\n")
    lines.append("Metrics: Jensen-Shannon divergence (JSD, in [0, 1]) for "
                 "categorical fields, Wasserstein-1 for difficulty. "
                 "Bootstrap 95% CIs, 1000 iterations per metric.\n")

    # Per-pair summary
    pairs = sorted({r["pair"] for r in all_rows})
    for pair in pairs:
        lines.append("\n## {}\n".format(pair.upper()))
        pair_rows = [r for r in all_rows if r["pair"] == pair
                     and r["metric"] in ("jensen_shannon", "wasserstein_1")]
        if not pair_rows:
            continue
        # Stable field order
        field_order = QA_FIELDS + ORDINAL_FIELDS
        pair_rows.sort(key=lambda r: (
            field_order.index(r["field"]) if r["field"] in field_order else 99
        ))
        lines.append("| Field | Metric | Value | 95% CI | Interpretation |")
        lines.append("|---|---|---|---|---|")
        for r in pair_rows:
            label = (
                "Wasserstein-1" if r["metric"] == "wasserstein_1"
                else "JSD"
            )
            interp = (
                _classify_jsd(r["value"]) if r["metric"] == "jensen_shannon"
                else ""
            )
            lines.append("| {} | {} | {:.3f} | [{:.3f}, {:.3f}] | {} |".format(
                r["field"], label, r["value"], r["ci_low"], r["ci_high"], interp
            ))

    # Task-format shift
    if task_stats:
        lines.append("\n## Task-format shift\n")
        lines.append("Task-format differences between source and synthetic. "
                     "These are not divergences but raw percentages/medians.\n")
        lines.append("| Pair | Side | n | %paired | %MCQ | %doc | %<think> "
                     "| Q toks | A toks | Full toks |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for pair, sides in task_stats.items():
            for split, s in sides.items():
                lines.append(
                    "| {} | {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} "
                    "| {:.0f} | {:.0f} | {:.0f} |".format(
                        pair, split, s["n"], s["pct_paired"], s["pct_mcq"],
                        s["pct_document"], s["pct_has_think_block"],
                        s["median_q_tokens"], s["median_a_tokens"],
                        s["median_full_tokens"],
                    )
                )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-dir", type=str, default="outputs/metadata")
    p.add_argument("--out-dir", type=str, default="outputs/divergence")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations for CI estimation.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation (CSVs only, for fast iteration).")
    return p.parse_args()


def main():
    args = parse_args()
    meta_dir = Path(args.metadata_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
        set_academic_style()

    # Load all 6 files
    files = {
        ("moove", "source"):        meta_dir / "moove__source.jsonl",
        ("moove", "synthetic"):     meta_dir / "moove__synthetic.jsonl",
        ("curated", "source"):     meta_dir / "Curated_QA__source.jsonl",
        ("curated", "synthetic"):  meta_dir / "Curated_QA__synthetic.jsonl",
        ("guidelines", "source"):   meta_dir / "guidelines__source.jsonl",
        ("guidelines", "synthetic"):meta_dir / "guidelines__synthetic.jsonl",
    }

    records_by_group = {}
    for key, path in files.items():
        recs = load_tagged(path)
        records_by_group[key] = recs
        print("[load] {}: {} records".format(path.name, len(recs)))

    # ------------------------------------------------------------------
    # Divergences per pair
    # ------------------------------------------------------------------
    all_rows = []
    for pair in ("moove", "curated", "guidelines"):
        src = records_by_group[(pair, "source")]
        syn = records_by_group[(pair, "synthetic")]
        if not src or not syn:
            print("[skip] pair {}: missing source or synthetic".format(pair))
            continue
        print("\n[{}] computing divergences ({} source, {} synthetic)".format(
            pair, len(src), len(syn)
        ))
        rows = compute_pair_divergences(pair, src, syn, n_boot=args.n_boot)
        for r in rows:
            if r["metric"] in ("jensen_shannon", "wasserstein_1"):
                print("  {:>14}  {:<15} = {:.3f}  [{:.3f}, {:.3f}]".format(
                    r["field"], r["metric"], r["value"], r["ci_low"], r["ci_high"]
                ))
        # Write per-pair CSV (public columns only)
        write_divergence_csv(rows, out_dir / "{}_divergences.csv".format(pair))
        all_rows.extend(rows)

    # Combined proportions CSV (handy for custom plots later)
    write_proportions_csv(all_rows, out_dir / "proportions.csv")

    # ------------------------------------------------------------------
    # Task-format stats
    # ------------------------------------------------------------------
    task_stats = {}
    for pair in ("moove", "curated", "guidelines"):
        task_stats[pair] = {
            "source": task_format_stats(records_by_group[(pair, "source")]),
            "synthetic": task_format_stats(records_by_group[(pair, "synthetic")]),
        }
    write_task_format_csv(task_stats, out_dir / "task_format.csv")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if not args.no_figures:
        print("\n[plots] generating figures...")
        # Paired bar charts + ordinal histograms
        for r in all_rows:
            if r["metric"] == "jensen_shannon":
                out_path = fig_dir / "{}_{}.pdf".format(r["pair"], r["field"])
                plot_paired_categorical(
                    r["_categories"], r["_props_source"], r["_props_synthetic"],
                    r["field"], r["pair"],
                    r["value"], r["ci_low"], r["ci_high"],
                    r["n_source"], r["n_synthetic"],
                    out_path=out_path,
                )
                print("  wrote {}".format(out_path.name))
            elif r["metric"] == "wasserstein_1":
                out_path = fig_dir / "{}_{}.pdf".format(r["pair"], r["field"])
                plot_paired_ordinal(
                    r["_values_source"], r["_values_synthetic"],
                    r["field"], r["pair"],
                    r["value"], r["ci_low"], r["ci_high"],
                    r["_mean_source"], r["_mean_synthetic"],
                    r["n_source"], r["n_synthetic"],
                    out_path=out_path,
                )
                print("  wrote {}".format(out_path.name))

        # Summary heatmap of JSD values
        pairs_order = ["moove", "curated", "guidelines"]
        fields_order = QA_FIELDS  # union of QA fields; doc pair has subset
        matrix = np.full((len(pairs_order), len(fields_order)), np.nan)
        for r in all_rows:
            if r["metric"] != "jensen_shannon":
                continue
            if r["pair"] in pairs_order and r["field"] in fields_order:
                i = pairs_order.index(r["pair"])
                j = fields_order.index(r["field"])
                matrix[i, j] = r["value"]
        plot_summary_heatmap(
            pairs_order, fields_order, matrix,
            out_path=fig_dir / "summary_heatmap.pdf",
        )
        print("  wrote summary_heatmap.pdf")

        # Task-format table
        plot_task_format_table(task_stats, out_path=fig_dir / "task_format.pdf")
        print("  wrote task_format.pdf")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    write_report(all_rows, task_stats, out_dir / "report.md")
    print("\n[done] wrote {}".format(out_dir / "report.md"))


if __name__ == "__main__":
    main()
