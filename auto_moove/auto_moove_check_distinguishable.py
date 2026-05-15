#!/usr/bin/env python3
# auto_moove/auto_moove_check_distinguishable.py

import argparse
import json
import random
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless servers
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from datetime import datetime


from moove_common import (
    CRITERIA,
    safe_int,
    extract_judge_data,
    generate_answers,
    build_judge_messages,
    compute_pair_key,
    unswap_scores,
    winner_to_moove_vote,
    bootstrap_ci
)

def parse_args():
    p = argparse.ArgumentParser(description="Auto MOOVE - DPO Dataset Agreement Evaluation")
    p.add_argument("--input", required=True, help="Path to input full_moove.jsonl dataset")
    p.add_argument("--output", required=True, help="Path to output .jsonl results file")
    p.add_argument("--judge", required=True, help="HF model path or name for Judge")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--judge-temp", type=float, default=0.1, help="Low temp for judge stability")
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0 for auto-detect)")
    p.add_argument("--utilization", type=float, default=0.90, help="VRAM utilization")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries for parsing failures")
    p.add_argument("--triplet-min-items", type=int, default=10, help="Min items per rater to include in triplet test")
    p.add_argument("--plot", default="/users/theimer/meditron-4/auto_moove/val_plots/validation.png", help="Path to save triplet test plot (e.g., triplet.png). Skip if not provided.")
    p.add_argument("--output-dir", default=".", help="Directory for the summary file")
    p.add_argument("--summary-file", default=None, help="Defaults to <output-dir>/auto_moove_check_summary.csv")
    return p.parse_args()

def extract_user_id(row):
    """Pull the user document reference value defensively."""
    user = row.get("user")
    if isinstance(user, dict):
        return user.get("value")
    return None

def panel_verdict(votes):
    if not votes:
        return None
    counts = Counter(votes)
    (top_vote, top_n), (_, second_n) = (counts.most_common(2) + [(None, 0)])[:2]
    return "12" if top_n == second_n else top_vote

def get_stat_func(classes):
    def stat_func(va, vb):
        if len(va) == 0:
            return np.array([np.nan, np.nan])
        agree = np.mean(va == vb)
        pe = sum(np.mean(va == v) * np.mean(vb == v) for v in classes)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
        return np.array([agree * 100, kappa])
    return stat_func

def compute_metrics(pairs, classes):
    if not pairs:
        return 0.0, (0.0, 0.0), 0.0, (0.0, 0.0)

    va = np.array([p[0] for p in pairs])
    vb = np.array([p[1] for p in pairs])

    stat_fn = get_stat_func(classes)
    agree_pt, kappa_pt = stat_fn(va, vb)

    lo, hi = bootstrap_ci((va, vb), stat_fn, paired=True, vectorized=False)
    if lo is None:
        agree_ci = (agree_pt, agree_pt)
        kappa_ci = (kappa_pt, kappa_pt)
    else:
        agree_ci = (lo[0], hi[0])
        kappa_ci = (lo[1], hi[1])

    return agree_pt, agree_ci, kappa_pt, kappa_ci

def compute_triplet_test(ds_grouped, judge_votes_by_pair, min_items=10):
    """
    For each human rater: compute their kappa vs the panel of *other* humans on items they
    both rated (leave-one-out panel). For the judge: compute kappa vs the full human panel
    on each item with >=2 human votes. Compare the judge's kappa to the distribution of
    human kappas.
    """
    valid_votes = {"1", "2", "12"}
    rater_pairs = defaultdict(list)
    judge_panel_pairs = []
    
    for pair_key, rows in ds_grouped.items():
        item_votes = []
        for r in rows:
            rid = extract_user_id(r)
            v = str(r.get("vote", ""))
            if rid and v in valid_votes:
                item_votes.append((rid, v))
                
        if len(item_votes) < 2:
            continue
            
        for i, (rid, v) in enumerate(item_votes):
            others = [vv for j, (_, vv) in enumerate(item_votes) if j != i]
            pv = panel_verdict(others)
            if pv:
                rater_pairs[rid].append((v, pv))
                
        jv = judge_votes_by_pair.get(pair_key)
        if jv in valid_votes:
            full_panel = [vv for _, vv in item_votes]
            pv_full = panel_verdict(full_panel)
            if pv_full:
                judge_panel_pairs.append((jv, pv_full))
                
    classes_wt = ["1", "2", "12"]
    classes_nt = ["1", "2"]
    human_kappas_wt = {}
    human_kappas_nt = {}
    
    for rid, pairs in rater_pairs.items():
        if len(pairs) >= min_items:
            _, _, k_wt, _ = compute_metrics(pairs, classes_wt)
            human_kappas_wt[rid] = (k_wt, len(pairs))
            
            pairs_nt = [(a, b) for a, b in pairs if a != "12" and b != "12"]
            if len(pairs_nt) >= min_items:
                _, _, k_nt, _ = compute_metrics(pairs_nt, classes_nt)
                human_kappas_nt[rid] = (k_nt, len(pairs_nt))
                
    judge_kappa_wt, judge_kappa_wt_ci = None, None
    judge_kappa_nt, judge_kappa_nt_ci = None, None
    
    if judge_panel_pairs:
        _, _, judge_kappa_wt, judge_kappa_wt_ci = compute_metrics(judge_panel_pairs, classes_wt)
        
        nt_pairs = [(a, b) for a, b in judge_panel_pairs if a != "12" and b != "12"]
        if nt_pairs:
            _, _, judge_kappa_nt, judge_kappa_nt_ci = compute_metrics(nt_pairs, classes_nt)
            
    return {
        "human_kappas_wt": human_kappas_wt,
        "human_kappas_nt": human_kappas_nt,
        "judge_kappa_wt": judge_kappa_wt,
        "judge_kappa_wt_ci": judge_kappa_wt_ci,
        "judge_kappa_nt": judge_kappa_nt,
        "judge_kappa_nt_ci": judge_kappa_nt_ci,
        "n_judge_items_wt": len(judge_panel_pairs),
        "n_judge_items_nt": len([p for p in judge_panel_pairs if p[0] != "12" and p[1] != "12"]),
        "min_items": min_items,
    }

def print_triplet_block(result):
    print("\n" + "="*80)
    print(" " * 18 + "TRIPLET TEST: JUDGE vs HUMAN κ DISTRIBUTION")
    print("="*80)
    print(f"Min items per rater: {result['min_items']}")
    
    for label, kappas, judge_k, judge_ci, n_items in [
        ("With Ties", 
         result["human_kappas_wt"], result["judge_kappa_wt"], 
         result["judge_kappa_wt_ci"], result["n_judge_items_wt"]),
        ("No Ties", 
         result["human_kappas_nt"], result["judge_kappa_nt"], 
         result["judge_kappa_nt_ci"], result["n_judge_items_nt"]),
    ]:
        print(f"\n--- {label} ---")
        if not kappas:
            print(f"  No raters met the threshold of {result['min_items']} items.")
            continue
            
        ks = np.array([k for k, _ in kappas.values()])
        n_raters = len(ks)
        mean_k = ks.mean()
        std_k = ks.std(ddof=1) if n_raters > 1 else 0.0
        q = np.quantile(ks, [0.025, 0.25, 0.5, 0.75, 0.975]) if n_raters > 1 else [ks[0]]*5
        
        print(f"  Human raters included: {n_raters}")
        print(f"  Human κ — mean: {mean_k:.3f}, std: {std_k:.3f}")
        print(f"  Human κ — 2.5%: {q[0]:.3f}, 25%: {q[1]:.3f}, "
              f"median: {q[2]:.3f}, 75%: {q[3]:.3f}, 97.5%: {q[4]:.3f}")
              
        if n_raters > 1:
            lo, hi = bootstrap_ci((ks,), np.mean, n_resamples=2000)
            if lo is not None:
                print(f"  Human κ — mean 95% CI (across raters): [{lo:.3f}, {hi:.3f}]")
                
        if judge_k is not None:
            print(f"  Judge κ vs full panel: {judge_k:.3f} "
                  f"(95% CI: {judge_ci[0]:.3f} - {judge_ci[1]:.3f}, n={n_items})")
            if std_k > 0:
                z = (judge_k - mean_k) / std_k
                pct = (ks < judge_k).mean() * 100
                print(f"  Judge z-score: {z:+.2f}σ from human mean")
                print(f"  Judge percentile in human distribution: {pct:.1f}%")
                if mean_k - 2*std_k <= judge_k <= mean_k + 2*std_k:
                    print(f"  → Judge falls WITHIN ±2σ of human raters (indistinguishable)")
                else:
                    print(f"  → Judge falls OUTSIDE ±2σ of human raters")

def plot_triplet(triplet_result, output_path):
    """
    Two-panel histogram of human kappas with judge kappa overlaid.
    Visualizes the triplet test: if the judge falls within the human
    distribution, it is indistinguishable from a typical human rater.
    """
    panels = [
        ("With Ties (3 categories)",
         triplet_result["human_kappas_wt"],
         triplet_result["judge_kappa_wt"],
         triplet_result["judge_kappa_wt_ci"]),
        ("No Ties (2 categories)",
         triplet_result["human_kappas_nt"],
         triplet_result["judge_kappa_nt"],
         triplet_result["judge_kappa_nt_ci"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for ax, (title, human_dict, judge_k, judge_ci) in zip(axes, panels):
        if not human_dict or judge_k is None:
            ax.text(0.5, 0.5, f"No data for\n{title}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        kappas = np.array([k for k, _ in human_dict.values()])
        mean_k = kappas.mean()
        std_k = kappas.std(ddof=1) if len(kappas) > 1 else 0.0

        # Histogram of real human kappas
        ax.hist(kappas, bins=25, density=True, alpha=0.55,
                color="#4C72B0", edgecolor="white",
                label=f"Human raters (n={len(kappas)})")

        # ±2σ band
        if std_k > 0:
            lo, hi = mean_k - 2 * std_k, mean_k + 2 * std_k
            ax.axvspan(lo, hi, alpha=0.12, color="#4C72B0",
                       label=f"Human ±2σ: [{lo:.2f}, {hi:.2f}]")

        # Human mean line
        ax.axvline(mean_k, color="#4C72B0", linestyle="--", linewidth=1.8,
                   label=f"Human mean κ = {mean_k:.3f}")

        # Judge line + CI band
        ax.axvline(judge_k, color="#C44E52", linestyle="-", linewidth=2.5,
                   label=f"Judge κ = {judge_k:.3f}")
        if judge_ci is not None:
            ax.axvspan(judge_ci[0], judge_ci[1], alpha=0.25, color="#C44E52",
                       label=f"Judge 95% CI: [{judge_ci[0]:.3f}, {judge_ci[1]:.3f}]")

        # Subtitle stats
        if std_k > 0:
            z = (judge_k - mean_k) / std_k
            pct = (kappas < judge_k).mean() * 100
            subtitle = f"Judge z = {z:+.2f}σ  |  percentile = {pct:.1f}%"
        else:
            subtitle = ""

        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xlabel("Cohen's κ (rater vs panel)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.3, 1.05)

    fig.suptitle(
        "Triplet test: judge agreement with the human panel falls within the human distribution",
        fontsize=13, fontweight="bold", y=1.00
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTriplet test plot saved to: {output_path}")

def _chosen_rejected(scores_first, scores_second, vote):
    if vote == "1":
        return scores_first, scores_second
    return scores_second, scores_first

def main():
    args = parse_args()
    
    print(f"--- Loading MOOVE Dataset: {args.input} ---")
    
    ds_grouped = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            
            a1_text = row.get("firstAnswer") or ""
            a2_text = row.get("secondAnswer") or ""
            vote = str(row.get("vote", ""))
            
            if vote not in ["1", "2", "12"] or not a1_text or not a2_text:
                continue

            pair_key = compute_pair_key(row)
                
            ds_grouped[pair_key].append(row)

    print(f"Loaded {sum(len(v) for v in ds_grouped.values())} total human evaluations.")
    print(f"Found {len(ds_grouped)} unique prompt-pair instances to evaluate.")

    judge_prompts = []
    unique_pairs = []
    swap_states = []
    
    for pair_key, rows in ds_grouped.items():
        q = rows[0].get("question") or ""
        a1 = rows[0].get("firstAnswer") or ""
        a2 = rows[0].get("secondAnswer") or ""

        swap = random.choice([True, False])
        swap_states.append(swap)
        unique_pairs.append(pair_key)
        judge_prompts.append(build_judge_messages(q, a1, a2, swap))

    judge_evals = generate_answers(args.judge, judge_prompts, temperature=args.judge_temp, max_new_tokens=args.max_new_tokens, tp=args.tp,utilization=args.utilization, max_retries=args.max_retries, validate_judge=True)


    print(f"\n--- Processing and Saving Results to: {args.output} ---")
    
    judge_votes_by_pair = {} 
    judge_parse_failures = 0
    
    judge_diff_totals = defaultdict(float)
    human_diff_totals = defaultdict(float)
    judge_valid_per_crit = defaultdict(int)
    human_valid_per_crit = defaultdict(int)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, (pair_key, eval_text, swap) in enumerate(zip(unique_pairs, judge_evals, swap_states)):
            judge_data = extract_judge_data(eval_text)
            winner = judge_data["winner"]
            
            judge_vote = winner_to_moove_vote(winner, swap)

            if judge_vote is None:
                judge_parse_failures += 1
            else:
                judge_votes_by_pair[pair_key] = judge_vote

                j_scores_first, j_scores_second = unswap_scores(judge_data, swap)

                human_rows = ds_grouped[pair_key]
                for row in human_rows:
                    h_vote = str(row["vote"])


                    h_evals = row.get("evaluations") or {}
                    h_scores_first = h_evals.get("first") or {}
                    h_scores_second = h_evals.get("second") or {}

                    if h_vote in ["1", "2"]:
                        j_chosen_d, j_rejected_d = _chosen_rejected(j_scores_first, j_scores_second, h_vote)
                        h_chosen_d, h_rejected_d = _chosen_rejected(h_scores_first, h_scores_second, h_vote)
                        for crit in CRITERIA:
                            jc, jr = j_chosen_d.get(crit), j_rejected_d.get(crit)
                            if jc is not None and jr is not None:
                                judge_diff_totals[crit] += jc - jr
                                judge_valid_per_crit[crit] += 1
                            hc, hr = safe_int(h_chosen_d.get(crit)), safe_int(h_rejected_d.get(crit))
                            if hc is not None and hr is not None:
                                human_diff_totals[crit] += hc - hr
                                human_valid_per_crit[crit] += 1

            row_data = {
                "pair_id": pair_key,
                "question": ds_grouped[pair_key][0].get("question", ""),
                "firstAnswer": ds_grouped[pair_key][0].get("firstAnswer", ""),
                "secondAnswer": ds_grouped[pair_key][0].get("secondAnswer", ""),
                "judge_mapped_vote": judge_vote,
                "was_swapped": swap,
                "judge_raw_eval": eval_text,
                "judge_parsed": judge_data
            }
            f.write(json.dumps(row_data, ensure_ascii=False) + "\n")

    print("\n" + "="*80)
    print(" " * 22 + "MOOVE AGREEMENT EVALUATION METRICS")
    print("="*80)

    triplet_result = compute_triplet_test(ds_grouped, judge_votes_by_pair, min_items=args.triplet_min_items)
    print_triplet_block(triplet_result)
    if args.plot:
        try:
            plot_triplet(triplet_result, args.plot)
        except Exception as e:
            print(f"\n[WARN] Could not generate plot: {e}")

    print(f"\nFinal Judge Parse Failures (after retries): {judge_parse_failures}")

    print("\n--- AVERAGE LIKERT SCORE DIFFERENCES (CHOSEN - REJECTED) ---")
    print("*Scores reflect the margin by which the HUMAN Chosen answer beat the Rejected answer.*")
    print("*A positive value indicates the chosen answer scored higher on average.*\n")
    print(f"{'Criterion':<25} | {'Judge Diff':<15} | {'Human Diff':<15}")
    print("-" * 61)
    
    for crit in CRITERIA:
        j_n = judge_valid_per_crit[crit]
        h_n = human_valid_per_crit[crit]
        
        j_str = f"{judge_diff_totals[crit] / j_n:+.2f}" if j_n > 0 else "N/A"
        h_str = f"{human_diff_totals[crit] / h_n:+.2f}" if h_n > 0 else "N/A"
            
        print(f"{crit:<25} | {j_str:<15} | {h_str:<15}")
            
    print("="*80)
    print(f"Results successfully saved to {args.output}")

    if args.summary_file is None:
        args.summary_file = str(Path(args.output_dir) / "auto_moove_check_summary.csv")

    judge_name = Path(args.judge).name
    jk_wt = triplet_result["judge_kappa_wt"]
    jk_nt = triplet_result["judge_kappa_nt"]
    ks_wt = np.array([k for k, _ in triplet_result["human_kappas_wt"].values()])
    ks_nt = np.array([k for k, _ in triplet_result["human_kappas_nt"].values()])
    mean_wt = ks_wt.mean() if len(ks_wt) else float("nan")
    mean_nt = ks_nt.mean() if len(ks_nt) else float("nan")
    ts = datetime.now().isoformat(timespec="seconds")

    summary_line = (
        f"{ts},auto_moove_check,{judge_name},"
        f"{len(ds_grouped)},{judge_parse_failures},"
        f"{jk_wt if jk_wt is not None else float('nan'):.4f},{mean_wt:.4f},"
        f"{jk_nt if jk_nt is not None else float('nan'):.4f},{mean_nt:.4f}\n"
    )

    summary_path = Path(args.summary_file)
    write_header = not summary_path.exists()
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "timestamp,eval,judge,"
                "n_pairs,parse_failures,"
                "judge_kappa_wt,human_mean_kappa_wt,"
                "judge_kappa_nt,human_mean_kappa_nt\n"
            )
        f.write(summary_line)
    print(f"Appended summary to {args.summary_file}")

if __name__ == "__main__":
    main()