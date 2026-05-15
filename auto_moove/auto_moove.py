#!/usr/bin/env python3
# auto_moove/auto_moove.py

import argparse
import json
import re
import random
import hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from moove_common import (
    CRITERIA,
    extract_judge_data,
    generate_answers,
    build_judge_messages,
    compute_ci,
    compute_pair_key,
    unswap_winner,
    unswap_scores,
)

def parse_args():
    p = argparse.ArgumentParser(description="Auto MOOVE - Automated Medical LLM Evaluation with Positional Debias")
    p.add_argument("--input", required=True, help="Path to input .jsonl dataset (MOOVE format)")
    p.add_argument("--output", required=True, help="Path to output .jsonl results file")
    p.add_argument("--model1", required=True, help="HF model path or name for Model 1")
    p.add_argument("--model2", required=True, help="HF model path or name for Model 2")
    p.add_argument("--judge", required=True, help="HF model path or name for Judge")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--judge-temp", type=float, default=0.1, help="Low temp for judge stability")
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0 for auto-detect)")
    p.add_argument("--utilization", type=float, default=0.90, help="VRAM utilization")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries for parsing failures")
    p.add_argument("--output-dir", default=".", help="Directory for the summary file")
    p.add_argument("--summary-file", default=None, help="Defaults to <output-dir>/auto_moove_summary.csv")
    return p.parse_args()

CACHE_ROOT = Path("/capstor/scratch/cscs/theimer/auto_moove_cache")

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def _prompt_hash(prompt) -> str:
    if isinstance(prompt, list):
        payload = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    else:
        payload = prompt
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def load_cache(model_name: str) -> dict:
    cache_file = CACHE_ROOT / _sanitize(model_name) / "answers.jsonl"
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["key"]] = row["answer"]
    return cache

def append_cache(model_name: str, new_entries: dict):
    cache_dir = CACHE_ROOT / _sanitize(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "answers.jsonl"
    with open(cache_file, "a", encoding="utf-8") as f:
        for key, answer in new_entries.items():
            f.write(json.dumps({"key": key, "answer": answer}, ensure_ascii=False) + "\n")

def generate_answers_cached(model_name, prompts, args, temperature):
    param_tag = f"t{temperature}_n{args.max_new_tokens}"
    keys = [f"{param_tag}_{_prompt_hash(p)}" for p in prompts]

    cache = load_cache(model_name)
    results = [cache.get(k) for k in keys]

    miss_indices = [i for i, r in enumerate(results) if r is None]
    if not miss_indices:
        print(f">>> Full cache hit for {model_name} ({len(prompts)} prompts) <<<")
        return results

    print(f">>> Cache: {len(prompts) - len(miss_indices)} hits, {len(miss_indices)} misses for {model_name} <<<")
    miss_prompts = [prompts[i] for i in miss_indices]
    miss_outputs = generate_answers(model_name, miss_prompts, temperature=temperature, max_new_tokens=args.max_new_tokens, tp=args.tp,utilization=args.utilization, max_retries=0, validate_judge=False)

    new_entries = {}
    for idx, out in zip(miss_indices, miss_outputs):
        results[idx] = out
        new_entries[keys[idx]] = out
    append_cache(model_name, new_entries)

    return results

def main():
    args = parse_args()
    
    print(f"--- Loading MOOVE Dataset: {args.input} ---")
    seen_keys = set()
    original_prompts = []
    prompt_keys = []  # keep for traceability in output

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = row.get("prompt") or row.get("question")
            
            if not prompt_text:
                print("Warning: Found a row without a 'prompt' or 'question' key. Skipping.")
                continue
            
            pair_key = compute_pair_key(row)
            
            if pair_key in seen_keys:
                continue
            seen_keys.add(pair_key)
            
            original_prompts.append(prompt_text)
            prompt_keys.append(pair_key)

    print(f"Loaded {len(original_prompts)} unique prompts (deduplicated).")

    m1_answers = generate_answers_cached(args.model1, original_prompts, args, args.temperature, is_judge=False)
    m2_answers = generate_answers_cached(args.model2, original_prompts, args, args.temperature, is_judge=False)

    # Step 3: Prepare Judge Prompts with Random Swapping
    judge_prompts = []
    swap_states = []
    
    for prompt, m1, m2 in zip(original_prompts, m1_answers, m2_answers):
        swap = random.choice([True, False])
        swap_states.append(swap)
        judge_prompts.append(build_judge_messages(prompt, m1, m2, swap))

    # Step 4: Generate Judge Evaluations (with Retries enabled)
    judge_evals = generate_answers(args.judge, judge_prompts, temperature=args.judge_temp, max_new_tokens=args.max_new_tokens, tp=args.tp,utilization=args.utilization, max_retries=args.max_retries, validate_judge=True)
    

    # Step 5: Process and Save Results
    print(f"\n--- Processing and Saving Results to: {args.output} ---")
    
    m1_wins = []
    m2_wins = []
    ties = []
    parse_failures = 0
    
    m1_scores_list = defaultdict(list)
    m2_scores_list = defaultdict(list)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, (prompt, m1, m2, eval_text, swap) in enumerate(zip(original_prompts, m1_answers, m2_answers, judge_evals, swap_states)):
            judge_data = extract_judge_data(eval_text)
            
            actual_winner = unswap_winner(judge_data["winner"], swap)
            actual_m1_scores, actual_m2_scores = unswap_scores(judge_data, swap)
            
            # Record Win Rates
            if actual_winner == "Model 1":
                m1_wins.append(1)
                m2_wins.append(0)
                ties.append(0)
            elif actual_winner == "Model 2":
                m1_wins.append(0)
                m2_wins.append(1)
                ties.append(0)
            elif actual_winner == "Tie":
                m1_wins.append(0)
                m2_wins.append(0)
                ties.append(1)
            else:
                parse_failures += 1

            # Tally Likert Scores for averages
            for crit in CRITERIA:
                s1 = actual_m1_scores.get(crit)
                s2 = actual_m2_scores.get(crit)
                if s1 is not None and s2 is not None:
                    m1_scores_list[crit].append(s1)
                    m2_scores_list[crit].append(s2)

            # Update the judge_data object for accurate JSON logging
            judge_data["actual_winner"] = actual_winner
            judge_data["actual_model_1_scores"] = actual_m1_scores
            judge_data["actual_model_2_scores"] = actual_m2_scores
            judge_data["was_swapped"] = swap

            row_data = {
                "id": i,
                "pair_id": prompt_keys[i],
                "prompt": prompt,
                "model_1": args.model1,
                "model_1_answer": m1,
                "model_2": args.model2,
                "model_2_answer": m2,
                "judge_raw_eval": eval_text,
                "judge_parsed": judge_data
            }
            f.write(json.dumps(row_data, ensure_ascii=False) + "\n")

    # Step 6: Print Bootstrapped Analysis
    total = len(original_prompts)
    
    print("\n" + "="*80)
    print(" " * 28 + "AUTO MOOVE ANALYSIS")
    print(" " * 26 + "(Positional Bias Mitigated)")
    print("="*80)
    print(f"Total Evaluated: {total}")
    print(f"Final Parse Failures (after retries): {parse_failures}\n")

    # Win Rates CIs
    m1_win_pt, m1_win_ci = compute_ci(m1_wins)
    m2_win_pt, m2_win_ci = compute_ci(m2_wins)
    ties_pt, ties_ci = compute_ci(ties)

    print("--- WIN RATES (WITH 95% CI) ---")
    print(f"Model 1 ({args.model1}) Wins: {sum(m1_wins)} ({m1_win_pt*100:.1f}% [95% CI: {m1_win_ci[0]*100:.1f}% - {m1_win_ci[1]*100:.1f}%])")
    print(f"Model 2 ({args.model2}) Wins: {sum(m2_wins)} ({m2_win_pt*100:.1f}% [95% CI: {m2_win_ci[0]*100:.1f}% - {m2_win_ci[1]*100:.1f}%])")
    print(f"Ties: {sum(ties)} ({ties_pt*100:.1f}% [95% CI: {ties_ci[0]*100:.1f}% - {ties_ci[1]*100:.1f}%])")

    print("\n--- AVERAGE LIKERT SCORES & DIFFERENCES (WITH 95% CI) ---")
    print("*Differences indicate Model 1 vs Model 2 margin.*")
    print("*A positive difference means Model 1 scored higher.*\n")
    print(f"{'Criterion':<25} | {'Model 1':<7} | {'Model 2':<7} | {'Difference (M1 - M2)':<25}")
    print("-" * 75)
    
    for crit in CRITERIA:
        if m1_scores_list[crit]:
            m1_pt, _ = compute_ci(m1_scores_list[crit])
            m2_pt, _ = compute_ci(m2_scores_list[crit])
            
            diffs = [a - b for a, b in zip(m1_scores_list[crit], m2_scores_list[crit])]
            diff_pt, diff_ci = compute_ci(diffs)
            
            print(f"{crit:<25} | {m1_pt:.2f}    | {m2_pt:.2f}    | {diff_pt:+.2f} (95% CI: {diff_ci[0]:+.2f} to {diff_ci[1]:+.2f})")
        else:
            print(f"{crit:<25} | N/A      | N/A      | N/A")
            
    print("\n--- DERIVED METRICS (WITH 95% CI) ---")

    # 1. Net Win Rate (M2 - M1), per-item signed score in {-1, 0, +1}
    #    +1 if M2 won, -1 if M1 won, 0 if tie. Mean = p2 - p1.
    net_per_item = [b - a for a, b in zip(m1_wins, m2_wins)]
    net_pt, net_ci = compute_ci(net_per_item)
    print(f"Net Win Rate (M2 - M1):     {net_pt*100:+.1f}% (95% CI: {net_ci[0]*100:+.1f}% to {net_ci[1]*100:+.1f}%)")

    # 2. Adjusted Win Rate for Model 2 (ties counted as 0.5)
    #    Per-item score in {0, 0.5, 1}.
    awr_per_item = [w + 0.5 * t for w, t in zip(m2_wins, ties)]
    awr_pt, awr_ci = compute_ci(awr_per_item)
    print(f"Adjusted Win Rate (M2):     {awr_pt*100:.1f}% (95% CI: {awr_ci[0]*100:.1f}% to {awr_ci[1]*100:.1f}%)")

    # 3. Overall Likert Delta — average per-item delta across all criteria,
    #    then bootstrap over items (preserves pairing, respects sample size).
    per_item_avg_delta = []
    n_items = min(len(m1_scores_list[c]) for c in CRITERIA if m1_scores_list[c])
    for i in range(n_items):
        deltas_i = []
        for crit in CRITERIA:
            if i < len(m1_scores_list[crit]):
                deltas_i.append(m1_scores_list[crit][i] - m2_scores_list[crit][i])
        if deltas_i:
            per_item_avg_delta.append(sum(deltas_i) / len(deltas_i))
    if per_item_avg_delta:
        d_pt, d_ci = compute_ci(per_item_avg_delta)
        print(f"Overall Likert Delta (M1-M2): {d_pt:+.3f} (95% CI: {d_ci[0]:+.3f} to {d_ci[1]:+.3f})")
    else:
        print("Overall Likert Delta (M1-M2): N/A")

    print("="*80)
    print(f"Results successfully saved to {args.output}")

    if args.summary_file is None:
        args.summary_file = str(Path(args.output_dir) / "auto_moove_summary.csv")

    m1_name = Path(args.model1).name
    m2_name = Path(args.model2).name
    judge_name = Path(args.judge).name
    ts = datetime.now().isoformat(timespec="seconds")

    summary_line = (
        f"{ts},auto_moove,{m1_name},{m2_name},{judge_name},"
        f"{total},{sum(m1_wins)},{sum(m2_wins)},{sum(ties)},"
        f"{m1_win_pt:.4f},{m1_win_ci[0]:.4f},{m1_win_ci[1]:.4f},"
        f"{m2_win_pt:.4f},{m2_win_ci[0]:.4f},{m2_win_ci[1]:.4f},"
        f"{net_pt:.4f},{net_ci[0]:.4f},{net_ci[1]:.4f}\n"
    )

    summary_path = Path(args.summary_file)
    write_header = not summary_path.exists()
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "timestamp,eval,model1,model2,judge,"
                "n,m1_wins,m2_wins,ties,"
                "m1_rate,m1_ci_lo,m1_ci_hi,"
                "m2_rate,m2_ci_lo,m2_ci_hi,"
                "net,net_ci_lo,net_ci_hi\n"
            )
        f.write(summary_line)
    print(f"Appended summary to {args.summary_file}")

if __name__ == "__main__":
    main()