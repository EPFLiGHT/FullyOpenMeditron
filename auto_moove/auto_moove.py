#!/usr/bin/env python3
# auto_moove/auto_moove.py


"""
bash new_launch.sh auto_moove/auto_moove.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --model1 swiss-ai/Apertus-v1.5-70B \
    --model2 <org>/Apertus-70B-MeditronFO \
    --judge Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --rubric moove \
    --output auto_moove/results

bash new_launch.sh auto_moove/auto_moove.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --model1 swiss-ai/Apertus-v1.5-70B \
    --model2 <org>/Apertus-70B-MeditronFO \
    --judge Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --rubric checklist \
    --output auto_moove/results


    
"""

import argparse, json, random
from pathlib import Path
from datetime import datetime
import pandas as pd

import auto_moove_check as chk
from auto_moove_prompt import MOOVE, criteria_for, verdict_model
from moove_meta import extract_meta, pair_key_for, question_of
from moove_stats import compute_ci, fmt_ci, HAVE_SCIPY
from auto_moove_common import (
    effective_temperature,
    generate,
    generate_answers_cached,
    build_judge_messages,
    extract_judge_data,
    has_text,
    unswap_winner,
    unswap_scores,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--model1", required=True)
    p.add_argument("--model2", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--rubric", choices=["moove", "checklist", "both", "preference", "winner",
                                        "moove_pref"],
                   default="both",
                   help="moove_pref: a preference per MOOVE criterion and no overall "
                        "verdict, the rubric moove_criteria_agreement.py validates")
    # auto_moove_check.py's verdict flags. Log-prob scoring and/or judging
    # both orders route through its judge_pairs; the default (generate, one
    # random order) keeps the path below so earlier runs reproduce.
    p.add_argument("--scoring", choices=["generate", "logprob"], default="generate")
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--disagreement-to-equal", action="store_true")
    p.add_argument("--equal-threshold", type=float, default=0.0)
    p.add_argument("--judge-utilization", type=float, default=None,
                   help="vLLM memory share for the judge (default --utilization); "
                        "log-prob scoring of large-vocabulary judges needs ~0.5")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--utilization", type=float, default=0.85)
    # vLLM context window. The judge sees the question plus BOTH answers, so it
    # needs far more room than answer generation does -- a hardcoded 8192 here
    # aborted the whole judge stage on the first over-long pair.
    # Apertus-70B supports 65536, Qwen3-30B-A3B 262144.
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="context window for answer generation")
    p.add_argument("--judge-max-model-len", type=int, default=32768,
                   help="context window for the judge (sees both answers)")
    # vLLM defaults to temperature=1.0. Greedy + a fixed seed makes a run
    # reproducible; both are part of the answer-cache key.
    p.add_argument("--temperature", type=float, default=0.0)
    # Per-side answer temperatures, e.g. one model against itself at 0.7 vs
    # 1.0. Default to --temperature; the judge always uses --temperature.
    p.add_argument("--temperature1", type=float, default=None)
    p.add_argument("--temperature2", type=float, default=None)
    # Sent as the system turn to both answer models (not the judge, which sees
    # only the question). Part of the cache key via the message list.
    p.add_argument("--system-prompt-file", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary-file", type=Path, default=Path("auto_moove_summary.csv"))
    args = p.parse_args()
    args.samples, args.reasoning_effort, args.batch = 1, None, False
    chk.check_judge_args(p, args)
    return args


def load_prompts(path):
    """Unique questions, their pair keys, and the case metadata of the rows they
    came from. Both the key and the metadata vocabulary live in moove_meta so
    that results_to_inspect.py can join the two back together the same way.

    The key falls back to a hash of the question. Its field order must match
    the question lookup here: full_moove.jsonl (the raw MOOVE export) carries
    "question", not "left_question_text", so getting that wrong hashed the
    empty string for every row and collapsed 24681 rows to ONE prompt."""
    seen = set()
    prompts, keys, metas = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = question_of(row)
            if not text:
                print("Warning: row without question text. Skipping.")
                continue
            key = pair_key_for(row)
            if key in seen:
                continue
            seen.add(key)
            prompts.append(text)
            keys.append(key)
            metas.append(extract_meta(row))
    return prompts, keys, metas


def main():
    random.seed(2001)
    args = parse_args()
    schema = verdict_model(args.rubric)
    crits = criteria_for(args.rubric)
    # only MOOVE scores are numeric; moove_pref's are per-criterion votes
    likert = [] if args.rubric == "moove_pref" else [c for c in crits if c in MOOVE]

    prompts, prompt_keys, prompt_metas = load_prompts(args.input)
    print(f"Loaded {len(prompts)} unique prompts.")

    # Each side generates at its own temperature (part of the cache key, so
    # the two never share answers); labels carry it so results stay legible
    # when both sides are the same model.
    t1 = args.temperature if args.temperature1 is None else args.temperature1
    t2 = args.temperature if args.temperature2 is None else args.temperature2
    per_side = args.temperature1 is not None or args.temperature2 is not None
    label1 = f"{args.model1}@T{t1}" if per_side else args.model1
    label2 = f"{args.model2}@T{t2}" if per_side else args.model2
    gen_prompts = prompts
    if args.system_prompt_file:
        sp = Path(args.system_prompt_file).read_text(encoding="utf-8")
        print(f"System prompt: {args.system_prompt_file} ({len(sp)} chars)")
        gen_prompts = [[{"role": "system", "content": sp},
                        {"role": "user", "content": q}] for q in prompts]
        label1, label2 = label1 + "+sys", label2 + "+sys"
    m1_answers = generate_answers_cached(
        args.model1, gen_prompts, argparse.Namespace(**{**vars(args), "temperature": t1}))
    m2_answers = generate_answers_cached(
        args.model2, gen_prompts, argparse.Namespace(**{**vars(args), "temperature": t2}))

    # Judge prompts with random swap to control position bias. A pair missing
    # an answer (failed API call, empty reply) is not judged: the judge would
    # read "None" or nothing as that model's answer and hand the other a free
    # win. Swaps are still drawn for every pair, so judged pairs keep theirs.
    swaps = [random.choice([True, False]) for _ in prompts]
    answered = [has_text(m1) and has_text(m2) for m1, m2 in zip(m1_answers, m2_answers)]
    judge_idx = [i for i, ok in enumerate(answered) if ok]
    judge_prompts = [build_judge_messages(prompts[i], m1_answers[i], m2_answers[i],
                                          swaps[i], args.rubric) for i in judge_idx]

    judge_util = args.judge_utilization or args.utilization
    judge_evals = [None] * len(prompts)
    verdicts = [None] * len(prompts)        # judge_pairs record, new path only
    # moove_pref's per-criterion votes are only decided in judge_pairs
    if args.scoring == "logprob" or args.bidirectional or args.rubric == "moove_pref":
        judging = argparse.Namespace(**{**vars(args), "utilization": judge_util})
        out = chk.judge_pairs([(i, prompts[i], m1_answers[i], m2_answers[i]) for i in judge_idx],
                              judging, max_model_len=args.judge_max_model_len) \
            if judge_idx else {}
        for i, v in out.items():
            verdicts[i] = v
            judge_evals[i] = v.get("eval_text")
    elif judge_prompts:
        judged = generate(
            args.judge, judge_prompts,
            max_new_tokens=args.max_new_tokens,
            tp=args.tp, utilization=judge_util,
            schema=schema,
            max_model_len=args.judge_max_model_len,
            temperature=args.temperature, seed=args.seed,
        )
        for i, e in zip(judge_idx, judged):
            judge_evals[i] = e

    ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / (
        f"{ts}_auto_moove_{Path(label1).name}_{Path(label2).name}"
        f"_{Path(args.judge).name}_{args.rubric}"
        f"{'_logprob' if args.scoring == 'logprob' else ''}"
        f"{'_bidir' if args.bidirectional else ''}.jsonl"
    )
    print(f"\n--- Processing and Saving Results to: {output_path} ---")
    records = []

    with open(output_path, "w", encoding="utf-8") as out:
        for i, (prompt, m1, m2, eval_text, swap) in enumerate(
                zip(prompts, m1_answers, m2_answers, judge_evals, swaps)):
            v = verdicts[i]
            if v is not None:                    # judge_pairs: already un-swapped
                j_vote, parsed = v["judge_vote"], v["parsed"]
                j_s1 = {c: v.get(f"j_L_{c}") for c in crits}
                j_s2 = {c: v.get(f"j_R_{c}") for c in crits}
            else:
                jd = extract_judge_data(eval_text, schema)
                parsed = jd is not None
                j_vote = {"Model 1": "1", "Model 2": "2", "Equal": "12"}.get(
                    unswap_winner(getattr(jd, "winner", None), swap))
                j_s1, j_s2 = unswap_scores(jd, swap, args.rubric)      # left, right

            rec = {
                "id": i,
                "pair_id": prompt_keys[i],
                "rubric": args.rubric,
                "prompt": prompt,
                "model_1": label1,
                "model_1_answer": m1,
                "model_2": label2,
                "model_2_answer": m2,
                "judge": args.judge,
                # None: the provider refused the requested temperature and
                # sampled at its own default, so that side was not greedy
                "judge_temperature": effective_temperature(args.judge, args.temperature),
                "model_1_temperature": effective_temperature(args.model1, t1),
                "model_2_temperature": effective_temperature(args.model2, t2),
                "judge_raw_eval": eval_text,
                "missing_answer": not answered[i],       # not judged, see above
                "parsed": parsed,
                "judge_vote": j_vote,            # "1" / "2" / "12" / None
                "scoring": args.scoring,
                "bidirectional": args.bidirectional,
                # one-order runs only; judge_pairs draws its own order
                "was_swapped": None if args.bidirectional else (
                    v.get("swap", swap) if v is not None else swap),
                # judge_pairs' full verdict (per-order votes, log-probs)
                "verdict": {k: x for k, x in v.items() if k not in ("eval_text",)}
                           if v is not None else None,
                # case metadata of the source row (source set, care level,
                # income tier, ...), so the results file can be grouped and
                # rendered without re-joining against the input
                "moove_meta": prompt_metas[i],
            }
            if args.rubric == "moove_pref":
                # vote code per criterion ("1" / "2" / "12" / None), un-swapped
                for c in crits:
                    rec[f"pref_{c}"] = v.get(f"j_pref_{c}") if v is not None else None
            else:
                for c in crits:
                    rec[f"m1_{c}"] = j_s1[c]
                    rec[f"m2_{c}"] = j_s2[c]

            records.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)

    df["m1_win"]  = df["judge_vote"] == "1"
    df["m2_win"]  = df["judge_vote"] == "2"
    df["tie"]     = df["judge_vote"] == "12"
    df["decided"] = df["judge_vote"].isin(["1", "2", "12"])

    total          = len(df)
    decided        = int(df["decided"].sum())
    missing        = int(df["missing_answer"].sum())
    # moove_pref has no overall verdict, so nothing is ever "decided"
    pref = args.rubric == "moove_pref"
    parse_failures = int(((~df["parsed"] if pref else ~df["decided"])
                          & ~df["missing_answer"]).sum())
    m1_wins_n      = int(df["m1_win"].sum())
    m2_wins_n      = int(df["m2_win"].sum())
    ties_n         = int(df["tie"].sum())

    def rate(count):
        return count / decided if decided else 0.0

    m1_rate, m2_rate, ties_rate = rate(m1_wins_n), rate(m2_wins_n), rate(ties_n)

    print("\n" + "=" * 80)
    print(" " * 28 + "AUTO MOOVE ANALYSIS")
    print(" " * 26 + "(Positional Bias Mitigated)")
    print("=" * 80)
    print(f"Total Evaluated: {total}")
    print(f"Final Parse Failures (after retries): {parse_failures}")
    if missing:
        print(f"Not judged, missing or empty answer: {missing}")
    print()

    print(f"Judge: {args.judge} | Rubric: {args.rubric}")
    if not HAVE_SCIPY:
        print("(scipy unavailable -- point estimates only, no CIs)")
    # Rows are one-per-pair (load_prompts dedups by pair_key), so a plain
    # bootstrap over decided rows is the right resampling unit.
    dec_rows = df[df["decided"]]
    if not pref:
        print("--- WIN RATES ---")
    _, m1_ci = compute_ci(dec_rows["m1_win"].astype(float)) if decided else (0.0, (None, None))
    _, m2_ci = compute_ci(dec_rows["m2_win"].astype(float)) if decided else (0.0, (None, None))
    _, tie_ci = compute_ci(dec_rows["tie"].astype(float)) if decided else (0.0, (None, None))
    if not pref:
        print(f"Model 1 ({label1}) Wins: {m1_wins_n} ({fmt_ci(m1_rate, m1_ci, pct=True, digits=1)})")
        print(f"Model 2 ({label2}) Wins: {m2_wins_n} ({fmt_ci(m2_rate, m2_ci, pct=True, digits=1)})")
        print(f"Ties: {ties_n} ({fmt_ci(ties_rate, tie_ci, pct=True, digits=1)})")

    if pref:
        print("\n--- PER-CRITERION PREFERENCE ---")
        print("*Win rate: Model 1's wins plus half the ties, over pairs with a parsed vote.*\n")
        print(f"{'Criterion':<25} | {'M1':>4} | {'Tie':>4} | {'M2':>4} | Model 1 win rate [95% CI]")
        print("-" * 86)
        for crit in crits:
            vote = df[f"pref_{crit}"].dropna()
            if vote.empty:
                print(f"{crit:<25} | N/A")
                continue
            w_pt, w_ci = compute_ci((vote == "1") + 0.5 * (vote == "12"))
            print(f"{crit:<25} | {(vote == '1').sum():>4} | {(vote == '12').sum():>4} | "
                  f"{(vote == '2').sum():>4} | {fmt_ci(w_pt, w_ci, pct=True, digits=1)}")
    elif likert:
        print("\n--- AVERAGE LIKERT SCORES & DIFFERENCES ---")
        print("*Positive difference means Model 1 scored higher.*\n")
        print(f"{'Criterion':<25} | {'Model 1':<7} | {'Model 2':<7} | {'Difference (M1 - M2) [95% CI]':<30}")
        print("-" * 86)
        for crit in likert:
            m1c, m2c = f"m1_{crit}", f"m2_{crit}"
            mask = df[m1c].notna() & df[m2c].notna()
            if mask.any():
                m1_pt, m2_pt = df.loc[mask, m1c].mean(), df.loc[mask, m2c].mean()
                # CI on the paired per-row difference, not on the two means
                d_pt_c, d_ci_c = compute_ci(df.loc[mask, m1c] - df.loc[mask, m2c])
                print(f"{crit:<25} | {m1_pt:.2f}    | {m2_pt:.2f}    | "
                      f"{fmt_ci(d_pt_c, d_ci_c, signed=True)}")
            else:
                print(f"{crit:<25} | N/A      | N/A      | N/A")
    else:
        # ponytail: checklist/preference values are strings; per-criterion rates
        # live in auto_moove_check.py, not duplicated here.
        print("\n(no numeric criteria for this rubric -- skipping Likert table)")

    dec = df[df["decided"]]
    if not pref:
        print("\n--- DERIVED METRICS ---")

    if decided:
        net_series = dec["m2_win"].astype(float) - dec["m1_win"].astype(float)
        net_pt, net_ci = compute_ci(net_series)
    else:
        net_pt, net_ci = 0.0, (None, None)
    if not pref:
        print(f"Net Win Rate (M2 - M1):       {fmt_ci(net_pt, net_ci, pct=True, signed=True, digits=1)}")

    if decided:
        awr_series = dec["m2_win"].astype(float) + 0.5 * dec["tie"].astype(float)
        awr_pt, awr_ci = compute_ci(awr_series)
    else:
        awr_pt, awr_ci = 0.0, (None, None)
    if not pref:
        print(f"Adjusted Win Rate (M2):       {fmt_ci(awr_pt, awr_ci, pct=True, digits=1)}")

    if likert:
        deltas = pd.DataFrame({c: df[f"m1_{c}"] - df[f"m2_{c}"] for c in likert})
        row_mean = deltas.mean(axis=1)
        if row_mean.notna().any():
            d_pt, d_ci = compute_ci(row_mean)
            print(f"Overall Likert Delta (M1-M2): {fmt_ci(d_pt, d_ci, signed=True, digits=3)}")
        else:
            print("Overall Likert Delta (M1-M2): N/A")

    print("=" * 80)
    print(f"Results successfully saved to {output_path}")

    summary_line = (
        f"{ts},auto_moove,{args.rubric},{Path(label1).name},{Path(label2).name},"
        f"{Path(args.judge).name},{total},{m1_wins_n},{m2_wins_n},{ties_n},"
        f"{m1_rate:.4f},{m2_rate:.4f},{ties_rate:.4f}\n"
    )
    write_header = not args.summary_file.exists()
    with open(args.summary_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,eval,rubric,model1,model2,judge,"
                    "n,m1_wins,m2_wins,ties,m1_rate,m2_rate,ties_rate\n")
        f.write(summary_line)
    print(f"Appended summary to {args.summary_file}")


if __name__ == "__main__":
    main()