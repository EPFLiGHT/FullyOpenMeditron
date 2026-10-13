#!/usr/bin/env python3
# auto_moove/auto_moove_check.py

"""
bash new_launch.sh auto_moove/auto_moove_check.py \
    --input moove/all_reviews_meditron.jsonl \
    --output auto_moove/judge_outputs/judged_moove.jsonl \
    --judge google/gemma-4-31B-it \
    --rubric moove

bash new_launch.sh auto_moove/auto_moove_check.py \
    --input moove/all_reviews_phc_train.jsonl \
    --output auto_moove/judge_outputs/judged_PHC_train_moove_qwen.jsonl \
    --judge Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --rubric moove

bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_checklist.jsonl \
    --judge google/gemma-4-31B-it \
    --rubric checklist

bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_moove.jsonl --out auto_moove/forest_plots/forest_chosen_meditronFO_only_moove_coverage.png 

bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_moove_full.jsonl --out auto_moove/forest_plots/forest_chosen_moove_human_full.png --predictors human

bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_preference.jsonl \
    --judge google/gemma-4-31B-it \
    --rubric preference

bash new_launch.sh auto_moove/auto_moove_check.py \
    --input moove/all_reviews_no_dup.jsonl \
    --output auto_moove/judge_outputs/judged_preference_full.jsonl \
    --judge google/gemma-4-31B-it \
    --rubric preference

bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_preference_full.jsonl --out auto_moove/forest_plots/forest_chosen_preference_coverage_full.png 


bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_moove_gemini.jsonl \
    --judge gemini:gemini-3.1-flash-lite \
    --rubric moove

bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_checklist_gemini_pro.jsonl \
    --judge gemini:gemini-3.1-pro \
    --rubric checklist

bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_checklist_gemini_flash.jsonl \
    --judge gemini:gemini-3.6-flash \
    --rubric checklist

    
bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_checklist_gpt5_4.jsonl \
    --judge openai:gpt-5.4 \
    --rubric checklist

bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_moove.jsonl --out auto_moove/forest_plots/forest_chosen_moove_human.png --predictors human
    
--judge gemini:gemini-3.1-flash-lite
--judge openai:gpt-5.4
--judge rcp:meta-llama/Llama-3.3-70B-Instruct
--judge google/gemma-4-31B-it






bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_checklist_gemini_flash.jsonl --out auto_moove/forest_plots/forest_chosen_checklist_gemini_flash.png 

bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_moove_full.jsonl --out auto_moove/forest_plots/forest_chosen_moove_coverage_full.png
bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_checklist.jsonl --out auto_moove/forest_plots/forest_chosen_checklist_coverage.png   

bash new_launch.sh auto_moove/auto_moove_forest_chosen_coverage.py --input auto_moove/judge_outputs/judged_checklist_gpt5_4.jsonl --out auto_moove/forest_plots/forest_chosen_checklist_coverage_gpt5_4.png   


bash new_launch.sh auto_moove/auto_moove_diag.py --input auto_moove/judge_outputs/judged_checklist.jsonl
bash new_launch.sh auto_moove/auto_moove_diag.py --input auto_moove/judge_outputs/judged_moove.jsonl

# a co-author's calibration protocol: winner-only verdict, the three candidate
# verdicts scored by log-prob, both answer orders averaged (local judge only).
# Tune the Equal region with --equal-threshold, or add --disagreement-to-equal.
bash new_launch.sh auto_moove/auto_moove_check.py \
    --input data/moove/all_reviews_no_dup_test_0_1.jsonl \
    --output auto_moove/judge_outputs/judged_winner_logprob_bidir.jsonl \
    --judge Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --rubric winner --scoring logprob --bidirectional


"""
import argparse, json, math, random, statistics
from collections import Counter
import pandas as pd

from auto_moove_prompt import MOOVE, CHECKLIST, criteria_for, verdict_model
from moove_meta import (answers_of, extract_meta, pair_key_for, question_of,
                        questions_match, review_vote_of)
from auto_moove_common import (
    PROVIDERS,
    effective_temperature,
    generate,
    score_judge_candidates,
    build_judge_messages,
    extract_judge_data,
    unswap_scores,
    winner_to_moove_vote,
)

# human rating columns are named after the original MOOVE labels
HUMAN_LABEL = {
    "clinical_soundness": "Clinical Soundness",
    "safety": "Safety",
    "contextual_fit": "Contextual Fit",
    "appropriate_confidence": "Appropriate Confidence",
    "clarity_completeness": "Clarity & Completeness",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--rubric", choices=["moove", "checklist", "both", "preference", "winner",
                                        "moove_pref"],
                   default="both", help="which criteria set the judge is asked to apply; "
                        "moove_pref: a preference per MOOVE criterion and no overall "
                        "verdict, scored by moove_criteria_agreement.py")
    # How the verdict is obtained. generate: the judge writes its JSON (any
    # rubric, local or API). logprob: the three candidate verdicts are scored
    # by log-probability instead -- a co-author's calibration protocol, winner
    # rubric and local vLLM only -- which makes the Equal region tunable.
    p.add_argument("--scoring", choices=["generate", "logprob"], default="generate")
    p.add_argument("--bidirectional", action="store_true",
                   help="judge every pair in both orders instead of one random order; "
                        "generate: an AB/BA disagreement becomes Equal; "
                        "logprob: the two orders' probabilities are averaged")
    p.add_argument("--disagreement-to-equal", action="store_true",
                   help="logprob + --bidirectional: predict Equal whenever the AB and "
                        "BA winners disagree, instead of trusting the average")
    p.add_argument("--equal-threshold", type=float, default=0.0,
                   help="logprob: predict Equal when P(Equal) is within this margin "
                        "of the more likely answer")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="judge context window (it sees the question and both answers)")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--utilization", type=float, default=0.85)
    # --samples K: every prompt judged K times at --temperature, each request
    # with its own seed; the verdict is the majority of the K (maj@K), and the
    # single-sample spread and pass@K are printed alongside.
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="judge decoding temperature; 0 is greedy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reasoning-effort", default=None,
                   choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                   help="API judges: reasoning effort; unset keeps the provider's default")
    p.add_argument("--batch", action="store_true",
                   help="openai: judge through the Batch API, about half price but "
                        "asynchronous; rerunning the same command collects a batch that "
                        "was interrupted while waiting (state in <output>.batch.json)")
    p.add_argument("--thinking", action="store_true",
                   help="local judges: reason before the verdict (chat template "
                        "enable_thinking); the JSON grammar applies once the reasoning "
                        "ends. Raise --max-new-tokens and --max-model-len to make room")
    args = p.parse_args()
    check_judge_args(p, args)
    return args


def check_judge_args(p, args):
    """Judge flag combinations the judging cannot honour. Also run by
    auto_moove_event_place.py on its own judge settings."""
    if args.scoring == "logprob":
        if args.rubric != "winner":
            p.error("--scoring logprob scores the winner-only verdict: add --rubric winner")
        if args.judge.partition(":")[0] in PROVIDERS:
            p.error("--scoring logprob needs prompt log-probs, i.e. a local vLLM judge")
    elif args.equal_threshold or args.disagreement_to_equal:
        p.error("--equal-threshold and --disagreement-to-equal apply to --scoring logprob "
                "(generate + --bidirectional already turns a disagreement into Equal)")
    if args.rubric == "moove_pref" and args.samples > 1:
        p.error("--rubric moove_pref has no overall verdict to take a majority of")
    if args.disagreement_to_equal and not args.bidirectional:
        p.error("--disagreement-to-equal needs --bidirectional")
    if args.samples > 1:
        if args.scoring != "generate" or args.judge.partition(":")[0] in PROVIDERS:
            p.error("--samples samples a generated verdict: --scoring generate "
                    "and a local vLLM judge")
        if args.temperature <= 0:
            p.error("--samples needs --temperature > 0; greedy samples are all identical")
    provider = args.judge.partition(":")[0]
    if args.reasoning_effort and provider not in PROVIDERS:
        p.error("--reasoning-effort applies to API judges")
    if args.batch and provider != "openai":
        p.error("--batch is implemented for openai: judges")
    if getattr(args, "thinking", False):        # auto_moove_event_place's settings lack it
        if args.scoring != "generate" or provider in PROVIDERS:
            p.error("--thinking reasons before a generated verdict: --scoring generate and a "
                    "local vLLM judge (API judges take --reasoning-effort)")


def crit_mean(scores):
    vals = [v for v in (scores or {}).values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def human_score_raw(row, side, crit):
    try:
        return float(row[f"{side} score: {HUMAN_LABEL[crit]}"])
    except (KeyError, TypeError, ValueError):
        return None


def _label(v):
    """Group label from a metadata value; the HF export's are lists."""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return v or ""


def load_pairs(path):
    """Unique answer pairs, each with every human review of it. Column names
    go through moove_meta, which reads both the MOOVE review export and the
    HF judge-calibration export."""
    pairs, degenerate = {}, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = question_of(row) or ""
            a1, a2 = answers_of(row)
            hv = review_vote_of(row)
            if not q.strip() or not a1.strip() or not a2.strip() or hv is None:
                continue
            # (from a co-author's PR) identical answers, or sides that were asked
            # different questions, leave no content difference to judge
            if a1.strip() == a2.strip() or not questions_match(row):
                degenerate += 1
                continue
            key = pair_key_for(row)
            if key not in pairs:
                meta = extract_meta(row)
                pairs[key] = {"question": q, "a1": a1, "a2": a2,
                              "prefix": _label(meta.get("source_set")),
                              "care_level": _label(meta.get("care_level")),
                              "humans": []}
            pairs[key]["humans"].append(row)
    if degenerate:
        print(f"Skipped {degenerate} reviews of pairs with identical answers "
              f"or mismatched questions.")
    return pairs


# ---- verdict per pair -------------------------------------------------------

def decide_criteria_prefs(by_swap, schema, crits):
    """moove_pref: a MOOVE vote code per criterion, un-swapped, in each order
    judged (j_pref_<c>_ab / _ba) and combined (j_pref_<c>) -- the one order,
    or with both, Equal when they disagree, as for the overall verdict. There
    is no overall verdict, so judge_vote stays None."""
    rec, parsed = {"judge_vote": None}, True
    for swap, text in by_swap.items():
        jd = extract_judge_data(text, schema)
        parsed &= jd is not None
        sfx = "_ba" if swap else "_ab"
        rec["eval_text" + ("_ba" if swap and len(by_swap) > 1 else "")] = text
        for c in crits:
            rec[f"j_pref_{c}{sfx}"] = (winner_to_moove_vote(getattr(jd.pref, c), swap)
                                       if jd is not None else None)
    for c in crits:
        votes = [rec[f"j_pref_{c}{'_ba' if s else '_ab'}"] for s in by_swap]
        rec[f"j_pref_{c}"] = (None if None in votes
                              else votes[0] if len(set(votes)) == 1 else "12")
    rec["parsed"] = parsed
    return rec


def decide_generated(by_swap, schema, rubric, crits):
    """Record fields for one pair from the judge's generated JSON, keyed by
    swap. With both orders judged, the criteria columns are the AB pass."""
    if rubric == "moove_pref":
        return decide_criteria_prefs(by_swap, schema, crits)
    def one(swap):
        jd = extract_judge_data(by_swap[swap], schema)
        return jd, winner_to_moove_vote(getattr(jd, "winner", None), swap)

    if len(by_swap) == 1:
        (swap, text), = by_swap.items()
        jd, vote = one(swap)
        rec = {"eval_text": text, "parsed": jd is not None, "judge_vote": vote}
    else:
        swap = False
        jd, winner_ab = one(False)
        jd_ba, winner_ba = one(True)
        # (from a co-author's PR) both orders must parse; when they disagree the
        # verdict followed position, not content, so it counts as Equal
        both = winner_ab is not None and winner_ba is not None
        rec = {"eval_text": by_swap[False], "eval_text_ba": by_swap[True],
               "parsed": jd is not None and jd_ba is not None,
               "winner_ab": winner_ab, "winner_ba": winner_ba,
               "orientation_consistent": winner_ab == winner_ba if both else None,
               "judge_vote": (winner_ab if winner_ab == winner_ba else "12") if both else None}
    left, right = unswap_scores(jd, swap, rubric)
    for c in crits:
        rec[f"j_L_{c}"] = left[c]
        rec[f"j_R_{c}"] = right[c]
    return rec


def majority_vote(votes):
    """Most frequent parsed vote; a tie at the top is Equal, the same call an
    AB/BA disagreement gets."""
    counts = Counter(v for v in votes if v is not None)
    if not counts:
        return None
    top = max(counts.values())
    tied = [v for v, c in counts.items() if c == top]
    return tied[0] if len(tied) == 1 else "12"


def decide_sampled(by_swap, schema, rubric, crits, samples):
    """decide_generated once per sample -- sample s of each order judged
    together -- and the majority of their votes. Texts and criteria columns
    are sample 0's."""
    per = [decide_generated({swap: texts[s] for swap, texts in by_swap.items()},
                            schema, rubric, crits)
           for s in range(samples)]
    votes = [r["judge_vote"] for r in per]
    return dict(per[0], parsed=all(r["parsed"] for r in per), judge_votes=votes,
                orientation_consistent_samples=[r.get("orientation_consistent") for r in per],
                judge_vote=majority_vote(votes))


LABELS = ("Model 1", "Model 2", "Equal")


def physical_probabilities(scores, swap):
    """Softmax over the three candidate scores, un-swapped to left / right."""
    top = max(scores[l] for l in LABELS)
    w = {l: math.exp(scores[l] - top) for l in LABELS}
    z = sum(w.values())
    first, second = w["Model 1"] / z, w["Model 2"] / z
    return {"left": second if swap else first,
            "right": first if swap else second,
            "equal": w["Equal"] / z}


def three_way_vote(probs, equal_threshold=0.0):
    """MOOVE vote code. Equal wins within equal_threshold of the more likely
    answer (and on exact ties); left wins a left/right tie."""
    if probs["equal"] >= max(probs["left"], probs["right"]) - equal_threshold:
        return "12"
    return "1" if probs["left"] >= probs["right"] else "2"


def _score_fields(scores, probs, sfx=""):
    return {f"score_model_1{sfx}": scores["Model 1"],
            f"score_model_2{sfx}": scores["Model 2"],
            f"score_equal{sfx}": scores["Equal"],
            f"prob_left{sfx}": probs["left"],
            f"prob_right{sfx}": probs["right"],
            f"prob_equal{sfx}": probs["equal"]}


def decide_logprob(by_swap, args):
    """Record fields for one pair from its candidate scores, keyed by swap.
    Field names follow a co-author's validation script."""
    if not args.bidirectional:
        (swap, scores), = by_swap.items()
        probs = physical_probabilities(scores, swap)
        return {"parsed": True, "swap": swap, "equal_threshold": args.equal_threshold,
                **_score_fields(scores, probs),
                "equal_margin": probs["equal"] - max(probs["left"], probs["right"]),
                "judge_vote": three_way_vote(probs, args.equal_threshold)}

    ab = physical_probabilities(by_swap[False], False)
    ba = physical_probabilities(by_swap[True], True)
    mean = {side: (ab[side] + ba[side]) / 2 for side in ab}
    winner_ab, winner_ba = three_way_vote(ab), three_way_vote(ba)   # raw, threshold 0
    averaged = three_way_vote(mean, args.equal_threshold)
    disagree = winner_ab != winner_ba
    return {"parsed": True, "equal_threshold": args.equal_threshold,
            "disagreement_to_equal": args.disagreement_to_equal,
            **_score_fields(by_swap[False], ab, "_ab"),
            **_score_fields(by_swap[True], ba, "_ba"),
            "prob_left_mean": mean["left"], "prob_right_mean": mean["right"],
            "prob_equal_mean": mean["equal"],
            "equal_margin": mean["equal"] - max(mean["left"], mean["right"]),
            "model_gap": abs(mean["left"] - mean["right"]),
            "winner_ab": winner_ab, "winner_ba": winner_ba,
            "orientation_consistent": not disagree,
            "averaged_judge_vote": averaged,
            "judge_vote": "12" if args.disagreement_to_equal and disagree else averaged}


# ---- agreement beyond raw accuracy (from a co-author's PR) --------------------------

VOTES = ("1", "2", "12")


def macro_f1(human, judge):
    """Mean per-class F1 over Model 1 / Model 2 / Equal."""
    f1 = []
    for v in VOTES:
        tp = sum(h == v and j == v for h, j in zip(human, judge))
        fp = sum(h != v and j == v for h, j in zip(human, judge))
        fn = sum(h == v and j != v for h, j in zip(human, judge))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1) / len(f1)


def cohen_kappa(human, judge):
    """Three-way agreement corrected for what the two label distributions
    would reach by chance."""
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judge)) / n
    expected = sum(human.count(v) * judge.count(v) for v in VOTES) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def judge_pairs(items, args, *, max_model_len, batch_state=None):
    """Verdict record fields per pair key. items: (key, question, left answer,
    right answer). args carries parse_args' judge settings: judge, rubric,
    scoring, bidirectional, samples, temperature, seed, reasoning_effort,
    max_new_tokens, tp, utilization and the logprob flags.

    One judge prompt per pair and order. A single order is drawn at random
    per pair, in items order, to control position bias -- the same draws as
    before, so earlier outputs reproduce; --bidirectional judges both orders
    instead."""
    schema = verdict_model(args.rubric)
    crits = criteria_for(args.rubric)
    prompts, slots = [], []
    for k, q, a1, a2 in items:
        orders = (False, True) if args.bidirectional else (random.choice([True, False]),)
        for swap in orders:
            prompts.append(build_judge_messages(q, a1, a2, swap, args.rubric))
            slots.append((k, swap))

    if args.scoring == "logprob":
        outs = score_judge_candidates(args.judge, prompts, tp=args.tp,
                                      utilization=args.utilization,
                                      max_model_len=max_model_len)
    else:
        outs = generate(args.judge, prompts,
                        max_new_tokens=args.max_new_tokens,
                        tp=args.tp, utilization=args.utilization,
                        schema=schema, max_model_len=max_model_len,
                        temperature=args.temperature, seed=args.seed,
                        samples=args.samples, reasoning_effort=args.reasoning_effort,
                        batch_state=batch_state, thinking=getattr(args, "thinking", False))
    judged = {k: {} for k, *_ in items}             # pair -> {swap: judge output}
    for (k, swap), o in zip(slots, outs):
        judged[k][swap] = o

    def decide(by_swap):
        if args.scoring == "logprob":
            return decide_logprob(by_swap, args)
        if args.samples > 1:
            return decide_sampled(by_swap, schema, args.rubric, crits, args.samples)
        return decide_generated(by_swap, schema, args.rubric, crits)
    return {k: decide(by_swap) for k, by_swap in judged.items()}


def main():
    random.seed(2001)
    args = parse_args()

    pairs = load_pairs(args.input)
    keys = list(pairs.keys())
    if not keys:
        raise SystemExit(f"No usable pair in {args.input}")
    print(f"Loaded {len(keys)} unique pairs, "
          f"{sum(len(p['humans']) for p in pairs.values())} human evaluations.")

    verdicts = judge_pairs([(k, pairs[k]["question"], pairs[k]["a1"], pairs[k]["a2"])
                            for k in keys], args, max_model_len=args.max_model_len,
                           batch_state=f"{args.output}.batch.json" if args.batch else None)

    records = []
    with open(args.output, "w", encoding="utf-8") as out:
        for k in keys:
            verdict = verdicts[k]
            base = {"pair_key": k, "rubric": args.rubric, "scoring": args.scoring,
                    "bidirectional": args.bidirectional,
                    # None: the provider refused 0.0 and sampled at its default,
                    # or logprob scoring, which samples nothing
                    "judge_temperature": (effective_temperature(args.judge, args.temperature)
                                          if args.scoring == "generate" else None),
                    "samples": args.samples,
                    "judge_reasoning_effort": args.reasoning_effort,
                    "judge_thinking": args.thinking,
                    "prefix": pairs[k]["prefix"], "care_level": pairs[k]["care_level"],
                    **verdict}
            judge_L = crit_mean({c: base.get(f"j_L_{c}") for c in MOOVE})
            judge_R = crit_mean({c: base.get(f"j_R_{c}") for c in MOOVE})

            for hrow in pairs[k]["humans"]:
                rec = dict(base, human_vote=review_vote_of(hrow))
                h_left  = {c: human_score_raw(hrow, "Left", c)  for c in MOOVE}
                h_right = {c: human_score_raw(hrow, "Right", c) for c in MOOVE}
                for c in MOOVE:
                    rec[f"h_L_{c}"] = h_left[c]
                    rec[f"h_R_{c}"] = h_right[c]
                rec["judge_L"] = judge_L
                rec["judge_R"] = judge_R
                rec["human_L"] = crit_mean(h_left)
                rec["human_R"] = crit_mean(h_right)
                rec["len_L"] = len(pairs[k]["a1"].split())
                rec["len_R"] = len(pairs[k]["a2"].split())
                records.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    if args.rubric == "moove_pref":
        # no overall verdict; the per-criterion agreement needs the physicians'
        # Likert scores, which moove_criteria_agreement.py joins back in
        pk = df.drop_duplicates("pair_key")
        print(f"Parsed {pk.parsed.mean()*100:.1f}% of {len(pk)} pairs.")
        print(pd.DataFrame({c: pk[f"j_pref_{c}"].value_counts() for c in MOOVE})
                .fillna(0).astype(int).to_string())
        print(f"Results written to {args.output}")
        return
    ok = df[df.judge_vote.notna()].copy()
    print(f"Parsed {df.parsed.mean()*100:.1f}% of judge outputs "
          f"({(~df.parsed).sum()} of {len(df)} rows, "
          f"{df.loc[~df.parsed, 'pair_key'].nunique()} pairs, excluded from agreement).")

    ok["decisive"] = ok["human_vote"].isin(["1", "2"])
    ok["agree"]    = ok["human_vote"] == ok["judge_vote"]

    def agreement(g):
        return pd.Series({
            "bin%":  100 * g.loc[g.decisive, "agree"].mean(),
            "bin_n": int(g.decisive.sum()),
            "3way%": 100 * g["agree"].mean(),
            "all_n": len(g),
            "h_tie": int((g.human_vote == "12").sum()),
            "j_tie": int((g.judge_vote == "12").sum()),
        })

    cols = ["decisive", "agree", "human_vote", "judge_vote"]
    for by in ("prefix", "care_level"):
        print(f"\n=== per-{by} agreement ===")
        tab = ok.groupby(by)[cols].apply(agreement)
        tab["parse%"] = 100 * df.groupby(by).parsed.mean()   # denominator you dropped from
        print(tab.to_string(float_format=lambda x: f"{x:.1f}"))

    if args.rubric in ("moove", "both"):
        print("\n=== mean MOOVE scores: judge vs human (-2..2) ===")
        print(ok.groupby("prefix")[["judge_L", "human_L", "judge_R", "human_R"]]
                .mean().to_string(float_format=lambda x: f"{x:.2f}"))

    print(f"\nBinary {ok.loc[ok.decisive,'agree'].mean()*100:.1f}% | "
          f"Three-way {ok['agree'].mean()*100:.1f}% "
          f"(n={len(ok)}, {len(df)-len(ok)} unparsed excluded)")
    if len(ok):
        # three-way accuracy rewards a judge that always says what humans
        # mostly say; these two do not
        hv, jv = ok["human_vote"].tolist(), ok["judge_vote"].tolist()
        print(f"Macro-F1 (three-way) {macro_f1(hv, jv)*100:.1f}% | "
              f"Cohen's kappa (three-way) {cohen_kappa(hv, jv):.3f}")
    if args.bidirectional:
        # per pair, not per review: each pair was judged once in each order
        cons = df.drop_duplicates("pair_key")["orientation_consistent"].dropna().astype(bool)
        print(f"AB/BA disagreement: {(1 - cons.mean())*100:.1f}% of {len(cons)} pairs")
        if args.scoring == "logprob":
            print(f"AB only: Three-way {(ok.human_vote == ok.winner_ab).mean()*100:.1f}% | "
                  f"BA only: Three-way {(ok.human_vote == ok.winner_ba).mean()*100:.1f}% "
                  f"(raw, threshold 0)")
    if args.samples > 1 and len(ok):
        # the verdicts above are maj@K; this is what one sample reaches on its
        # own, and what the best of the K would (pass@K, an upper bound only)
        hv = ok["human_vote"].tolist()
        per_sample = [list(v) for v in zip(*ok["judge_votes"])]
        three = [100 * sum(h == j for h, j in zip(hv, js)) / len(hv) for js in per_sample]
        f1 = [100 * macro_f1(hv, js) for js in per_sample]
        pass_k = 100 * statistics.mean(h in vs for h, vs in zip(hv, ok["judge_votes"]))
        same = 100 * statistics.mean(len(set(vs)) == 1 for vs in ok["judge_votes"])
        print(f"\n=== {args.samples} samples at temperature {args.temperature} ===")
        print(f"One sample: three-way {statistics.mean(three):.1f} ± "
              f"{statistics.stdev(three):.1f}% (range {min(three):.1f}-{max(three):.1f}) | "
              f"macro-F1 {statistics.mean(f1):.1f} ± {statistics.stdev(f1):.1f}%")
        print(f"maj@{args.samples}: three-way {ok['agree'].mean()*100:.1f}% | "
              f"pass@{args.samples}: {pass_k:.1f}%")
        print(f"All {args.samples} samples give the same verdict on {same:.1f}% of pairs")
        if args.bidirectional:
            cons = [c for cs in ok["orientation_consistent_samples"] for c in cs if c is not None]
            print(f"AB/BA disagreement over all samples: "
                  f"{100 * (1 - statistics.mean(cons)):.1f}%")
    if len(ok):
        print("\n=== three-way confusion (rows: human, cols: judge) ===")
        print(pd.crosstab(ok["human_vote"], ok["judge_vote"],
                          rownames=["Human"], colnames=["Judge"], margins=True))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()