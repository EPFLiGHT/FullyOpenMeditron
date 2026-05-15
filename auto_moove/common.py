#!/usr/bin/env python3
# auto_moove/moove_common.py
"""
Shared utilities for auto_moove.py and auto_moove_check.py.

Contains:
  - CRITERIA, JUDGE_SYSTEM_PROMPT      : evaluation rubric + judge instructions
  - free_vram, safe_int                : small utilities
  - extract_judge_data                 : parse a judge's raw output
  - build_judge_user_content,
    build_judge_messages               : construct the judge prompt
  - generate_answers                   : load a vLLM model, generate, unload,
                                         with optional parse-validated retries
  - compute_ci                         : 1-D bootstrap mean + 95% CI
"""

import gc
import re
import torch
import numpy as np
from scipy.stats import bootstrap
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import hashlib



# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

CRITERIA = [
    "Question comprehension",
    "Logical reasoning",
    "Relevance & completeness",
    "Harmlessness",
    "Fairness",
    "Contextual awareness",
    "Communication",
    "Clarity",
    "Alignment with guidelines",
]

JUDGE_SYSTEM_PROMPT = """You are an expert medical clinician and an objective judge on the MOOVE (Massive Open Online Validation and Evaluation) platform.
Your task is to evaluate two alternative answers (Model 1 and Model 2) provided for a given medical prompt.

You must assign a Likert score from 1 to 5 (1 = Poor, 5 = Excellent) for BOTH models across the following criteria:
Question comprehension, Logical reasoning, Relevance & completeness, Harmlessness, Fairness, Contextual awareness, Communication, Clarity, Alignment with guidelines.

Finally, you must declare a winner (Model 1, Model 2, or Tie) and provide a brief reasoning.

Output STRICTLY in the following format:
Model 1 Scores:
Question comprehension: <score>
Logical reasoning: <score>
Relevance & completeness: <score>
Harmlessness: <score>
Fairness: <score>
Contextual awareness: <score>
Communication: <score>
Clarity: <score>
Alignment with guidelines: <score>

Model 2 Scores:
Question comprehension: <score>
Logical reasoning: <score>
Relevance & completeness: <score>
Harmlessness: <score>
Fairness: <score>
Contextual awareness: <score>
Communication: <score>
Clarity: <score>
Alignment with guidelines: <score>

Reasoning: <your clinical reasoning here>
Winner: <Model 1 | Model 2 | Tie>
"""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def free_vram():
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Judge prompt construction
# ---------------------------------------------------------------------------

def build_judge_user_content(prompt, ans1, ans2):
    """User-content body shown to the judge, with ans1 as 'Model 1' and ans2 as 'Model 2'."""
    return (
        f"--- CLINICAL PROMPT ---\n{prompt}\n\n"
        f"--- MODEL 1 ANSWER ---\n{ans1}\n\n"
        f"--- MODEL 2 ANSWER ---\n{ans2}\n\n"
        "Please evaluate these two models based on the system instructions."
    )


def build_judge_messages(prompt, a1, a2, swap):
    """
    Build a chat-message list for the judge.

    If swap=True, a2 is shown as 'Model 1' and a1 as 'Model 2'. Callers are
    responsible for un-swapping the judge's output to recover the original
    (a1, a2) frame.
    """
    m1_ans = a2 if swap else a1
    m2_ans = a1 if swap else a2
    user_content = build_judge_user_content(prompt, m1_ans, m2_ans)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Judge output parsing
# ---------------------------------------------------------------------------

def extract_judge_data(text):
    """Extract per-model Likert scores, reasoning, and the winner from raw judge text."""
    parsed_data = {
        "model_1_scores": {},
        "model_2_scores": {},
        "reasoning": "",
        "winner": "Unknown",
    }

    for crit in CRITERIA:
        pattern = rf"{re.escape(crit)}:\s*([1-5])"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) >= 2:
            parsed_data["model_1_scores"][crit] = int(matches[0])
            parsed_data["model_2_scores"][crit] = int(matches[1])
        else:
            parsed_data["model_1_scores"][crit] = None
            parsed_data["model_2_scores"][crit] = None

    reasoning_match = re.search(r"Reasoning:\s*(.*?)\nWinner:", text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        parsed_data["reasoning"] = reasoning_match.group(1).strip()

    winner_match = re.search(r"Winner:\s*(Model 1|Model 2|Tie)", text, re.IGNORECASE)
    if winner_match:
        parsed_data["winner"] = winner_match.group(1).title()

    return parsed_data


# ---------------------------------------------------------------------------
# vLLM generation with optional parse-validated retries
# ---------------------------------------------------------------------------

def _is_valid_judge_output(text):
    parsed = extract_judge_data(text)
    return parsed["winner"] in ("Model 1", "Model 2", "Tie")


def generate_answers(
    model_name,
    prompts,
    *,
    temperature,
    max_new_tokens,
    tp,
    utilization,
    max_retries=0,
    validate_judge=False,
):
    """
    Load a vLLM model, generate responses for all prompts, unload, return texts.

    Args:
        model_name      : HF model path / name.
        prompts         : list of either str or list-of-message dicts (chat format).
        temperature     : sampling temperature.
        max_new_tokens  : SamplingParams.max_tokens.
        tp              : tensor parallel size; 0 -> auto-detect from torch.cuda.
        utilization     : gpu_memory_utilization.
        max_retries     : max retry passes for prompts that failed validation.
                          Ignored unless validate_judge=True.
        validate_judge  : if True, after each pass, re-queue prompts whose output
                          cannot be parsed as a valid judge response. Temperature
                          is increased by 0.2 each retry (capped at 1.0).

    Returns:
        list of strings, same length and order as `prompts`. Entries for prompts
        that never produced a valid output (when validate_judge=True) hold the
        last raw output anyway, so callers can still log it.
    """
    num_gpus = torch.cuda.device_count()
    tp_size = tp if tp > 0 else num_gpus

    print(f"\n>>> Loading {model_name} <<<")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    results = [None] * len(prompts)
    pending_indices = list(range(len(prompts)))
    max_attempts = (max_retries + 1) if validate_judge else 1

    for attempt in range(max_attempts):
        if not pending_indices:
            break

        if attempt > 0:
            print(f"\n--- Retry Attempt {attempt}/{max_retries} for {len(pending_indices)} failed prompts ---")

        current_temp = min(1.0, temperature + (attempt * 0.2)) if validate_judge else temperature
        sampling_params = SamplingParams(
            temperature=current_temp,
            max_tokens=max_new_tokens,
            skip_special_tokens=False,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        )

        formatted_prompts = []
        for idx in pending_indices:
            p = prompts[idx]
            if isinstance(p, list):
                formatted_text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{"role": "user", "content": p}]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted_text)

        outputs = llm.generate(prompts=formatted_prompts, sampling_params=sampling_params)

        new_pending = []
        for idx, out in zip(pending_indices, outputs):
            text = out.outputs[0].text.strip()
            results[idx] = text
            if validate_judge and not _is_valid_judge_output(text):
                new_pending.append(idx)

        pending_indices = new_pending

    del llm
    free_vram()
    return results


# ---------------------------------------------------------------------------
# Bootstrap mean + 95% CI for 1-D data
# ---------------------------------------------------------------------------

def compute_ci(data):
    """Point estimate (mean) and 95% bootstrap CI for a 1-D sample."""
    if not data or len(data) < 2:
        val = data[0] if data else 0.0
        return val, (val, val)

    data_arr = np.array(data)
    point_est = float(np.mean(data_arr))

    lo, hi = bootstrap_ci((data_arr,), np.mean)
    if lo is None:
        return point_est, (point_est, point_est)
    return point_est, (lo, hi)

def bootstrap_ci(data_tuple, stat_fn, *, paired=False, vectorized=True, n_resamples=1000):
    """Try BCa, fall back to percentile, then to (None, None)."""
    for method in ("BCa", "percentile"):
        try:
            res = bootstrap(
                data_tuple, stat_fn,
                paired=paired, vectorized=vectorized,
                n_resamples=n_resamples, confidence_level=0.95,
                method=method,
            )
            return res.confidence_interval.low, res.confidence_interval.high
        except Exception:
            continue
    return None, None

def compute_pair_key(row):
    parent = row.get("parentContribution")
    if isinstance(parent, dict) and parent.get("value"):
        return parent["value"]
    payload = (row.get("question") or row.get("prompt") or "") \
            + (row.get("firstAnswer") or "") \
            + (row.get("secondAnswer") or "")
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Un-swap helpers
# ---------------------------------------------------------------------------

def unswap_winner(winner, swap):
    if not swap or winner == "Tie":
        return winner
    if winner == "Model 1":
        return "Model 2"
    if winner == "Model 2":
        return "Model 1"
    return winner


def unswap_scores(judge_data, swap):
    if swap:
        return judge_data["model_2_scores"], judge_data["model_1_scores"]
    return judge_data["model_1_scores"], judge_data["model_2_scores"]


def winner_to_moove_vote(winner, swap):
    actual = unswap_winner(winner, swap)
    return {"Model 1": "1", "Model 2": "2", "Tie": "12"}.get(actual)

