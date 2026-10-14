#!/usr/bin/env python3
# distill_common.py
"""
Shared utilities for distill_with_retries.py and translate_with_retries.py
(and any future per-turn batch-generation scripts over a conversations[] corpus).

Contains:
  - load_llm                : vLLM model loader with auto tp and custom stop tokens
  - get_user_assistant_idx  : find first user + first following assistant turn
  - load_conversations_jsonl: load .jsonl, strip system turns, apply --limit/--resume
  - generate_complete       : llm.generate that resamples truncated completions
                              with a growing budget and drops those that never finish
  - run_retry_loop          : the core "collect pending → generate → validate → save"
                              loop shared by distill and translate
"""

import json
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llm(model_name, *, tp=0, utilization=0.90, extra_stop_tokens=None):
    """
    Load a vLLM model with auto tensor-parallel sizing.

    Args:
        model_name        : HF model path / name.
        tp                : tensor parallel size; 0 -> auto-detect from torch.cuda.
        utilization       : gpu_memory_utilization.
        extra_stop_tokens : list of literal strings to add as stop tokens IF present
                            in the tokenizer vocab (e.g. ["<end_of_turn>"]).

    Returns:
        (llm, tokenizer, stop_token_ids)
    """
    num_gpus = torch.cuda.device_count()
    tp_size = tp if tp > 0 else num_gpus

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    for tok in (extra_stop_tokens or []):
        if tok in tokenizer.vocab:
            stop_token_ids.append(tokenizer.vocab[tok])

    return llm, tokenizer, stop_token_ids


def make_sampling_params(temperature, max_new_tokens, stop_token_ids,
                         skip_special_tokens=False):
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        stop_token_ids=stop_token_ids or None,
    )


# ---------------------------------------------------------------------------
# Generation that never keeps a truncated completion
# ---------------------------------------------------------------------------

FINAL = "<|channel|>final<|message|>"


def final_message(text):
    """gpt-oss harmony: keep the final channel only (no-op for other models)."""
    text = text.strip()
    if FINAL in text:
        text = text.split(FINAL)[-1]
    return text.strip()


def is_complete(text, finish_reason):
    """
    A completion is usable only if it stopped on its own and, for harmony
    output, reached the final channel. A capped completion ("length") saved as
    a finished turn is what taught MeditronFO to stop mid-sentence
    (truncation/README.md).
    """
    if finish_reason != "stop":
        return False
    if "<|channel|>" in text and FINAL not in text:
        return False                    # stopped inside the analysis channel
    return bool(final_message(text))


def add_length_args(p, default_max_new_tokens=16384):
    p.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens,
                   help="token budget of the first attempt")
    p.add_argument("--length-retries", type=int, default=3,
                   help="attempts per prompt before an incomplete completion is dropped")
    p.add_argument("--grow", type=float, default=2.0,
                   help="token budget multiplier per retry")


def generate_complete(llm, prompts, *, temperature, max_new_tokens, stop_token_ids,
                      length_retries=3, grow=2.0):
    """
    llm.generate, resampling every prompt whose completion is not complete
    (see is_complete) with a bigger budget each retry.

    Returns one (text, finish_reason) per prompt. text is the final-channel
    message, or None if the prompt never completed; callers must drop those.
    """
    results = [(None, None)] * len(prompts)
    pending = list(range(len(prompts)))
    budget = max_new_tokens
    for attempt in range(1, length_retries + 1):
        if not pending:
            break
        params = make_sampling_params(temperature, budget, stop_token_ids)
        outs = llm.generate(prompts=[prompts[i] for i in pending], sampling_params=params)
        still = []
        for i, out in zip(pending, outs):
            o = out.outputs[0]
            if is_complete(o.text, o.finish_reason):
                results[i] = (final_message(o.text), o.finish_reason)
            else:
                results[i] = (None, o.finish_reason)
                still.append(i)
        print(f"--- Length attempt {attempt}/{length_retries} (max_tokens={budget}): "
              f"{len(pending) - len(still)}/{len(pending)} complete ---", flush=True)
        pending = still
        budget = int(budget * grow)
    if pending:
        print(f"--- {len(pending)}/{len(prompts)} prompts never completed; dropping them ---")
    return results


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------

def get_user_assistant_idx(conversations):
    """Find index of first user turn and the assistant turn that follows it."""
    user_idx, assistant_idx = None, None
    for i, turn in enumerate(conversations):
        role = turn.get("from")
        if user_idx is None and role in ("human", "user"):
            user_idx = i
        elif user_idx is not None and role in ("assistant", "gpt"):
            assistant_idx = i
            break
    return user_idx, assistant_idx


def load_conversations_jsonl(path, *, limit=None, strip_system=True):
    """Load a JSONL file of {conversations: [...]} rows."""
    ds = load_dataset("json", data_files=path, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [dict(x) for x in ds]
    if strip_system:
        for row in rows:
            if "conversations" in row:
                row["conversations"] = [t for t in row["conversations"]
                                        if t.get("from") != "system"]
    return rows


def save_jsonl(rows, path, *, drop_keys=()):
    drop = set(drop_keys)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if drop:
                row = {k: v for k, v in row.items() if k not in drop}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# The retry loop
# ---------------------------------------------------------------------------

def run_retry_loop(
    rows,
    *,
    llm,
    tokenizer,
    sampling_params,
    collect_jobs,
    build_messages,
    apply_output,
    max_tries,
    output_path,
    save_drop_keys=(),
    progress_label="Attempt",
):
    """
    Generic batch-generate-with-retries loop.

    The caller customizes behavior with three callbacks:

      collect_jobs(rows) -> list of opaque "job" objects
          Returns every unit of work that still needs generating this pass.
          A "job" is whatever the caller wants — typically a tuple identifying
          a row + turn + source text. Empty list = we're done.

      build_messages(job) -> list of chat-message dicts
          Convert a job into the message list passed to apply_chat_template.

      apply_output(job, generated_text, rows) -> bool
          Write the generated text back into `rows` and return True if the
          result was accepted (valid). If False, the job will be re-collected
          on the next pass (assuming collect_jobs still surfaces it).

    Saves `rows` to output_path after every attempt.
    """
    for attempt in range(1, max_tries + 1):
        jobs = collect_jobs(rows)
        if not jobs:
            print(f"--- All jobs complete. Ending early at attempt {attempt} ---")
            break

        print(f"--- {progress_label} {attempt}/{max_tries}: {len(jobs)} jobs ---")

        prompts = [
            tokenizer.apply_chat_template(
                build_messages(job), tokenize=False, add_generation_prompt=True
            )
            for job in jobs
        ]

        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

        n_ok = 0
        for job, out in zip(jobs, outputs):
            text = out.outputs[0].text.strip()
            if apply_output(job, text, rows):
                n_ok += 1

        print(f"    Accepted this pass: {n_ok}/{len(jobs)}")
        print(f"--- Saving to: {output_path} ---")
        save_jsonl(rows, output_path, drop_keys=save_drop_keys)