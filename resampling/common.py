#!/usr/bin/env python3
# resampling/common.py
"""
Shared machinery for resample_answers.py.

Three things live here:

  * sample_batch      -- llm.generate for n samples per prompt that never
                         returns a truncated completion. A sample counts only
                         if it stopped on its own and reached the final
                         channel; prompts short of their quota are resampled
                         with a bigger token budget.
  * Checkpoint        -- append-only resume state, one record per resolved row,
                         so a walltime kill loses at most one chunk.
  * row/schema helpers -- one Arrow-safe schema per output file.

The MCQ contract (MCQ_SUFFIX, is_mcq, chosen_letter) is imported from
truncation/regenerate_truncated.py rather than copied: the generator and the
gate must not be able to disagree about what a row is, which is how the two
bugs in truncation/README.md happened.
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truncation.regenerate_truncated import (      # noqa: E402
    FINAL, MCQ_SUFFIX, chosen_letter, final_message, is_mcq, unusable_question,
)

__all__ = ["FINAL", "MCQ_SUFFIX", "chosen_letter", "final_message", "is_mcq", "normalize_rows",
           "option_letters", "row_kind", "question_of", "unusable_question",
           "letterize_options", "prepare_row",
           "HARMONY_SYSTEM", "is_complete", "clean_answer", "build_prompt",
           "load_llm", "sample_batch", "Checkpoint", "read_jsonl", "fill_defaults",
           "add_sampling_args", "add_io_args"]

HARMONY_SYSTEM = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n"
    "Reasoning: {reasoning}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)


# ---------------------------------------------------------------------------
# What counts as a usable sample
# ---------------------------------------------------------------------------

def clean_answer(text):
    """The answer the model actually committed to, with reasoning stripped.

    gpt-oss puts it in the final channel; Qwen-style models wrap reasoning in
    <think>...</think>. Anything else is returned as-is.
    """
    text = final_message(text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def is_complete(text, finish_reason):
    """True only for a sample that stopped on its own with a real answer.

    finish_reason == "length" means the token cap cut it off. Keeping those as
    finished turns is what taught MeditronFO to stop mid-sentence
    (truncation/README.md), so they are never usable here.
    """
    if finish_reason != "stop":
        return False
    if "<|channel|>" in text and FINAL not in text:
        return False                      # stopped inside the analysis channel
    if "<think>" in text and "</think>" not in text:
        return False                      # stopped inside a think block
    return bool(clean_answer(text))


# ---------------------------------------------------------------------------
# What kind of row is this
# ---------------------------------------------------------------------------

# meditron_4_synthetic_qa_v3 writes its options three ways, and a row whose
# options are only unrecognised looks identical to a row that has none.
OPTION_LINE = re.compile(r"^\s*(?:\*\*)?([A-J])\s*[).\-:]", re.M)   # "A) x" / "**A.** x"
OPTION_JSON = re.compile(r'"option\d+"\s*:\s*"\s*\(?([A-J])[).\.]', re.I)  # {"option1": "A. x"}
OPTION_INLINE = re.compile(r"(?<![A-Za-z0-9])([A-J])[).]\s+\S")        # "... A) x B) y ..."


def option_letters(question):
    """The option letters the question actually offers, or None if it offers none.

    A vote or an answer letter that is not on the menu is a parsing artefact,
    not an opinion. The inline form needs three letters before it counts: two
    would fire on ordinary prose ("vitamin D. The next step...").
    """
    for pat, need in ((OPTION_LINE, 2), (OPTION_JSON, 2), (OPTION_INLINE, 3)):
        found = set(pat.findall(question))
        if len(found) >= need:
            return found
    return None


# {"option1": "Steroids", ...}: the options are there but carry no letters, so
# the label ("C" = the third) is positional and the model has nothing to name.
# Rewriting them as a lettered list makes the row verifiable and gets a raw
# JSON blob out of the prompt. Content is untouched; only the formatting moves.
OPTIONS_JSON = re.compile(r"\n?\s*Options\s*:\s*(\{.*?\})\s*$", re.S | re.I)
OPTION_KEY = re.compile(r"^option\s*(\d+)$", re.I)
# some values carry their own letter ("A. Hydrochlorothiazide"): strip it, or
# the rewrite reads "A. A. Hydrochlorothiazide"
VALUE_LETTER = re.compile(r"^\(?([A-J])\s*[).\.]\s+")


def letterize_options(question):
    """(question, True) with JSON options rewritten as "A. x" lines, else unchanged."""
    m = OPTIONS_JSON.search(question)
    if not m:
        return question, False
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return question, False
    items = []
    for key, value in obj.items():
        km = OPTION_KEY.match(str(key).strip())
        if not km or not isinstance(value, str) or not value.strip():
            return question, False              # not the shape we understand
        text = value.strip()
        vm = VALUE_LETTER.match(text)
        n = int(km.group(1))
        if vm:
            if vm.group(1) != chr(64 + n):
                return question, False          # its own letters disagree with its order
            text = text[vm.end():].strip()
        if not text:
            return question, False
        items.append((n, text))
    if not 2 <= len(items) <= 10:
        return question, False
    items.sort()
    if [i for i, _ in items] != list(range(1, len(items) + 1)):
        return question, False                  # gaps would shift every letter
    listed = "\n".join(f"{chr(64 + n)}. {v}" for n, v in items)
    return question[:m.start()].rstrip() + "\n" + listed, True


def question_of(row):
    for turn in row.get("conversations", []):
        if turn.get("from") in ("human", "user"):
            return turn.get("value", "").strip()
    return ""


def prepare_row(row):
    """Normalisations applied before a row is classified or prompted."""
    conv = row.get("conversations", [])
    for turn in conv:
        if turn.get("from") in ("human", "user"):
            new_q, changed = letterize_options(turn.get("value", ""))
            if changed:
                turn["value"] = new_q
                row["options_letterized"] = True
            break
    row.setdefault("options_letterized", False)
    return row


def row_kind(row):
    """"mcq" (verifiable), "free" (no verifier), or a reason to drop the row.

    The third case is not hypothetical. 11.1% of the MCQ rows in
    meditron_4_synthetic_qa_v3 ask "which of the following" and then list no
    options: the generator wrote them into the answer only. A model that never
    sees the options cannot name the labelled letter, so such a row is neither
    verifiable nor answerable as written -- it would be rejected wholesale, or
    worse, pass by coincidence one time in five and carry a label that means
    nothing. guidelines_qa_v3_full adds 2.0% whose question is an elided stub
    ("... A)... B)..."), which unusable_question() already recognises.
    """
    q = question_of(row)
    if not q:
        return "drop:no_question"
    if unusable_question(q):
        return "drop:unusable_question"
    if is_mcq(row):
        return "mcq" if option_letters(q) else "drop:no_options"
    return "free"


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, question, *, mcq, harmony, reasoning="low", date="",
                 thinking=False):
    """The question as the answering model sees it.

    MCQ rows carry the answer-format contract; free-form rows must not, or the
    model is told to end an open-ended answer on an option letter.
    """
    messages = []
    if harmony:
        messages.append({"role": "system",
                         "content": HARMONY_SYSTEM.format(reasoning=reasoning, date=date)})
    messages.append({"role": "user",
                     "content": question.strip() + (MCQ_SUFFIX if mcq else "")})
    # Qwen3-style templates read enable_thinking; templates that do not use it
    # ignore it. Off by default: the MeditronFO corpus trains on answers without
    # reasoning traces, and thinking multiplies the decode cost.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)


def load_llm(args):
    """vLLM with the stop tokens the answering model needs. Imports live here
    so --selfcheck runs on a login node, where vllm is not installed."""
    import torch
    from vllm import LLM

    kwargs = {}
    if getattr(args, "max_model_len", 0):
        kwargs["max_model_len"] = args.max_model_len
    llm = LLM(model=args.model,
              tensor_parallel_size=args.tp or torch.cuda.device_count(),
              gpu_memory_utilization=args.utilization,
              trust_remote_code=True, **kwargs)
    tokenizer = llm.get_tokenizer()

    stop_ids = [tokenizer.eos_token_id]
    for tok in ("<end_of_turn>", "<|assistant_end|>"):      # gemma / apertus
        if tok in tokenizer.vocab:
            stop_ids.append(tokenizer.vocab[tok])
    return llm, tokenizer, stop_ids


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_batch(llm, prompts, *, n, need, temperature, max_new_tokens, stop_token_ids,
                 seed=None, length_retries=2, grow=2.0, log=print):
    """n samples per prompt, keeping only the complete ones.

    Returns one list of cleaned answers per prompt, length 0..n.

    A prompt that comes back with fewer than `need` complete samples is
    resampled -- only for its shortfall, and with the token budget multiplied
    by `grow` each time, since the reason to miss the quota is almost always
    the cap. This is the truncation retry; it is deliberately separate from the
    pass@k waves, which resample answers that are *wrong* rather than *cut off*.
    """
    from vllm import SamplingParams

    kept = [[] for _ in prompts]
    pending = list(range(len(prompts)))
    budget = max_new_tokens
    for attempt in range(1, length_retries + 2):          # first pass + retries
        if not pending:
            break
        want = {i: n - len(kept[i]) for i in pending}
        # one request per distinct shortfall, so nobody oversamples
        for shortfall in sorted(set(want.values()), reverse=True):
            idx = [i for i in pending if want[i] == shortfall]
            # a retry must not reuse the seed: same prompt + same seed gives
            # back the same sample, which would inflate a maj@k vote
            sp_seed = None if seed is None else seed + 1000 * attempt + shortfall
            params = SamplingParams(n=shortfall, temperature=temperature,
                                    max_tokens=budget, seed=sp_seed,
                                    skip_special_tokens=False,
                                    stop_token_ids=stop_token_ids or None)
            outs = llm.generate(prompts=[prompts[i] for i in idx], sampling_params=params)
            for i, out in zip(idx, outs):
                for o in out.outputs:
                    if is_complete(o.text, o.finish_reason):
                        kept[i].append(clean_answer(o.text))
        short = [i for i in pending if len(kept[i]) < need]
        log(f"    sampling pass {attempt} (max_tokens={budget}): "
            f"{len(pending) - len(short)}/{len(pending)} prompts reached {need} complete sample(s)")
        pending = short
        budget = int(budget * grow)
    return kept


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

class Checkpoint:
    """Append-only {key: record} log. Delete the file to start over."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = {}
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as f:              # a killed run can leave a
                f.seek(-1, 2)                             # line with no newline;
                torn = f.read(1) != b"\n"                 # appending would merge records
            if torn:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write("\n")
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.done[rec["key"]] = rec
        self.fh = open(self.path, "a", encoding="utf-8")

    def __contains__(self, key):
        return key in self.done

    def get(self, key):
        return self.done.get(key)

    def add(self, key, record):
        self.done[key] = record
        self.fh.write(json.dumps({"key": key, **record}, ensure_ascii=False) + "\n")

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fill_defaults(row, defaults):
    """Every row gets every key, with typed empties -- never None.

    A column that is null across the block Arrow infers from becomes type
    `null`, and the first real value later in the file dies with "Couldn't cast
    array of type string to null" (CLAUDE.md, and the trial that hit it in
    truncation/regenerate_truncated.py).
    """
    for k, v in defaults.items():
        if row.get(k) is None:
            row[k] = v
    return row


EMPTY = {str: "", bool: False, int: 0, float: 0.0, list: [], dict: {}}


def normalize_rows(rows, drop=()):
    """One Arrow-safe schema for the file: no nulls, no stale columns.

    Nulls arrive from the inputs themselves (synthetic_moove carries
    exact_match: null on every row), and a column that is null across the block
    Arrow infers from becomes type `null`, which kills the first real value
    later in the file. Columns that are null in *every* row hold nothing, so
    they are dropped rather than invented.
    """
    kinds, dropped = {}, set(drop)
    for row in rows:
        for k, v in row.items():
            if v is not None and k not in kinds:
                kinds[k] = type(v)
    for row in rows:
        for k in list(row):
            if k in dropped or k not in kinds:      # null in every row
                del row[k]
            elif row[k] is None:
                row[k] = EMPTY.get(kinds[k], "")
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


class Rate:
    """Throughput printer. A drop here caught both format bugs in
    truncation/README.md before any check did."""

    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()

    def step(self, k):
        self.done += k
        return (f"{self.done}/{self.total} rows, "
                f"{self.done / max(time.time() - self.t0, 1e-9):.1f} rows/s")


# ---------------------------------------------------------------------------
# Shared CLI
# ---------------------------------------------------------------------------

def add_sampling_args(p, *, temperature=0.7, max_new_tokens=16384):
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--temperature", type=float, default=temperature)
    p.add_argument("--max-new-tokens", type=int, default=max_new_tokens,
                   help="token budget of the first sampling pass")
    p.add_argument("--length-retries", type=int, default=2,
                   help="resamples with a grown budget before a row is given up on")
    p.add_argument("--grow", type=float, default=2.0, help="budget multiplier per retry")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=0, help="tensor parallel size (0 = all GPUs)")
    p.add_argument("--utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=0, help="0 = model default")
    p.add_argument("--reasoning", default="low", help="gpt-oss reasoning effort")
    p.add_argument("--thinking", default="off", choices=["off", "on"],
                   help="enable_thinking for Qwen3-style templates; reasoning is "
                        "stripped from the kept answer either way")


def add_shard_args(p):
    p.add_argument("--shard", default="",
                   help="'i/N' to process every Nth row, or 'auto' to read the shard "
                        "from the Slurm array task / srun rank. Each shard writes its "
                        "own output and checkpoint; merge_shards.py joins them.")


def shard_spec(arg):
    """(index, count) from --shard, or (0, 1) for an unsharded run.

    'auto' reads a Slurm job array first (one independent job per shard, which
    starts as nodes free up) and falls back to srun ranks (one allocation, all
    nodes at once).
    """
    import os
    if not arg:
        return 0, 1
    if arg == "auto":
        for idx, cnt in (("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT"),
                         ("SLURM_PROCID", "SLURM_NTASKS")):
            if os.environ.get(idx) and os.environ.get(cnt):
                return int(os.environ[idx]), int(os.environ[cnt])
        raise SystemExit("--shard auto: no Slurm array or srun rank in the environment")
    i, _, n = arg.partition("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise SystemExit(f"--shard {arg}: index must be in 0..{n - 1}")
    return i, n


def shard_suffix(shard, nshards):
    return "" if nshards == 1 else f".shard{shard}of{nshards}"


def add_io_args(p):
    p.add_argument("--input", action="append",
                   help="input .jsonl; repeat for several datasets")
    p.add_argument("--outdir", help="output directory")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="rows per vLLM call, i.e. checkpoint granularity")
    p.add_argument("--limit", type=int, default=None, help="first N rows per input")
    p.add_argument("--smoke", type=int, default=0,
                   help="smoke test: N rows per input, into <outdir>/smoke, with a summary")
    p.add_argument("--selfcheck", action="store_true",
                   help="run the offline unit checks and exit (no GPU needed)")
