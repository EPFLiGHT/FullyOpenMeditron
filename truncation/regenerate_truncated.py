#!/usr/bin/env python3
# truncation/regenerate_truncated.py
"""
Regenerate the truncated assistant turns in the MeditronFO training sets with
gpt-oss-120b.

Straight sampling, no rejection sampling: the answer is not checked against
label_letter and rows are never dropped for being "wrong". The only reason a
row is sampled again is finish_reason == "length" -- i.e. it truncated again --
and each retry gets a bigger token budget.

Which turns get regenerated:
  * finish_reason == "length" on the row (recorded by
    distill_with_retries_v3_no_truncation.py), or
  * the answer was flagged by the LM judge in train_truncation.py's
    results/train/truncated.jsonl.
Everything else is passed through byte-for-byte.

A row is only overwritten when the fresh sample came back finish_reason ==
"stop" with a non-empty final channel. Otherwise the original is kept and
marked "regen_failed": true, so a later pass can drop the stragglers.

Truncated turns whose *question* is itself unusable are dropped instead of
regenerated -- there is nothing to regenerate from. This is almost entirely
guidelines_qa_v3_full, where the 10-vignettes-per-call generator leaked its own
analysis channel into the question of 98% of the truncated rows (against a 2.7%
baseline in that corpus). --keep-unusable regenerates them anyway.

Resumable. Generation runs in chunks and every finished turn is appended to a
checkpoint immediately, so a run killed by the walltime loses at most one
chunk. Relaunching with the same --outdir picks up where it stopped, which is
what makes this practical on the 1h30 debug partition: launch it repeatedly
until it reports nothing pending. Chunking is what makes that work at all --
vLLM returns a batch only when its slowest prompt finishes, so a single
30k-prompt call would checkpoint nothing until the very end.

The per-input <name>_fixed.jsonl files are written only once every turn is
resolved, so a partial run leaves no half-fixed corpus lying around.

usage:
bash new_launch.sh --partition debug --time 1:29:59 --requirements none \
    truncation/regenerate_truncated.py --model openai/gpt-oss-120b --outdir data/fixed

    --max-new-tokens 0   no cap (generate to the end of the context window)
    --limit 50           smoke test: regenerate at most 50 turns
    --chunk-size 512     turns per vLLM call, i.e. checkpoint granularity
    --retry-failed       re-attempt turns a previous run gave up on
    --gpu-log-interval   seconds between GPU utilisation samples (0 disables)

writes:
    <outdir>/<name>_fixed.jsonl   each input with its truncated turns redone
    <outdir>/regen_report.csv     per dataset: flagged / fixed / still truncated
    <outdir>/checkpoint.jsonl     resume state; delete it to start over
"""

import argparse, collections, csv, hashlib, json, re, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path

SYSTEM_MESSAGE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n"
    "Reasoning: {reasoning}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

# The four datasets in apertus_70b_meditronFO_fix_truncation.yaml. The moove
# entry is the *unfiltered* no_truncation file, a superset of the _filtered one
# the config points at: it still carries the 2186 finish_reason == "length"
# rows that filtering threw away, which this script recovers instead.
DEFAULT_INPUTS = [
    "data/curated_qa/gpt_oss_8retries_v3.jsonl",
    "data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl",
    "data/guidelines_qa/guidelines_qa_v3_full.jsonl",
    "data/synthetic_moove/synthetic_moove_v2_gpt_oss_no_truncation.jsonl",
]
DEFAULT_FLAGS = "truncation/results/train/truncated.jsonl"

ASSISTANT = {"assistant", "gpt"}
USER = {"user", "human"}

FINAL = "<|channel|>final<|message|>"

# distill_synthetic_qa_v3.py generated question AND answer together under a
# DEV_MESSAGE that specified exam format, so 93.6% of that corpus ends on an
# option letter. Regenerating from the bare question loses that contract and
# gpt-oss answers conversationally instead (0.6% stated a letter). Rows that
# carry a label_letter get the contract restored; free-form rows (the moove
# corpus, the unlabeled MedMCQA questions) must NOT get it -- only 0.2% of
# those state a letter and the regeneration already matched that.
MCQ_SUFFIX = ("\n\nWork through the options and explain your reasoning, then end your "
              "response with \"The answer is X.\" where X is the correct option letter.")

# Same patterns distill_with_retries_v3.py used, so a relabelled row is read
# back the way the rest of the corpus is.
ANSWER_PATTERNS = (re.compile(r"answer is \(?([A-J])\)?", re.I),
                   re.compile(r"(?i)Answer[^A-J]*(?::)*[^A-J]*(?:boxed)*[^A-J]*([A-J])\b"))


def is_mcq(row):
    """Must this row's answer end on an option letter?

    Not the same as "has a label_letter". meditron_4_synthetic_qa_v3 carries a
    label_letter on all 214,654 rows but an is_labeled flag that is False for
    17,507 of them: those are open-ended questions whose letter is a spurious
    artifact of extract_chosen_answer() run over a free-form answer. Truncation
    is concentrated in that open-ended block (17,375 of the 24,252 flagged
    rows), so reading the letter alone puts the MCQ contract on precisely the
    rows that must not have it. The explicit flag wins wherever it exists.
    """
    if "is_labeled" in row:
        return bool(row["is_labeled"])
    return bool(row.get("label_letter"))


def chosen_letter(text):
    """The option letter an answer settles on, or None."""
    for pat in ANSWER_PATTERNS:
        m = pat.findall(text)
        if m:
            return m[-1].upper()
    return None

# Generator-speak that leaked into a question instead of a clinical vignette:
# an elided opening, harmony channel markers, or the vignette-writing brief
# itself. Deliberately narrow -- a bare mention of "guideline" is legitimate.
UNUSABLE_Q = re.compile(
    r"^\s*\.\.\.|Provide answer|Ensure unique|options A-[A-E]|We'?ll (give|use)"
    r"|The prompt says|<\|channel\|>", re.I)
# Only degenerate stubs. The source corpora legitimately contain very short
# questions ("Question: Management thyroid ca"), which are answerable as-is.
MIN_QUESTION_CHARS = 12

# Written to every output row so the file has one Arrow schema throughout.
# Same-typed empties, not None, where the field is a bool or a count: a column
# that is sometimes null and sometimes bool infers as null and breaks later.
# Empty STRINGS, not None. A column that is null across the block Arrow infers
# from becomes type `null`, and the first real value later in the file dies
# with "Couldn't cast array of type string to null" -- which is how the second
# trial (job 3327881) failed after the first fix used None here.
REGEN_FIELD_DEFAULTS = {
    "regen": False,
    "regen_tries": 0,
    "regen_stated_letter": False,
    "label_letter_before_regen": "",
    "finish_reason": "",
}


def unusable_question(q):
    """A truncated turn whose question is broken has nothing to regenerate from."""
    return len(q.strip()) < MIN_QUESTION_CHARS or bool(UNUSABLE_Q.search(q))


def _h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def job_key(name, line_no, turn_idx):
    """Stable across runs: the inputs are read in the same order every time."""
    return f"{name}|{line_no}|{turn_idx}"


def load_checkpoint(path, retry_failed=False):
    """key -> result dict (or None for a turn a previous run gave up on)."""
    done = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # half-written last line from a killed run
            if r.get("failed"):
                if not retry_failed:
                    done[r["key"]] = None
            else:
                done[r["key"]] = {"answer": r["answer"],
                                  "finish_reason": r["finish_reason"],
                                  "tries": r["tries"]}
    return done


def start_gpu_logger(interval):
    """Sample nvidia-smi in the background so the log shows whether all four
    GPUs are actually working, not just allocated."""
    if interval <= 0:
        return
    q = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"]

    def loop():
        while True:
            try:
                out = subprocess.run(q, capture_output=True, text=True, timeout=30).stdout
                util, mem = [], []
                for row in out.strip().splitlines():
                    idx, u, used, total = [x.strip() for x in row.split(",")]
                    util.append(f"{idx}:{u}%")
                    mem.append(f"{idx}:{int(used)/1024:.0f}/{int(total)/1024:.0f}G")
                if util:
                    print(f"[gpu] util {' '.join(util)} | mem {' '.join(mem)}", flush=True)
            except Exception as e:                      # never kill the run over telemetry
                print(f"[gpu] sampling failed: {e}", flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


def load_flags(path):
    """dataset stem -> {sha1(answer)}, from train_truncation.py's audit.

    Keyed on the answer text rather than the recorded line number: that audit
    dedupes on (dataset, answer), so its line number is one occurrence of an
    answer that may repeat. Same matching write_filtered() uses.
    """
    flagged = collections.defaultdict(set)
    if not path:
        return flagged
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("truncated") is True:
                flagged[r["dataset"]].add(_h(r["answer"]))
    return flagged


def assistant_slots(conv):
    """(turn index, question, answer) per assistant turn; question = preceding user turn."""
    out, last_user = [], ""
    for i, m in enumerate(conv):
        role, content = m.get("from"), m.get("value") or ""
        if role in USER:
            last_user = content
        elif role in ASSISTANT and content.strip():
            out.append((i, last_user, content))
    return out


def scan(path, flagged, keep_unusable=False):
    """Truncated turns in one input.

    Returns (jobs, drop_lines, n_rows, n_turns): jobs are
    (line_no, turn_idx, question, labeled) to regenerate, drop_lines are rows
    whose question is unusable and are therefore left out of the output.
    `labeled` marks a row that must answer in MCQ format.
    """
    jobs, drop, n_rows, n_turns = [], set(), 0, 0
    # finish_reason is per row, not per turn; these corpora are single-turn.
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            by_length = row.get("finish_reason") == "length"
            for turn_idx, question, answer in assistant_slots(row.get("conversations") or []):
                n_turns += 1
                if not (by_length or _h(answer) in flagged):
                    continue
                if not keep_unusable and unusable_question(question):
                    drop.add(line_no)
                else:
                    jobs.append((line_no, turn_idx, question, is_mcq(row)))
    jobs = [j for j in jobs if j[0] not in drop]
    return jobs, drop, n_rows, n_turns


def build_prompt(tokenizer, question, gptoss, reasoning, labeled=False):
    messages = []
    if gptoss:
        messages.append({"role": "system", "content": SYSTEM_MESSAGE.format(
            reasoning=reasoning, date=datetime.now().strftime("%Y-%m-%d"))})
    content = question.strip() + (MCQ_SUFFIX if labeled else "")
    messages.append({"role": "user", "content": content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def final_message(text):
    """gpt-oss harmony: keep the final channel only."""
    text = text.strip()
    if FINAL in text:
        text = text.split(FINAL)[-1]
    return text.strip()


def accept(text, finish_reason):
    """A sample is usable only if it ran to a stop with a non-empty final channel."""
    return finish_reason == "stop" and bool(final_message(text))


def regenerate(jobs, args, ckpt_path, done):
    """Sample each unresolved job until it stops cleanly.

    Generates in chunks and appends every result to the checkpoint as the chunk
    returns, so a walltime kill costs at most one chunk. `done` is the state
    loaded from a previous run and is updated in place.
    """
    import torch
    from vllm import LLM, SamplingParams

    keys = [job_key(Path(path).stem, ln, ti) for ln, ti, path, _, _ in jobs]
    pending = [i for i in range(len(jobs)) if keys[i] not in done]
    if not pending:
        print("--- every turn already in the checkpoint; nothing to generate ---")
        return
    print(f"--- {len(done)} turns already done, {len(pending)} to generate ---")

    start_gpu_logger(args.gpu_log_interval)

    llm = LLM(model=args.model,
              tensor_parallel_size=args.tp if args.tp > 0 else torch.cuda.device_count(),
              gpu_memory_utilization=args.utilization,
              trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    gptoss = "gpt-oss" in args.model

    prompts = [build_prompt(tokenizer, q, gptoss, args.reasoning, lab)
               for _, _, _, q, lab in jobs]
    budget = args.max_new_tokens or None          # 0 -> no cap, use the context window

    if ckpt_path.exists() and ckpt_path.stat().st_size:
        with open(ckpt_path, "rb") as f:                # a killed run can leave a
            f.seek(-1, 2)                               # line with no newline;
            torn = f.read(1) != b"\n"                   # appending onto it would
        if torn:                                        # merge and lose two records
            with open(ckpt_path, "a", encoding="utf-8") as f:
                f.write("\n")
    ckpt = open(ckpt_path, "a", encoding="utf-8")
    try:
        for attempt in range(1, args.max_tries + 1):
            if not pending:
                break
            print(f"--- Attempt {attempt}/{args.max_tries}: {len(pending)} turns, "
                  f"max_tokens={budget or 'context limit'}, chunk={args.chunk_size} ---",
                  flush=True)
            params = SamplingParams(temperature=args.temperature, max_tokens=budget,
                                    skip_special_tokens=False,
                                    stop_token_ids=[tokenizer.eos_token_id])
            still, t0 = [], time.time()
            for c in range(0, len(pending), args.chunk_size):
                chunk = pending[c:c + args.chunk_size]
                outs = llm.generate(prompts=[prompts[i] for i in chunk],
                                    sampling_params=params)
                for i, out in zip(chunk, outs):
                    o = out.outputs[0]
                    if accept(o.text, o.finish_reason):
                        res = {"answer": final_message(o.text),
                               "finish_reason": o.finish_reason, "tries": attempt}
                        done[keys[i]] = res
                        ckpt.write(json.dumps({"key": keys[i], **res},
                                              ensure_ascii=False) + "\n")
                    else:
                        still.append(i)
                ckpt.flush()
                nd = c + len(chunk)
                rate = nd / max(time.time() - t0, 1e-9)
                print(f"    chunk {c // args.chunk_size + 1}: {nd}/{len(pending)} "
                      f"attempted, {len(still)} not yet clean, {rate:.1f} turns/s",
                      flush=True)
            pending = still
            if budget:
                budget = int(budget * args.grow)

        for i in pending:                          # gave up; record so a relaunch skips it
            done[keys[i]] = None
            ckpt.write(json.dumps({"key": keys[i], "failed": True}) + "\n")
        ckpt.flush()
    finally:
        ckpt.close()


def apply_fixes(path, jobs, results, dest, drop_lines=frozenset()):
    """Stream the input back out, substituting the regenerated answers."""
    fixes = collections.defaultdict(dict)
    for i, (line_no, turn_idx, _, _, _) in enumerate(jobs):
        fixes[line_no][turn_idx] = results.get(i)

    fixed = failed = dropped = 0
    with open(path, encoding="utf-8") as f, open(dest, "w", encoding="utf-8") as g:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            if line_no in drop_lines:
                dropped += 1
                continue
            if line_no not in fixes:
                g.write(line if line.endswith("\n") else line + "\n")
                continue
            row = json.loads(line)
            for turn_idx, res in fixes[line_no].items():
                if res is None:
                    row["regen_failed"] = True
                    failed += 1
                    continue
                row["conversations"][turn_idx]["value"] = res["answer"]
                # The corpus label is whatever the generator itself concluded
                # (extract_chosen_answer in distill_synthetic_qa_v3.py), so a
                # regenerated answer has to carry its own letter, not the old
                # one -- which on these rows came from a regex over truncated
                # text and mostly does not survive scrutiny.
                if is_mcq(row):
                    new = chosen_letter(res["answer"])
                    row["label_letter_before_regen"] = row["label_letter"]
                    row["label_letter"] = new
                    row["regen_stated_letter"] = new is not None
                row["regen"] = True
                row["regen_tries"] = res["tries"]
                row["finish_reason"] = res["finish_reason"]
                fixed += 1
            # Every row carries the bookkeeping keys, regenerated or not.
            # HuggingFace `datasets` infers an Arrow schema from the first
            # chunk of the file and hard-fails when a later chunk introduces
            # columns -- so writing these only on rewritten rows makes the
            # corpus unloadable by axolotl. Cost a debug run to find (job
            # 3327839); see truncation/normalize_schema.py, which repairs
            # files written before this.
            for k, v in REGEN_FIELD_DEFAULTS.items():
                row.setdefault(k, v)
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    return fixed, failed, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    p.add_argument("--flags", default=DEFAULT_FLAGS,
                   help="train_truncation.py's truncated.jsonl ('' to use finish_reason only)")
    p.add_argument("--outdir", default="data/fixed")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--max-new-tokens", type=int, default=16384,
                   help="0 = no cap, generate to the end of the context window")
    p.add_argument("--grow", type=float, default=1.5,
                   help="token budget multiplier per retry")
    p.add_argument("--max-tries", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--reasoning", default="low")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--utilization", type=float, default=0.90)
    p.add_argument("--limit", type=int, default=None,
                   help="regenerate at most this many turns (smoke test)")
    # Above vLLM's default max_num_seqs (256) so the scheduler always has
    # queued work: a chunk sized exactly to the batch drains to a trickle of
    # stragglers at every boundary, idling the GPUs.
    p.add_argument("--chunk-size", type=int, default=512,
                   help="turns per vLLM call; also the checkpoint granularity")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-attempt turns a previous run gave up on")
    p.add_argument("--gpu-log-interval", type=int, default=60,
                   help="seconds between GPU utilisation samples (0 disables)")
    p.add_argument("--keep-unusable", action="store_true",
                   help="regenerate truncated turns with a broken question instead of dropping them")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    flagged = load_flags(args.flags)
    print(f"Judge flags: { {k: len(v) for k, v in flagged.items()} }")

    # (line_no, turn_idx, question) per input -> one flat batch, one model load
    jobs, per_file, stats = [], [], []
    for path in args.input:
        name = Path(path).stem
        found, drop, n_rows, n_turns = scan(path, flagged[name], args.keep_unusable)
        start = len(jobs)
        jobs += [(ln, ti, path, q, lab) for ln, ti, q, lab in found]
        per_file.append((path, name, start, len(jobs), drop))
        stats.append({"dataset": name, "rows": n_rows, "turns": n_turns,
                      "flagged": len(found) + len(drop), "unusable_question": len(drop)})
        n_mcq = sum(1 for _, _, _, lab in found if lab)
        print(f"{name}: {len(found) + len(drop)} truncated of {n_turns} turns "
              f"({n_rows} rows) -> regenerate {len(found)} "
              f"({n_mcq} MCQ / {len(found) - n_mcq} free-form), drop {len(drop)}")

    if args.limit:
        jobs = jobs[:args.limit]
        keep = [k for k, (_, _, start, _, _) in enumerate(per_file) if start < len(jobs)]
        per_file = [(*per_file[k][:3], min(per_file[k][3], len(jobs)), per_file[k][4])
                    for k in keep]
        stats = [stats[k] for k in keep]          # keep the two lists in lockstep
        print(f"--- limit: regenerating the first {len(jobs)} turns only, "
              f"writing {len(per_file)} of {len(args.input)} outputs ---")

    if not jobs:
        print("Nothing to regenerate.")
        return

    ckpt_path = out / "checkpoint.jsonl"
    done = load_checkpoint(ckpt_path, args.retry_failed)
    print(f"\n--- {len(jobs)} turns to regenerate with {args.model} "
          f"(checkpoint: {len(done)} already resolved) ---")
    regenerate(jobs, args, ckpt_path, done)

    # Only write the corpora once every turn is resolved, so a run cut short by
    # the walltime never leaves a half-fixed dataset behind.
    keys = [job_key(Path(path).stem, ln, ti) for ln, ti, path, _, _ in jobs]
    unresolved = [k for k in keys if k not in done]
    if unresolved:
        print(f"\n>>> {len(unresolved)} of {len(jobs)} turns still unresolved. "
              f"No corpora written. Relaunch with the same --outdir to continue: "
              f"the checkpoint at {ckpt_path} holds the {len(done)} already done. <<<")
        return

    for (path, name, start, end, drop), st in zip(per_file, stats):
        sub = jobs[start:end]
        res = {i - start: done.get(keys[i]) for i in range(start, end)}
        dest = out / f"{name}_fixed.jsonl"
        fixed, failed, dropped = apply_fixes(path, sub, res, dest, drop)
        st["fixed"], st["still_truncated"], st["dropped"] = fixed, failed, dropped
        print(f"{name}: fixed {fixed}, still truncated {failed}, dropped {dropped} -> {dest}")

    with open(out / "regen_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "rows", "turns", "flagged",
                                          "unusable_question", "fixed",
                                          "still_truncated", "dropped"])
        w.writeheader()
        for st in stats:
            for k in ("fixed", "still_truncated", "dropped"):
                st.setdefault(k, 0)
            w.writerow(st)
    print(f"\n-> {out}")


def _selfcheck():
    # ponytail: detection (finish_reason OR judge flag) and the substitution pass
    tmp = Path("/tmp/_regen_selfcheck")
    tmp.mkdir(exist_ok=True)
    src = tmp / "ds.jsonl"
    LONG_Q = "A 62-year-old man presents with progressive exertional dyspnea. What is next?"
    LONG_Q2 = "A 5-year-old girl has recurrent respiratory infections and ataxia. Diagnosis?"
    rows = [
        {"conversations": [{"from": "human", "value": LONG_Q},
                           {"from": "gpt", "value": "cut off mid-"}]},          # judge flag
        {"conversations": [{"from": "user", "value": "q2"},
                           {"from": "assistant", "value": "fine."}]},           # untouched
        {"finish_reason": "length", "label_letter": "C",                       # MCQ row
         "conversations": [{"from": "user", "value": LONG_Q2},
                           {"from": "assistant", "value": "also cut"}]},
        {"finish_reason": "length",                                             # unusable question
         "conversations": [{"from": "user", "value": "... options A-D. Provide answer, then Answer: X."},
                           {"from": "assistant", "value": "leaked"}]},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    flags = tmp / "flags.jsonl"
    flags.write_text(json.dumps({"dataset": "ds", "answer": "cut off mid-",
                                 "truncated": True}) + "\n")
    flagged = load_flags(flags)
    assert _h("cut off mid-") in flagged["ds"]

    jobs, drop, n_rows, n_turns = scan(src, flagged["ds"])
    assert n_rows == 4 and n_turns == 4, (n_rows, n_turns)
    assert [(ln, ti) for ln, ti, _, _ in jobs] == [(0, 1), (2, 1)], jobs
    assert [q for _, _, q, _ in jobs] == [LONG_Q, LONG_Q2], jobs
    assert [lab for _, _, _, lab in jobs] == [False, True], jobs   # row 2 has a label
    assert drop == {3}, drop            # truncated, but its question is generator-speak
    assert not unusable_question(LONG_Q)
    assert unusable_question("...options A-D. Provide answer with rationale, then Answer: X.")
    assert unusable_question("short")
    assert not unusable_question("Question: Management thyroid ca")   # real MedMCQA stub
    kept, _, _, _ = scan(src, flagged["ds"], keep_unusable=True)
    assert len(kept) == 3, kept

    # job 0 regenerates cleanly, job 1 never converges
    full = [(ln, ti, str(src), q, lab) for ln, ti, q, lab in jobs]
    results = {0: {"answer": "whole answer.", "finish_reason": "stop", "tries": 2},
               1: {"answer": "Reasoning. The answer is D.", "finish_reason": "stop", "tries": 1}}
    dest = tmp / "ds_fixed.jsonl"
    fixed, failed, dropped = apply_fixes(src, full, results, dest, drop)
    assert (fixed, failed, dropped) == (2, 0, 1), (fixed, failed, dropped)

    got = [json.loads(l) for l in dest.read_text().splitlines()]
    assert len(got) == 3                                        # the unusable row is gone
    assert got[0]["conversations"][1]["value"] == "whole answer."
    assert got[0]["regen"] is True and got[0]["regen_tries"] == 2
    assert got[1] == rows[1]                                    # passed through untouched
    # the MCQ row is relabelled from its own new answer, old label preserved
    assert got[2]["conversations"][1]["value"] == "Reasoning. The answer is D."
    assert got[2]["label_letter"] == "D", got[2]
    assert got[2]["label_letter_before_regen"] == "C", got[2]
    assert got[2]["regen_stated_letter"] is True

    # format contract: only labeled rows get the MCQ suffix
    class _Tok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[-1]["content"]
    assert MCQ_SUFFIX in build_prompt(_Tok(), "q", False, "low", labeled=True)
    assert MCQ_SUFFIX not in build_prompt(_Tok(), "q", False, "low", labeled=False)
    # is_labeled beats a stray label_letter; absent flag falls back to the letter
    assert is_mcq({"is_labeled": True, "label_letter": "A"})
    assert not is_mcq({"is_labeled": False, "label_letter": "A"})   # the 17,507 rows
    assert is_mcq({"label_letter": "A"}) and not is_mcq({})
    assert chosen_letter("blah. The answer is D.") == "D"
    assert chosen_letter("an essay with no letter at all") is None
    assert chosen_letter("first answer is A, later the answer is C.") == "C"

    # checkpoint round-trip: successes reload, failures are skipped unless asked for
    ck = tmp / "checkpoint.jsonl"
    ck.write_text("\n".join([
        json.dumps({"key": "ds|0|1", "answer": "whole answer.",
                    "finish_reason": "stop", "tries": 2}),
        json.dumps({"key": "ds|2|1", "failed": True}),
        '{"key": "ds|9|1", "answer": "trunc',            # killed mid-write
    ]) + "\n")
    ld = load_checkpoint(ck)
    assert set(ld) == {"ds|0|1", "ds|2|1"}, ld           # torn line ignored
    assert ld["ds|0|1"]["answer"] == "whole answer."
    assert ld["ds|2|1"] is None                          # failed -> skip on relaunch
    assert set(load_checkpoint(ck, retry_failed=True)) == {"ds|0|1"}
    assert job_key("ds", 2, 1) == "ds|2|1"

    assert final_message("<|channel|>analysis<|message|>think" + FINAL + " answer ") == "answer"
    assert final_message("plain answer") == "plain answer"
    assert accept("x" + FINAL + "a", "stop") and not accept("x" + FINAL + "a", "length")
    assert not accept("<|channel|>analysis<|message|>only thinking" + FINAL, "stop")
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
