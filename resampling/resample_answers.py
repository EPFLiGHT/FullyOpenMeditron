#!/usr/bin/env python3
# resampling/resample_answers.py
"""
Re-answer every question in the decontaminated training sets with pass@k
rejection sampling, keeping only answers that finished on their own.

What it fixes, relative to distill_with_retries_v3.py:

  * That loop overwrote the assistant turn with the newest sample whether or
    not it matched the label, so a row that failed all 8 tries kept the 8th
    wrong answer next to exact_match: false. Here a verified sample is kept and
    everything else is written to rejected.jsonl and dropped.
  * It capped generation at 2048 tokens and never looked at finish_reason, so
    every completion that hit the cap became a finished turn -- the learned
    mid-sentence stop in truncation/README.md. Here an incomplete sample is
    never eligible, whatever its letter.

Sampling is in waves (--k-schedule, default 1,3,4): one sample for every row,
then 3 more only for the rows that failed, then 4 more. Flat pass@8 would pay
8x decode on the rows that pass first try; the waves pay it only where it is
needed, in 3 batched rounds instead of 8 sequential single-sample ones. Pass
--k-schedule 8 for the flat version.

Which rows get verified: MCQ rows with a label (is_mcq(), the shared contract
from truncation/regenerate_truncated.py); the synthetic sets use the label the
question generator wrote. Free-form rows have no
verifier, so they take one complete sample and are checked for completeness
only.

usage:
bash new_launch.sh --partition debug --time 1:29:59 --requirements none \\
    resampling/resample_answers.py --model openai/gpt-oss-120b \\
    --input data/decontaminated/gpt_oss_8retries_v3_fixed_no_medexqa_decontam.jsonl \\
    --outdir $SCRATCH/meditronfo/resampling/answers

    --k-schedule 1,3,4   samples per wave
    --label-field auto   auto = label_maj if present, else label_letter
    --smoke 200          200 rows per input into <outdir>/smoke, with a summary
    --selfcheck          offline unit checks, no GPU

writes:
    <outdir>/<name>_resampled.jsonl   verified rows only, answers replaced
    <outdir>/<name>_rejected.jsonl    dropped rows, with what the model said
    <outdir>/resample_report.csv      per dataset: pass@1, pass@k, rejects
    <outdir>/checkpoint.jsonl         resume state; delete it to start over
"""

import argparse
import collections
import csv
import json
import re
import random
from datetime import datetime
from pathlib import Path

from common import (Checkpoint, Rate, add_io_args, add_sampling_args, add_shard_args, build_prompt,
                    chosen_letter, chunked, fill_defaults, load_llm, normalize_rows, prepare_row,
                    read_jsonl, row_kind, sample_batch, shard_spec, shard_suffix, write_jsonl)

RESAMPLE_DEFAULTS = {
    "resample": False,
    "resample_model": "",
    "resample_verified": False,   # the kept answer named the labelled letter
    "resample_letter": "",
    "resample_label": "",         # the label it was verified against
    "resample_samples": 0,        # complete samples drawn for this row
    "resample_wave": 0,           # wave the kept answer came from
    "resample_kind": "",          # mcq (verified) or free (completeness only)
    "finish_reason": "stop",      # kept answers are complete by construction
}


def turns(row):
    """(question, index of the assistant turn to overwrite)."""
    conv = row.get("conversations", [])
    q, aidx = "", None
    for i, turn in enumerate(conv):
        role = turn.get("from")
        if not q and role in ("human", "user"):
            q = turn.get("value", "").strip()
        elif q and role in ("assistant", "gpt") and aidx is None:
            aidx = i
    return q, aidx


def answer_of(row):
    """The assistant turn these corpora carry -- single-turn, so the last one."""
    for turn in reversed(row.get("conversations", [])):
        if turn.get("from") in ("assistant", "gpt"):
            return turn.get("value", "")
    return ""


def label_of(row, field):
    if field == "auto":
        return (row.get("label_maj") or row.get("label_letter") or "").strip()
    return (row.get(field) or "").strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(p)
    add_sampling_args(p)
    add_shard_args(p)
    p.add_argument("--k-schedule", default="1,3,4",
                   help="samples per wave; rows verified in one wave skip the rest")
    p.add_argument("--label-field", default="auto",
                   choices=["auto", "label_maj", "label_letter"])
    p.add_argument("--no-options", default="drop", choices=["drop", "free"],
                   help="MCQ rows whose question lists no options (11.1% of "
                        "meditron_4_synthetic_qa_v3): drop them, or answer them "
                        "free-form and keep them unverified")
    p.add_argument("--keep-rejected-samples", type=int, default=2,
                   help="samples stored per rejected row, for audit")
    p.add_argument("--drop-cols", default="raw_output",
                   help="comma-separated columns to drop; raw_output is the question "
                        "generator's stale harmony text and 8x the guidelines file size")
    return p.parse_args()


def classify(row, args):
    """"mcq", "free", or "" for a row that cannot be used at all."""
    kind = row_kind(row)
    if kind == "mcq":
        # no label means nothing to verify against, so it is not an MCQ job
        return "mcq" if label_of(row, args.label_field) else "free"
    if kind == "free":
        return "free"
    if kind == "drop:no_options" and args.no_options == "free":
        return "free"
    return ""


def resolve(rows, jobs, args, tokenizer, llm, stop_ids, ckpt, name, harmony, date):
    """Run the waves over `jobs` (row indices), checkpointing each resolved row."""
    rng = random.Random(args.seed)
    schedule = [int(x) for x in args.k_schedule.split(",") if x.strip()]
    pending = [i for i in jobs if f"{name}:{i}" not in ckpt]
    drawn = {i: 0 for i in pending}

    for wave, k in enumerate(schedule, start=1):
        if not pending:
            break
        print(f"--- wave {wave}/{len(schedule)}: {len(pending)} rows x {k} sample(s) ---",
              flush=True)
        rate = Rate(len(pending))
        still = []
        for _, chunk in chunked(pending, args.chunk_size):
            prompts, metas = [], []
            for i in chunk:
                q, aidx = turns(rows[i])
                mcq = classify(rows[i], args) == "mcq"
                prompts.append(build_prompt(tokenizer, q, mcq=mcq, harmony=harmony,
                                            reasoning=args.reasoning, date=date,
                                            thinking=args.thinking == "on"))
                metas.append((i, aidx, mcq))
            samples = sample_batch(llm, prompts, n=k, need=1,
                                   temperature=args.temperature,
                                   max_new_tokens=args.max_new_tokens,
                                   stop_token_ids=stop_ids,
                                   seed=None if args.seed is None else args.seed + 100 * wave,
                                   length_retries=args.length_retries, grow=args.grow,
                                   log=lambda m: print(m, flush=True))
            for (i, aidx, mcq), answers in zip(metas, samples):
                drawn[i] += len(answers)
                label = label_of(rows[i], args.label_field)
                if mcq:
                    good = [a for a in answers if chosen_letter(a) == label]
                else:
                    good = answers
                if good:
                    # uniformly at random, not the shortest: picking the
                    # shortest correct answer biases the corpus toward terse
                    # answers, which is the failure this whole effort is about
                    keep = rng.choice(good)
                    ckpt.add(f"{name}:{i}", {
                        "answer": keep, "verified": bool(mcq),
                        "letter": chosen_letter(keep) or "", "label": label,
                        "samples": drawn[i], "wave": wave, "aidx": aidx,
                        "kind": "mcq" if mcq else "free",
                    })
                elif wave == len(schedule):
                    ckpt.add(f"{name}:{i}", {
                        "failed": True, "label": label, "samples": drawn[i],
                        "letters": [chosen_letter(a) or "" for a in answers],
                        "texts": answers[:args.keep_rejected_samples], "aidx": aidx,
                    })
                else:
                    still.append(i)
            ckpt.flush()
            print(f"    {rate.step(len(chunk))}", flush=True)
        pending = still


def run(args):
    assert args.input and args.outdir, "--input and --outdir are required"
    outdir = Path(args.outdir) / ("smoke" if args.smoke else "")
    outdir.mkdir(parents=True, exist_ok=True)
    limit = args.smoke or args.limit
    date = datetime.now().strftime("%Y-%m-%d")
    harmony = "gpt-oss" in args.model

    shard, nshards = shard_spec(args.shard)
    sfx = shard_suffix(shard, nshards)
    if nshards > 1:
        print(f"--- shard {shard} of {nshards}: 1 row in {nshards} ---", flush=True)

    llm, tokenizer, stop_ids = load_llm(args)
    ckpt = Checkpoint(outdir / f"checkpoint{sfx}.jsonl")
    report = []

    for path in args.input:
        # a shard suffix on the input must not nest in the output name
        name = re.sub(r"\.shard\d+of\d+$", "", Path(path).name.replace(".jsonl", ""))
        print(f"=== {name} ===", flush=True)
        rows = [prepare_row(r) for r in read_jsonl(path, limit)]
        kinds = {i: classify(r, args) for i, r in enumerate(rows)}
        mine = [i for i in range(len(rows)) if i % nshards == shard]
        jobs = [i for i in mine if kinds[i]]
        n_mcq = sum(1 for i in mine if kinds[i] == "mcq")
        unusable = collections.Counter(row_kind(rows[i]) for i in mine if not kinds[i])
        print(f"    preflight: {n_mcq} MCQ / {len(jobs) - n_mcq} free-form / "
              f"{len(mine) - len(jobs)} unusable {dict(unusable)}", flush=True)

        resolve(rows, jobs, args, tokenizer, llm, stop_ids, ckpt, name, harmony, date)

        kept, rejected = [], []
        for i in mine:
            if not kinds[i]:
                rejected.append({"dataset": name, "row": i, "question": turns(rows[i])[0],
                                 "label": "", "samples": 0, "letters": [], "texts": [],
                                 "reason": row_kind(rows[i])})
        for i in jobs:
            rec = ckpt.get(f"{name}:{i}")
            row = rows[i]
            if rec is None or rec.get("failed"):
                rejected.append({
                    "dataset": name, "row": i, "question": turns(row)[0],
                    "label": (rec or {}).get("label", ""),
                    "samples": (rec or {}).get("samples", 0),
                    "letters": (rec or {}).get("letters", []),
                    "texts": (rec or {}).get("texts", []),
                    "reason": "no sample passed" if rec else "not attempted",
                })
                continue
            aidx = rec.get("aidx")
            if aidx is None:
                row["conversations"].append({"from": "assistant", "value": rec["answer"]})
            else:
                row["conversations"][aidx]["value"] = rec["answer"]
            row.update({
                "resample": True, "resample_model": args.model,
                "resample_verified": rec["verified"], "resample_letter": rec["letter"],
                "resample_label": rec["label"], "resample_samples": rec["samples"],
                "resample_wave": rec["wave"], "resample_kind": rec.get("kind", ""),
                "finish_reason": "stop", "source_row": i,   # merge_shards.py restores order
            })
            kept.append(fill_defaults(row, RESAMPLE_DEFAULTS))

        normalize_rows(kept, drop=[c for c in args.drop_cols.split(",") if c])
        write_jsonl(kept, outdir / f"{name}_resampled{sfx}.jsonl")
        write_jsonl(rejected, outdir / f"{name}_rejected{sfx}.jsonl")
        mcq_kept = [r for r in kept if r["resample_verified"]]
        report.append({
            "dataset": name, "shard": sfx.lstrip("."), "rows": len(mine),
            "questions": len(jobs), "mcq": n_mcq,
            "unusable": len(mine) - len(jobs),
            "kept": len(kept), "rejected": len(rejected),
            "verified": len(mcq_kept),
            "pass_at_1": round(sum(1 for r in mcq_kept if r["resample_wave"] == 1)
                               / max(n_mcq, 1), 4),
            "pass_at_k": round(len(mcq_kept) / max(n_mcq, 1), 4),
            "samples_per_row": round(sum(r["resample_samples"] for r in kept)
                                     / max(len(kept), 1), 2),
            "mean_answer_chars": round(sum(len(answer_of(r)) for r in kept)
                                       / max(len(kept), 1)),
        })
        print(f"    {json.dumps(report[-1])}", flush=True)

    ckpt.close()
    with open(outdir / f"resample_report{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    print(f"--- wrote {outdir}/resample_report{sfx}.csv ---")

    if args.smoke:
        smoke_gate(outdir, report, sfx)


def smoke_gate(outdir, report, sfx=""):
    """Cheap checks that would have caught both bugs in truncation/README.md."""
    print("\n--- smoke gate ---")
    ok = True
    for r in report:
        kept = read_jsonl(outdir / f"{r['dataset']}_resampled{sfx}.jsonl")
        if not kept:
            print(f"  FAIL {r['dataset']}: nothing kept"); ok = False; continue
        nulls = {k for row in kept for k, v in row.items() if v is None}
        empty = [row for row in kept if not answer_of(row).strip()]
        mcq = [row for row in kept if row["resample_verified"]]
        bad = [row for row in mcq if chosen_letter(answer_of(row)) != row["resample_label"]]
        lens = sorted(len(answer_of(row)) for row in kept)
        print(f"  {r['dataset']}: kept {len(kept)}, pass@1 {r['pass_at_1']}, "
              f"pass@k {r['pass_at_k']}, answer chars p50 {lens[len(lens) // 2]}, "
              f"p90 {lens[int(len(lens) * 0.9)]}")
        for label, bad_rows in (("null fields", nulls), ("empty answers", empty),
                                ("verified rows whose answer is not the label", bad)):
            if bad_rows:
                print(f"  FAIL {r['dataset']}: {len(bad_rows)} {label}"); ok = False
    print("--- smoke gate: " + ("PASS" if ok else "FAIL") + " ---")
    return ok


def _selfcheck():
    row = {"conversations": [{"from": "human", "value": " q "},
                             {"from": "assistant", "value": "old"}]}
    assert turns(row) == ("q", 1)
    assert turns({"conversations": [{"from": "human", "value": "q"}]}) == ("q", None)
    # auto prefers the voted label over the generator's claim
    assert label_of({"label_maj": "B", "label_letter": "A"}, "auto") == "B"
    assert label_of({"label_maj": "", "label_letter": "A"}, "auto") == "A"
    assert label_of({"label_maj": "B", "label_letter": "A"}, "label_letter") == "A"
    # the MCQ contract comes from the shared classifier, not from having a letter
    class A:   # stand-in for parsed args
        label_field, no_options = "auto", "drop"
    mcq_row = {"is_labeled": True, "label_letter": "A",
               "conversations": [{"from": "human", "value": "Pick.\nA) x\nB) y"},
                                 {"from": "assistant", "value": "old"}]}
    assert classify(mcq_row, A) == "mcq"
    assert classify({**mcq_row, "is_labeled": False}, A) == "free"
    # an MCQ row with no options in the question: dropped, or answered free-form
    noopt = {**mcq_row, "conversations": [{"from": "human", "value": "Which of the following?"},
                                          {"from": "assistant", "value": "old"}]}
    assert classify(noopt, A) == ""
    class B(A):
        no_options = "free"
    assert classify(noopt, B) == "free"
    # a labelled row with no label to verify against falls back to free-form
    assert classify({**mcq_row, "label_letter": ""}, A) == "free"
    assert chosen_letter("blah. The answer is C.") == "C"
    assert answer_of(row) == "old"
    assert chosen_letter("no letter here") is None
    assert fill_defaults({}, RESAMPLE_DEFAULTS)["resample_verified"] is False
    assert all(v is not None for v in fill_defaults({}, RESAMPLE_DEFAULTS).values())
    print("selfcheck ok")


if __name__ == "__main__":
    args = parse_args()
    if args.selfcheck:
        _selfcheck()
    else:
        run(args)
