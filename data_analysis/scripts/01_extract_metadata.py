#!/usr/bin/env python3
"""
01_extract_metadata.py
----------------------
Tag sampled records from each (pair, split) group with clinical metadata.

Workflow
--------
For each of the 6 groups:
  1. Load records via src/loader.py
  2. Drop invalid/junk via metadata_extract.filter_valid
  3. Sample to --n-per-group (deterministic, stratified where appropriate)
  4. Resume: read any existing output JSONL to find already-tagged IDs
  5. Build chat messages for the vLLM model
  6. Batch-generate via llm.chat()
  7. Parse + validate JSON output
  8. Append to output JSONL

Output layout
-------------
  {out_dir}/metadata/
    moove__source.jsonl
    moove__synthetic.jsonl
    Curated_QA__source.jsonl
    Curated_QA__synthetic.jsonl
    guidelines__source.jsonl
    guidelines__synthetic.jsonl

Each line of an output JSONL is a Record dict plus:
    "analysis_metadata": { <tagged fields>, "_parse_ok": bool }

Resuming
--------
Re-running with the same --out-dir skips records already present in the
output JSONL. Safe to cancel and restart. If you raise --n-per-group, only
the additional records are tagged.

Data paths are resolved by `load_data_paths(args)` (see src/config.py), with
priority: CLI flag > YAML config (--config) > built-in default.
"""

import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Default config path is script-relative so the script runs correctly from
# anywhere (parent repo root, the analysis subdir, etc.).
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)

from loader import (            # noqa: E402
    load_moove_source,
    load_moove_synthetic,
    load_meditron_source,
    load_meditron_synthetic,
    load_guidelines_source,
    load_guidelines_synthetic,
)
from sampling import sample_group  # noqa: E402
from metadata_extract import (     # noqa: E402
    filter_valid,
    build_messages,
    parse_model_output,
)
from config import load_data_paths  # noqa: E402


# ---------------------------------------------------------------------------
# Group plan
# ---------------------------------------------------------------------------

def iter_guidelines_source_dir(dir_path):
    """Stream every Guidelines source file under the dir."""
    dir_path = Path(dir_path)
    for f in sorted(dir_path.glob("*.jsonl")):
        for r in load_guidelines_source(f):
            yield r


def build_plan(args):
    """Return list of (group_name, pair, split, loader_callable)."""
    paths = load_data_paths(args)

    return [
        ("moove__source",          "moove",      "source",    lambda: load_moove_source(paths["moove_source"])),
        ("moove__synthetic",       "moove",      "synthetic", lambda: load_moove_synthetic(paths["moove_synthetic"])),
        ("Curated_QA__source",     "meditron",   "source",    lambda: load_meditron_source(paths["curated_source"])),
        ("Curated_QA__synthetic",  "meditron",   "synthetic", lambda: load_meditron_synthetic(paths["curated_synthetic"])),
        ("guidelines__source",     "guidelines", "source",    lambda: iter_guidelines_source_dir(paths["guidelines_dir"])),
        ("guidelines__synthetic",  "guidelines", "synthetic", lambda: load_guidelines_synthetic(paths["guidelines_synthetic"])),
    ]


# ---------------------------------------------------------------------------
# Resume: read already-tagged IDs
# ---------------------------------------------------------------------------

def load_already_tagged(out_path):
    """Return set of record IDs that have already been written to out_path."""
    if not out_path.exists():
        return set()
    tagged = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            if rid:
                tagged.add(rid)
    return tagged


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                   help="YAML config with data paths. CLI flags override config values. "
                        "Pass --config '' to skip and use built-in defaults.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--moove-source", type=str, default=None)
    p.add_argument("--moove-synthetic", type=str, default=None)
    p.add_argument("--meditron-source", type=str, default=None)
    p.add_argument("--meditron-synthetic", type=str, default=None)
    p.add_argument("--guidelines-dir", type=str, default=None)
    p.add_argument("--guidelines-synthetic", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs")

    p.add_argument("--model", type=str, required=True,
                   help="HF model path or name (e.g. Qwen/Qwen3-32B or openai/gpt-oss-120b)")
    p.add_argument("--tp", type=int, default=2,
                   help="tensor_parallel_size. Use 2 for Qwen3-32B on 2xH100.")
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max tokens the model can generate per request. "
                        "Metadata JSON is tiny (~150 tokens) — keep this low.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Keep Qwen3 <think> blocks enabled. Default is off for "
                        "this stage — we just want the JSON fast.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-group", type=int, default=2000,
                   help="Records to tag per (pair, split). -1 for full run.")
    p.add_argument("--only-group", type=str, default=None,
                   help="Restrict to a single group name (e.g. moove__source)")
    p.add_argument("--reasoning-effort", type=str, default="low",
                   choices=["low", "medium", "high"],
                   help="GPT-OSS reasoning effort. Ignored by Qwen.")

    p.add_argument("--dry-run", action="store_true",
                   help="Print the sample counts + first prompt, then exit "
                        "without loading vLLM.")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Flush output JSONL every N records (for resume safety).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# vLLM adapter
# ---------------------------------------------------------------------------

def init_vllm(args):
    """Import + construct vLLM LLM. Kept in a function so --dry-run avoids it."""
    from vllm import LLM, SamplingParams

    print("[vllm] loading model: {}".format(args.model))
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    print("[vllm] model loaded in {:.1f}s".format(time.time() - t0))

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    # Stash reasoning_effort so generate_batch can pick it up for GPT-OSS
    sampling._reasoning_effort = args.reasoning_effort
    return llm, sampling


def generate_batch(llm, sampling, chat_messages_list, model_name, enable_thinking):
    """Run a batch of chat requests. Returns list of raw output texts.

    Selects the right chat_template_kwargs based on model family:
      - Qwen3:   enable_thinking flag
      - GPT-OSS: reasoning_effort flag (already set in sampling)
      - Others:  no kwargs
    """
    model_lower = (model_name or "").lower()
    chat_template_kwargs = {}

    if "qwen" in model_lower:
        chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
    # GPT-OSS: reasoning_effort is passed via SamplingParams.extra_body or
    # by setting it in the tokenizer's chat template. vLLM handles this
    # automatically from the chat messages if the server is configured,
    # but for offline mode we rely on defaults — "medium" unless told otherwise.
    # If you need "low" effort for speed, set via chat_template_kwargs:
    elif "gpt-oss" in model_lower or "gpt_oss" in model_lower:
        # reasoning_effort is the supported knob for GPT-OSS Harmony template
        # Value comes from args.reasoning_effort, passed in via the sampling object
        effort = getattr(sampling, "_reasoning_effort", "low")
        chat_template_kwargs["reasoning_effort"] = effort

    outputs = llm.chat(
        messages=chat_messages_list,
        sampling_params=sampling,
        chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
        use_tqdm=False,
    )
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_group(group_name, pair, split, loader_fn, args, llm, sampling):
    """Run the extractor over one group."""
    out_dir = Path(args.out_dir) / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "{}.jsonl".format(group_name)

    print("\n" + "=" * 70)
    print("Group: {}  ({}, {})".format(group_name, pair, split))
    print("Output: {}".format(out_path))

    already = load_already_tagged(out_path)
    if already:
        print("  resuming — {} records already tagged".format(len(already)))

    # Step 1: load + filter + sample
    print("  loading + filtering + sampling (target n={})".format(args.n_per_group))
    t0 = time.time()
    raw_records = loader_fn()
    valid_records = filter_valid(raw_records)
    sampled = list(sample_group(valid_records, pair, split, args.n_per_group, seed=args.seed))
    print("  sampled {} records in {:.1f}s".format(len(sampled), time.time() - t0))

    # Skip already-tagged
    to_tag = [r for r in sampled if r.id not in already]
    print("  {} records to tag this run".format(len(to_tag)))

    if args.dry_run:
        if to_tag:
            print("  [dry-run] first prompt preview:")
            msgs = build_messages(to_tag[0])
            print("    system: {}...".format(msgs[0]["content"][:80]))
            print("    user:   {}...".format(msgs[1]["content"][:200]))
        return

    if not to_tag:
        print("  nothing to do")
        return

    # Step 2: batch-generate
    BATCH = 256
    n_done = 0
    n_parse_ok = 0
    t_start = time.time()

    with out_path.open("a", encoding="utf-8") as out_f:
        for i in range(0, len(to_tag), BATCH):
            batch = to_tag[i:i + BATCH]
            messages_list = [build_messages(r) for r in batch]
            try:
                raw_outputs = generate_batch(
                    llm, sampling, messages_list, args.model, args.enable_thinking
                )
            except Exception as e:
                print("  [error in batch starting at {}: {}: {}]".format(
                    i, type(e).__name__, e))
                continue

            for r, raw in zip(batch, raw_outputs):
                meta = parse_model_output(raw, is_document=r.is_document)
                if meta.get("_parse_ok"):
                    n_parse_ok += 1
                d = r.to_dict()
                d.pop("raw", None)
                d["analysis_metadata"] = meta
                out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_done += 1

                if n_done % args.flush_every == 0:
                    out_f.flush()

            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(to_tag) - n_done) / rate if rate > 0 else 0
            print("  [{}/{}] parse_ok={}/{} rate={:.1f}rec/s eta={:.0f}s".format(
                n_done, len(to_tag), n_parse_ok, n_done, rate, eta))
        out_f.flush()

    print("  done: {} tagged, {} parse_ok ({:.1%})".format(
        n_done, n_parse_ok, n_parse_ok / max(n_done, 1)))


def main():
    args = parse_args()
    plan = build_plan(args)

    if args.only_group:
        plan = [p for p in plan if p[0] == args.only_group]
        if not plan:
            print("No group matches --only-group {}".format(args.only_group))
            return

    # Dry run: no vLLM
    if args.dry_run:
        print("[dry-run] skipping vLLM initialization")
        for group_name, pair, split, loader_fn in plan:
            process_group(group_name, pair, split, loader_fn, args, None, None)
        return

    llm, sampling = init_vllm(args)

    for group_name, pair, split, loader_fn in plan:
        try:
            process_group(group_name, pair, split, loader_fn, args, llm, sampling)
        except Exception as e:
            print("[ERROR in group {}: {}: {}]".format(group_name, type(e).__name__, e))
            continue

    print("\nAll groups complete.")


if __name__ == "__main__":
    main()