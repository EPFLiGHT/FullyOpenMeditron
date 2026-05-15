#!/usr/bin/env python3
"""
00_sanity.py
-------------
Sanity-check all six input files.

Outputs:
  - stdout: per-file counts, samples, and flags
  - outputs/sanity_summary.csv: one row per (pair, split, file)
  - outputs/sanity_samples.jsonl: first few records per group
  - outputs/sanity_empty_samples.jsonl: when a loader reports empty questions
    or empty answers, 5 full raw records sampled across the whole file so we
    can see WHY they're empty (schema variant, upstream bug, etc).

Data paths are resolved by `load_data_paths(args)` (see src/config.py), with
priority: CLI flag > YAML config (--config) > built-in default.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Default config path is script-relative so the script runs correctly from
# anywhere (parent repo root, the analysis subdir, etc.).
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)

from schema import Record  # noqa: E402
from loader import (        # noqa: E402
    load_moove_source,
    load_moove_synthetic,
    load_meditron_source,
    load_meditron_synthetic,
    load_guidelines_source,
    load_guidelines_synthetic,
)
from config import load_data_paths  # noqa: E402


def _tok_count(text):
    return len(text.split()) if text else 0


def _summarize(name, records, n_samples=3, max_empty_samples=5):
    total = 0
    empty_full_text = 0
    empty_question = 0
    empty_answer = 0
    mcq = 0
    paired = 0
    doc = 0
    think = 0
    parse_failed = 0
    duplicate_ids = 0
    seen_ids = set()

    tok_sums = {"full_text": 0, "question": 0, "answer": 0}
    tok_max = {"full_text": 0, "question": 0, "answer": 0}
    tok_counts = {"full_text": 0, "question": 0, "answer": 0}

    vote_counter = Counter()
    guideline_sources = Counter()
    countries = Counter()
    source_files = Counter()

    samples = []
    empty_q_samples = []  # deep-scan: records where question is empty
    empty_a_samples = []  # deep-scan: records where answer is empty

    for idx, r in enumerate(records):
        total += 1
        source_files[r.source_file] += 1

        q_empty = (r.question is None or str(r.question).strip() == "")
        a_empty = (r.answer is None or str(r.answer).strip() == "")

        if not r.full_text:
            empty_full_text += 1
        if q_empty:
            empty_question += 1
        if a_empty:
            empty_answer += 1

        if r.is_mcq:
            mcq += 1
        if r.is_paired:
            paired += 1
        if r.is_document:
            doc += 1
        if r.has_think_block:
            think += 1
        if r.parse_failed:
            parse_failed += 1

        if r.id in seen_ids:
            duplicate_ids += 1
        else:
            seen_ids.add(r.id)

        for field_name, val in (("full_text", r.full_text),
                                ("question", r.question),
                                ("answer", r.answer)):
            if val:
                n = _tok_count(val)
                tok_sums[field_name] += n
                tok_counts[field_name] += 1
                if n > tok_max[field_name]:
                    tok_max[field_name] = n

        if r.vote is not None:
            vote_counter[r.vote] += 1
        if r.guideline_source:
            guideline_sources[r.guideline_source] += 1
        if r.country:
            countries[r.country] += 1

        if len(samples) < n_samples:
            d = r.to_dict()
            d.pop("raw", None)
            samples.append(d)

        # Deep-scan: reservoir-style sampling for problem records across the
        # whole file (not just the head). We keep at most max_empty_samples.
        if q_empty and len(empty_q_samples) < max_empty_samples:
            empty_q_samples.append({
                "file_index": idx,
                "record_id": r.id,
                "source_file": r.source_file,
                "raw_keys": sorted(list(r.raw.keys())) if isinstance(r.raw, dict) else None,
                "raw_preview": (json.dumps(r.raw, ensure_ascii=False)[:500]
                                if r.raw else None),
            })
        elif q_empty and len(empty_q_samples) >= max_empty_samples:
            # Replace one at random as we scan further, so we sample across file
            import random
            if random.random() < 0.01:
                empty_q_samples[random.randrange(len(empty_q_samples))] = {
                    "file_index": idx,
                    "record_id": r.id,
                    "source_file": r.source_file,
                    "raw_keys": sorted(list(r.raw.keys())) if isinstance(r.raw, dict) else None,
                    "raw_preview": (json.dumps(r.raw, ensure_ascii=False)[:500]
                                    if r.raw else None),
                }

        if a_empty and len(empty_a_samples) < max_empty_samples:
            empty_a_samples.append({
                "file_index": idx,
                "record_id": r.id,
                "source_file": r.source_file,
                "raw_keys": sorted(list(r.raw.keys())) if isinstance(r.raw, dict) else None,
                "raw_preview": (json.dumps(r.raw, ensure_ascii=False)[:500]
                                if r.raw else None),
            })

    def _avg(name):
        return round(tok_sums[name] / tok_counts[name], 1) if tok_counts[name] else 0.0

    row = {
        "group": name,
        "pair": samples[0]["pair"] if samples else "",
        "split": samples[0]["split"] if samples else "",
        "total": total,
        "empty_full_text": empty_full_text,
        "empty_question": empty_question,
        "empty_answer": empty_answer,
        "is_mcq": mcq,
        "is_paired": paired,
        "is_document": doc,
        "has_think_block": think,
        "parse_failed": parse_failed,
        "duplicate_ids": duplicate_ids,
        "avg_tok_full_text": _avg("full_text"),
        "avg_tok_question": _avg("question"),
        "avg_tok_answer": _avg("answer"),
        "max_tok_full_text": tok_max["full_text"],
        "top_votes": ";".join("{}={}".format(k, v) for k, v in vote_counter.most_common(5)),
        "top_guideline_sources": ";".join("{}={}".format(k, v) for k, v in guideline_sources.most_common(10)),
        "top_countries": ";".join("{}={}".format(k, v) for k, v in countries.most_common(5)),
    }
    return row, samples, empty_q_samples, empty_a_samples


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
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include-magic", action="store_true",
                   help="Include Guidelines/magic.jsonl even though it's login pages")
    return p.parse_args()


def _maybe_limited(iterator, limit):
    if limit is None:
        for x in iterator:
            yield x
        return
    for i, x in enumerate(iterator):
        if i >= limit:
            return
        yield x


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_data_paths(args)

    plan = [
        ("MOOVE source",         paths["moove_source"],         load_moove_source),
        ("MOOVE synthetic",      paths["moove_synthetic"],      load_moove_synthetic),
        ("Meditron source",      paths["curated_source"],       load_meditron_source),
        ("Meditron synthetic",   paths["curated_synthetic"],    load_meditron_synthetic),
        ("Guidelines synthetic", paths["guidelines_synthetic"], load_guidelines_synthetic),
    ]

    summary_rows = []
    sample_records = []
    empty_q_records = []
    empty_a_records = []

    for name, path, fn in plan:
        print("\n" + "=" * 70 + "\n" + name + "\n  file: " + str(path))
        if not path.exists():
            print("  [MISSING] skipping")
            continue
        try:
            it = _maybe_limited(fn(path), args.limit)
            row, samples, eq, ea = _summarize(name, it)
        except Exception as e:
            print("  [ERROR] {}: {}".format(type(e).__name__, e))
            continue
        summary_rows.append(row)
        for s in samples:
            sample_records.append(dict([("group", name)] + list(s.items())))
        for e in eq:
            empty_q_records.append(dict([("group", name)] + list(e.items())))
        for e in ea:
            empty_a_records.append(dict([("group", name)] + list(e.items())))
        for k, v in row.items():
            if k == "group":
                continue
            print("    {:>24}: {}".format(k, v))

    # Guidelines source: iterate directory
    guidelines_dir = paths["guidelines_dir"]
    print("\n" + "=" * 70 + "\nGuidelines source\n  dir: " + str(guidelines_dir))
    if not guidelines_dir.exists():
        print("  [MISSING] skipping")
    else:
        files = sorted(guidelines_dir.glob("*.jsonl"))
        if not files:
            print("  [EMPTY] no *.jsonl files under this dir")
        else:
            print("  found {} jsonl files: {}".format(len(files), [f.name for f in files]))
            combined_total = 0
            merged_counter = Counter()
            for f in files:
                try:
                    it = _maybe_limited(
                        load_guidelines_source(f, include_magic=args.include_magic),
                        args.limit,
                    )
                    row, samples, eq, ea = _summarize("Guidelines src / " + f.stem, it)
                except Exception as e:
                    print("  [{}] ERROR: {}: {}".format(f.name, type(e).__name__, e))
                    continue
                summary_rows.append(row)
                combined_total += row["total"]
                merged_counter[f.stem] = row["total"]
                if samples:
                    sample_records.append(dict(
                        [("group", "Guidelines src / " + f.stem)] + list(samples[0].items())
                    ))
                # Guidelines have answer=None by design — don't flood empty_a_records
                # with those, but DO capture empty full_text (parse_failed=True records)
                print("  - {}: total={} avg_tok_full={} max_tok={} parse_failed={}".format(
                    f.name, row["total"], row["avg_tok_full_text"],
                    row["max_tok_full_text"], row["parse_failed"]
                ))

            print("\n  COMBINED guidelines source: total={}  per-source={}".format(
                combined_total, dict(merged_counter)))

    # Write CSV
    csv_path = out_dir / "sanity_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        for r in summary_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print("\nWrote {}".format(csv_path))

    samples_path = out_dir / "sanity_samples.jsonl"
    if sample_records:
        with samples_path.open("w", encoding="utf-8") as f:
            for s in sample_records:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print("Wrote {}  ({} records)".format(samples_path, len(sample_records)))

    empty_q_path = out_dir / "sanity_empty_questions.jsonl"
    if empty_q_records:
        with empty_q_path.open("w", encoding="utf-8") as f:
            for s in empty_q_records:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print("Wrote {}  ({} empty-question diagnostics)".format(empty_q_path, len(empty_q_records)))

    empty_a_path = out_dir / "sanity_empty_answers.jsonl"
    if empty_a_records:
        with empty_a_path.open("w", encoding="utf-8") as f:
            for s in empty_a_records:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print("Wrote {}  ({} empty-answer diagnostics)".format(empty_a_path, len(empty_a_records)))


if __name__ == "__main__":
    main()