#!/usr/bin/env python3
"""Build the unanimous-agreement evaluation set.

Selection rule, applied to the triplets of ``all_reviews.jsonl``:

1. at least two distinct physicians reviewed the triplet;
2. all of them gave the same final verdict (strict unanimity);
3. the question is neither amharic nor oromo.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "data_HF" / "moove_eval_unanimous_576.jsonl"
)
CLASSES = ("Model 1", "Model 2", "Equal")
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")
ETHIOPIAN_PREFIX_RE = re.compile(r"LMH Ethiopia|Ethiopia", re.IGNORECASE)
OROMO_TOKENS = {
    "fi", "irraa", "keessatti", "qaba", "akka", "kana", "kun", "isa",
    "ishee", "maal", "gochuu", "qabdi", "qabu", "yeroo", "malee",
    "danda", "taasifamu", "hojjettuun", "fayyaa", "waggaa", "tokko",
    "booda", "irratti", "jira",
}
ENGLISH_TOKENS = {
    "the", "and", "is", "to", "a", "with", "child", "should", "has",
    "no", "for", "of", "this", "history", "be", "criteria", "discharge",
    "treatment", "active", "breathing", "admitted", "what", "how", "use",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", " ", text)


def parse_timestamp(value: Any) -> dt.datetime:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    return dt.datetime.min


def physician_id(row: dict[str, Any]) -> str:
    for field in ("User ID", "rater", "SSO ID", "User name"):
        value = normalize_text(row.get(field))
        if value:
            return value
    return ""


def canonicalize(row: dict[str, Any]) -> tuple[tuple[str, str, str], str] | None:
    question = normalize_text(row.get("left_question_text"))
    right_question = normalize_text(row.get("right_question_text"))
    left = normalize_text(row.get("left_answer_text"))
    right = normalize_text(row.get("right_answer_text"))
    if not question or question != right_question or not left or not right or left == right:
        return None
    vote = normalize_text(row.get("Vote")).lower()
    if vote not in {"left", "right", "equal"}:
        return None
    answer_a, answer_b = sorted((left, right))
    if vote == "equal":
        label = "Equal"
    else:
        physical_winner = left if vote == "left" else right
        label = "Model 1" if physical_winner == answer_a else "Model 2"
    return (question, answer_a, answer_b), label


def stable_gold_id(key: tuple[str, str, str]) -> str:
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def oromo_score(text: str) -> tuple[float, int]:
    text = text.lower()
    words = re.findall(r"[a-z']+", text)
    oromo = sum(word in OROMO_TOKENS for word in words)
    oromo += len(re.findall(r"[a-z]'[a-z]", text))
    oromo += 0.1 * sum(bool(re.search(r"(aa|ee|ii|oo|uu)", word)) for word in words)
    english = sum(word in ENGLISH_TOKENS for word in words)
    return oromo, english


def language(row: dict[str, Any]) -> str:
    fields = (
        row.get("question", ""),
        row.get("model_1_answer", ""),
        row.get("model_2_answer", ""),
    )
    if any(ETHIOPIC_RE.search(field) for field in fields):
        return "amharic"

    if not any(ETHIOPIAN_PREFIX_RE.search(p) for p in row.get("prefixes", [])):
        return "other"

    spans = (fields[0], " ".join(fields))
    if any(oromo >= english for oromo, english in map(oromo_score, spans)):
        return "oromo"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Raw all_reviews.jsonl export")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Defaults to {DEFAULT_OUTPUT}, independently of the "
                             "working directory")
    parser.add_argument("--min-raters", type=int, default=2)
    parser.add_argument("--keep-amharic-oromo", action="store_true",
                        help="Skip the language filter (yields the full unanimous pool)")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    if args.min_raters < 1:
        parser.error("--min-raters must be at least 1")
    return args


def main() -> None:
    args = parse_args()

    grouped: dict[tuple[str, str, str], dict[str, tuple[Any, ...]]] = (
        collections.defaultdict(dict)
    )
    metadata: dict[tuple[str, str, str], dict[str, set[str]]] = (
        collections.defaultdict(lambda: {"prefix": set(), "care_level": set(),
                                         "working_group": set()})
    )
    counters: collections.Counter[str] = collections.Counter()

    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counters["physical_lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{args.input}:{line_number}: malformed JSON ({error})"
                ) from None
            canonical = canonicalize(row)
            rater = physician_id(row)
            if canonical is None or not rater:
                counters["invalid_review"] += 1
                continue

            key, label = canonical
            timestamp = parse_timestamp(row.get("Creation timestamp"))
            annotation_id = normalize_text(row.get("ID")) or f"line-{line_number}"
            previous = grouped[key].get(rater)
            if previous is None or timestamp >= previous[0]:
                if previous is not None:
                    counters["duplicate_rater_reviews_replaced"] += 1
                grouped[key][rater] = (timestamp, label, annotation_id)
            else:
                counters["older_duplicate_rater_reviews_ignored"] += 1

            for field, source in (("prefix", "prefix"),
                                  ("care_level", "care_level"),
                                  ("working_group", "Working group name")):
                value = normalize_text(row.get(source))
                if value:
                    metadata[key][field].add(value)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        reviews = grouped[key]
        if len(reviews) < args.min_raters:
            counters["too_few_raters"] += 1
            continue

        labels = {review[1] for review in reviews.values()}
        if len(labels) != 1:
            counters["not_unanimous"] += 1
            continue

        question, answer_a, answer_b = key
        meta = metadata[key]
        if not args.keep_amharic_oromo:
            detected = language({"question": question,
                                 "model_1_answer": answer_a,
                                 "model_2_answer": answer_b,
                                 "prefixes": sorted(meta["prefix"])})
            if detected != "other":
                counters[f"excluded_{detected}"] += 1
                continue

        label = labels.pop()
        n_raters = len(reviews)
        rows.append({
            "gold_id": stable_gold_id(key),
            "question": question,
            "model_1_answer": answer_a,
            "model_2_answer": answer_b,
            "label": label,
            "n_raters": n_raters,
            "stratum": "gold_unanimous" if n_raters >= 3 else "silver_unanimous",
            "vote_counts": {c: n_raters if c == label else 0 for c in CLASSES},
            "vote_distribution": {c: 1.0 if c == label else 0.0 for c in CLASSES},
            "majority_probability": 1.0,
            "unanimous": True,
            "annotation_ids": sorted(review[2] for review in reviews.values()),
            "prefixes": sorted(meta["prefix"]),
            "care_levels": sorted(meta["care_level"]),
            "working_groups": sorted(meta["working_group"]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "criteria": {
            "min_distinct_raters": args.min_raters,
            "required_agreement": "strict unanimity on the final verdict",
            "language_filter": not args.keep_amharic_oromo,
        },
        "source": {**dict(sorted(counters.items())),
                   "canonical_triplets": len(grouped)},
        "unanimous_triplets": len(rows),
        "label_counts": dict(collections.Counter(r["label"] for r in rows)),
        "stratum_counts": dict(collections.Counter(r["stratum"] for r in rows)),
        "rater_count_distribution": dict(sorted(
            collections.Counter(r["n_raters"] for r in rows).items())),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
