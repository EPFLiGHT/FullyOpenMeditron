#!/usr/bin/env python3
# moove/moove_build.py

"""
Build the pooled per-review artifact that the merge and agreement scripts read.

This is stage 1 of three:

    moove_build.py            CSV exports  -> all_reviews.jsonl (one row/review)
    moove_merge_v2.py         artifact     -> deduped pairs + train/test splits
    moove_agreement_stats.py  artifact     -> coverage and inter-rater agreement

Everything that is a property of a single review happens here: joining the
question and answer bodies onto both sides, building the pair and group keys,
attaching site metadata, auditing the raw score ranges, and mapping the raw
per-criterion evaluations onto the five dimensions. Nothing here collapses
reviews together, so both downstream scripts see identical inputs.

The raw `Left/Right evaluations: <criterion>` columns are kept in the artifact
so the range audit can be rerun without touching the CSVs again.

usage:

    bash new_launch.sh moove/moove_build.py
    python moove/moove_build.py --prefix "Malawi PHC" --verbose
    python moove/moove_build.py --data-dir /some/where --out-dir /other
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

ARTIFACT_NAME = "all_reviews.jsonl"
QUESTIONS_NAME = "all_questions.jsonl"
META_SUFFIX = ".meta.json"

prefix_list = [
    "Belgium",
    "CHUV - Sprint Testing",
    "CHUV (Centre Hospitalier Universitaire Vaudois)",
    "CHUV Research Space",
    "HUG (Hôpitaux Universitaires de Genève)",
    "India-ValiDATAthon-Ashoka",
    "India-ValiDATAthon-BLR",
    "India-ValiDATAthon-HYD",
    "India-ValiDATAthon-NIMS",
    "Kenya ValiDATAthon",
    "LMH Ethiopia MOOVE 1",
    "LMH Ethiopia MOOVE 2",
    "Malawi PHC",
    "Tanzania PHC",
    "Tanzania Validatathon",
    "Kenya PHC mini",
    "Kenya PHC",
    "Malawi PHC Mini",
]

ETHIOPIA_PREFIXES = {
    "LMH Ethiopia MOOVE 1",
    "LMH Ethiopia MOOVE 2",
}

# removed prefixes
#    "University of Nairobi",
#    "Brazil Moove",

# criterion mapping
CRITERION_TO_DIMENSION = {
    "Accuracy":                  "Clinical Soundness",
    "Relevance":                 "Contextual Fit",
    "Relevance & completeness":  "Contextual Fit",
    "Completeness":              "Clarity & Completeness",
    "Clarity & Completeness":    "Clarity & Completeness",
    "Clarity":                   "Clarity & Completeness",
    "Communication":             "Clarity & Completeness",
    "Clinical Reasoning":        "Clinical Soundness",
    "Logical reasoning":         "Clinical Soundness",
    "Clinical Soundness":        "Clinical Soundness",
    "Instruction Following":     "Clarity & Completeness",
    "Harmfulness (Safety)":      "Safety",
    "Harmlessness (Safety)":     "Safety",
    "Harmlessness":              "Safety",
    "Safety":                    "Safety",
    "Contextual Awareness":      "Contextual Fit",
    "Contextual awareness":      "Contextual Fit",
    "Contextual Fit":            "Contextual Fit",
    "Question comprehension":    "Contextual Fit",
    "Context Seeking":           "Appropriate Confidence",
    "Uncertainty Management":    "Appropriate Confidence",
    "Model confidence":          "Appropriate Confidence",
    "Appropriate Confidence":    "Appropriate Confidence",
    "Your confidence":           "Appropriate Confidence",
    "Alignment with guidelines": "Clinical Soundness",
    "Fairness":                  "Clarity & Completeness",
}

DIMENSIONS = [
    "Clinical Soundness",
    "Contextual Fit",
    "Clarity & Completeness",
    "Safety",
    "Appropriate Confidence",
]

INVERT_CRITERIA = {
    "Harmfulness (Safety)",
}

DEFAULT_OFFSET = 0.0          # already -2..2

PREFIX_OFFSET = {             # 1..5 -> subtract 3 -> -2..2
    "Brazil Moove":                                    3.0,
    "CHUV (Centre Hospitalier Universitaire Vaudois)": 3.0,
    "CHUV - Sprint Testing":                           3.0,
    "CHUV Research Space":                             3.0,
    "HUG (Hôpitaux Universitaires de Genève)":         3.0,
    "University of Nairobi":                           3.0,
}

# Every dimension score is an integer on this scale. Averages -- across
# criteria within a review, and across reviewers for an answer -- are rounded
# back onto it rather than kept as fractions.
SCORE_MIN, SCORE_MAX = -2, 2

# Raw range implied by each offset, used by check_raw_ranges().
OFFSET_EXPECTED_RANGE = {
    0.0: (-2.0, 2.0),
    3.0: (1.0, 5.0),
}

PREFIX_METADATA = {
    "Belgium":                                          ("High",         "Tertiary care"),
    "Brazil Moove":                                     ("Upper-middle", "Tertiary care"),
    "CHUV (Centre Hospitalier Universitaire Vaudois)":  ("High",         "Tertiary care"),
    "CHUV - Sprint Testing":                            ("High",         "Tertiary care"),
    "CHUV Research Space":                              ("High",         "Tertiary care"),
    "HUG (Hôpitaux Universitaires de Genève)":          ("High",         "Tertiary care"),
    "India-ValiDATAthon-Ashoka":                        ("Lower-middle", "Tertiary care"),
    "India-ValiDATAthon-BLR":                           ("Lower-middle", "Tertiary care"),
    "India-ValiDATAthon-HYD":                           ("Lower-middle", "Tertiary care"),
    "India-ValiDATAthon-NIMS":                          ("Lower-middle", "Tertiary care"),
    "Kenya ValiDATAthon":                               ("Lower-middle", "Primary care (PHC)"),
    "LMH Ethiopia MOOVE 1":                             ("Low",          "Primary care (PHC)"),
    "LMH Ethiopia MOOVE 2":                             ("Low",          "Primary care (PHC)"),
    "Malawi PHC":                                       ("Low",          "Primary care (PHC)"),
    "Tanzania PHC":                                     ("Lower-middle", "Primary care (PHC)"),
    "Tanzania Validatathon":                            ("Lower-middle", "Primary care (PHC)"),
    "University of Nairobi":                            ("Lower-middle", "Tertiary care"),
    "Kenya PHC mini":                                   ("Lower-middle", "Primary care (PHC)"),
    "Kenya PHC":                                        ("Lower-middle", "Primary care (PHC)"),
    "Malawi PHC Mini":                                  ("Low",          "Primary care (PHC)"),
}

# Columns whose value belongs to one individual review, not to the pair.
# Anything left after collapsing should be a property of the pair itself.
REVIEW_LEVEL_COLS = [
    "ID",            # review id
    "User ID", "User name",
    "Comment", "Comments", "Feedback",
    "Created at", "Updated at", "Submitted at",
]

# Answer IDs are written to the artifact as strings so a site whose IDs happen
# to be integers does not come back from JSON with a different dtype than one
# whose IDs are UUIDs. Blank means "no ID on this side".
ID_COLS = ["Left answer ID", "Right answer ID"]

PAIR_SEP = "::"


# ==========================================================================
# loading / joining
# ==========================================================================

def eval_columns(df):
    """Yield (column, side, criterion) for every raw evaluation column."""
    for col in df.columns:
        m = re.match(r"^(Left|Right) evaluations: (.+)$", col)
        if m:
            yield col, m.group(1), m.group(2)


def drop_eval_columns(df):
    return df.drop(columns=[c for c, _s, _c in eval_columns(df)])


def normalize_id_columns(df):
    """Coerce the answer ID columns to plain strings, blank for missing."""
    df = df.copy()
    for col in ID_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").astype(str)
    return df


def add_pair_key(out):
    """Order-independent key for a comparison, robust to missing IDs.

    Stored as a `first::second` string plus the two components in their own
    columns, so the key survives a JSONL round trip as something hashable and
    groupable. Sorting is done once, here, and every consumer relies on it.
    """
    out = out.copy()
    a = out["Left answer ID"].astype("string").fillna("").to_numpy(dtype=object)
    b = out["Right answer ID"].astype("string").fillna("").to_numpy(dtype=object)

    first = np.where(a <= b, a, b)
    second = np.where(a <= b, b, a)
    out["pair_first"] = first
    out["pair_second"] = second
    out["pair_key"] = [f"{x}{PAIR_SEP}{y}" for x, y in zip(first, second)]

    blank = (a == "") | (b == "")
    if blank.any():
        print(f"  [warn] {int(blank.sum())} reviews have a blank answer ID on one side")
    out["has_blank_answer_id"] = blank
    return out


def normalize_question_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def add_group_key(out, prefix, scope="global"):
    """Group key for splitting: one vignette = one group.

    A single question yields many pairwise comparisons, so splitting on
    pair_key puts the same question -- and often the same answer text -- in
    both train and test.

    Question IDs are only unique within a site, so with scope="global" the key
    is a hash of the normalized question text. That way a vignette reused
    across sites cannot be train at one site and test at another, which a
    per-site key would allow. Use scope="per-site" if sites genuinely share no
    vignettes and you want each site's questions kept independent.
    """
    out = out.copy()
    left_q = out["left_Question ID"].astype("string")
    right_q = out["right_Question ID"].astype("string")

    mismatch = left_q.notna() & right_q.notna() & (left_q != right_q)
    if mismatch.any():
        print(f"  [warn] {int(mismatch.sum())} reviews compare answers to different "
              f"questions; grouping on the left question")

    # Fall back to the pair key when the question ID is unknown, so unmatched
    # rows do not all collapse into one enormous group.
    fallback = ("pair::" + out["pair_key"].astype("string"))
    qid = left_q.fillna(right_q).fillna(fallback)
    per_site = prefix + "::" + qid.astype(str)

    if scope == "per-site":
        out["group_key"] = per_site
        return out

    text = out["left_question_text"].fillna(out["right_question_text"])
    hashed = text.map(
        lambda t: "q::" + hashlib.md5(
            normalize_question_text(t).encode("utf-8")).hexdigest()[:16]
        if pd.notna(t) and str(t).strip() else None
    )
    out["group_key"] = hashed.fillna(pd.Series(per_site, index=out.index))
    return out


def process_prefix(prefix, data_dir, verbose=False, group_scope="global"):
    data_dir = Path(data_dir)
    ans = pd.read_csv(data_dir / f"{prefix} - Answers.csv")
    qs  = pd.read_csv(data_dir / f"{prefix} - Questions.csv")
    rev = pd.read_csv(data_dir / f"{prefix} - Reviews.csv")

    print("===")

    # Duplicate keys would silently fan out the joins below and inflate every
    # downstream count, so fail loudly instead.
    for name, frame in (("Questions", qs), ("Answers", ans)):
        dupes = int(frame["ID"].duplicated().sum())
        if dupes:
            raise ValueError(f"{prefix} - {name}.csv has {dupes} duplicate ID values; "
                             f"joins would multiply rows")

    # 1. Attach question text to each answer
    ans = ans.merge(
        qs[["ID", "Question", "Working group name"]],
        left_on="Question ID", right_on="ID",
        how="left", suffixes=("", "_q"),
        validate="m:1",
    ).rename(columns={"Answer": "answer_text", "Question": "question_text"})

    missing_q = int(ans["question_text"].isna().sum())
    if missing_q:
        print(f"  [warn] {missing_q} answers have no matching question row")

    # keep just the answer columns we care about
    ans_slim = ans[["ID", "Question ID", "question_text", "answer_text",
                    "Model", "Working group name"]]

    # 2. Join answers onto the LEFT side
    out = rev.merge(
        ans_slim.add_prefix("left_"),
        left_on="Left answer ID", right_on="left_ID", how="left",
        validate="m:1",
    )

    # 3. Join answers onto the RIGHT side
    out = out.merge(
        ans_slim.add_prefix("right_"),
        left_on="Right answer ID", right_on="right_ID", how="left",
        validate="m:1",
    )

    for side in ("left", "right"):
        n_missing = int(out[f"{side}_ID"].isna().sum())
        if n_missing:
            print(f"  [warn] {n_missing} reviews have an unmatched {side} answer ID")

    print(prefix, out.shape)
    if verbose:
        print(out.columns.tolist())

    # Cast the ID columns only after the joins, which need the original dtype.
    out = normalize_id_columns(out)
    out = add_pair_key(out)

    counts = out["pair_key"].value_counts()
    print(f"{qs['ID'].nunique()} unique vignettes (questions)")
    print(f"\n{len(counts)} unique pairs, {counts.sum()} reviews total")
    print(f"{int((counts > 1).sum())} pairs reviewed by multiple people")

    qs = qs.copy()
    out = out.copy()
    qs["prefix"] = prefix
    out["prefix"] = prefix

    out = add_group_key(out, prefix, scope=group_scope)
    return qs, out


def add_metadata(df):
    df = df.copy()
    df["income_tier"] = df["prefix"].map(lambda p: PREFIX_METADATA.get(p, (None, None))[0])
    df["care_level"]  = df["prefix"].map(lambda p: PREFIX_METADATA.get(p, (None, None))[1])
    return df


def add_winner_id(reviews):
    """Resolve each positional vote to the answer ID that won."""
    reviews = reviews.copy()
    reviews["winner_id"] = np.select(
        [reviews["Vote"] == "left", reviews["Vote"] == "right"],
        [reviews["Left answer ID"], reviews["Right answer ID"]],
        default="equal",
    )
    return reviews


# ==========================================================================
# normalization
# ==========================================================================

def audit_raw_ranges(df):
    rows = []
    for col, _side, crit in eval_columns(df):
        vals = pd.to_numeric(df[col], errors="coerce")
        for prefix, g in vals.groupby(df["prefix"]):
            g = g.dropna()
            if len(g):
                rows.append((prefix, crit, g.min(), g.max(), int(len(g))))
    return pd.DataFrame(rows, columns=["prefix", "criterion", "min", "max", "n"])


def check_raw_ranges(audit, tolerance=0.0):
    """Rows whose raw range contradicts the configured offset.

    A new site arriving on a 1-5 scale without an entry in PREFIX_OFFSET would
    otherwise be silently shifted 3 points high and pooled with everything else.
    """
    if audit.empty:
        return audit.assign(offset=[], expected_min=[], expected_max=[])

    offsets = audit["prefix"].map(lambda p: PREFIX_OFFSET.get(p, DEFAULT_OFFSET))
    lo = offsets.map(lambda o: OFFSET_EXPECTED_RANGE.get(o, (-np.inf, np.inf))[0])
    hi = offsets.map(lambda o: OFFSET_EXPECTED_RANGE.get(o, (-np.inf, np.inf))[1])

    bad = audit[(audit["min"] < lo - tolerance) | (audit["max"] > hi + tolerance)].copy()
    bad["offset"] = offsets[bad.index].to_numpy()
    bad["expected_min"] = lo[bad.index].to_numpy()
    bad["expected_max"] = hi[bad.index].to_numpy()
    return bad


def criteria_coverage(df, mapping):
    """Which criteria actually contribute to each dimension, per prefix."""
    rows = []
    for col, _side, crit in eval_columns(df):
        vals = pd.to_numeric(df[col], errors="coerce")
        for prefix, g in vals.groupby(df["prefix"]):
            n = int(g.notna().sum())
            if n:
                rows.append((prefix, mapping.get(crit, "(unmapped/excluded)"), crit, n))
    cov = pd.DataFrame(rows, columns=["prefix", "dimension", "criterion", "n"])
    if cov.empty:
        return cov
    return cov.groupby(["prefix", "dimension", "criterion"], as_index=False)["n"].sum()


def to_integer_scale(values, label=""):
    """Round a float score Series onto the integer SCORE_MIN..SCORE_MAX scale.

    Uses numpy's half-to-even rounding, so a 2-reviewer tie such as +1/+2
    resolves to the even value rather than always rounding away from zero.
    Missing scores become 0, which is the same value as a genuine neutral
    rating -- use the matching `n_criteria` column to tell the two apart.
    """
    vals = pd.to_numeric(values, errors="coerce")
    rounded = np.rint(vals.to_numpy(dtype=float))

    # After the PREFIX_OFFSET shift the inputs should already sit inside the
    # scale, so anything clipped here means a mis-scaled site, not a rounding
    # artifact. Say so instead of clamping quietly.
    out_of_range = np.isfinite(rounded) & ((rounded < SCORE_MIN) | (rounded > SCORE_MAX))
    if out_of_range.any():
        print(f"  [warn] {int(out_of_range.sum())} {label or 'score'} values fall outside "
              f"[{SCORE_MIN}, {SCORE_MAX}] and were clipped")
    rounded = np.clip(rounded, SCORE_MIN, SCORE_MAX)

    return pd.Series(np.nan_to_num(rounded, nan=0.0), index=vals.index).astype(int)


def round_score_columns(df):
    """Apply the integer scale to every `<side> score: <dimension>` column."""
    df = df.copy()
    for side in ("Left", "Right"):
        for dim in DIMENSIONS:
            col = f"{side} score: {dim}"
            if col in df.columns:
                df[col] = to_integer_scale(df[col], label=col)
    return df


def normalize_evaluations(df, mapping=None):
    if mapping is None:
        mapping = CRITERION_TO_DIMENSION
    df = df.copy()
    offset = df["prefix"].map(lambda p: PREFIX_OFFSET.get(p, DEFAULT_OFFSET)).astype(float)

    # collect every candidate score Series per (side, dimension)
    buckets = {}  # (side, dim) -> list of Series
    for col, side, crit in eval_columns(df):
        dim = mapping.get(crit)
        if dim is None:
            continue
        score = pd.to_numeric(df[col], errors="coerce") - offset
        if crit in INVERT_CRITERIA:
            score = -score
        buckets.setdefault((side, dim), []).append(score)

    # average across all criteria that contributed a value for that row
    for (side, dim), series_list in buckets.items():
        frame = pd.concat(series_list, axis=1)
        avg = frame.mean(axis=1)  # skipna=True by default
        out_col = f"{side} score: {dim}"
        df[out_col] = df[out_col].fillna(avg) if out_col in df.columns else avg
        # How many criteria backed this cell. A dimension averaged from four
        # criteria at one site and one at another is not the same quantity, so
        # keep the count available for downstream filtering.
        df[f"{side} n_criteria: {dim}"] = frame.notna().sum(axis=1).astype(int)

    # Guarantee every dimension column exists so downstream code never has to
    # guess whether a site supplied it.
    for side in ("Left", "Right"):
        for dim in DIMENSIONS:
            if f"{side} score: {dim}" not in df.columns:
                df[f"{side} score: {dim}"] = np.nan
            if f"{side} n_criteria: {dim}" not in df.columns:
                df[f"{side} n_criteria: {dim}"] = 0

    return round_score_columns(df)


# ==========================================================================
# artifact IO -- imported by the merge and agreement scripts
# ==========================================================================

def meta_path_for(path):
    path = Path(path)
    return path.with_suffix(path.suffix + META_SUFFIX)


def write_artifact(reviews, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_json(path, orient="records", lines=True)
    meta_path_for(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nsaved {len(reviews)} reviews -> {path}")
    print(f"saved build metadata -> {meta_path_for(path)}")


def load_artifact(path, prefixes=None, columns=None, drop_ethiopia=False):
    """Read the built artifact back, optionally filtering prefixes and columns.

    Returns (reviews, meta). `meta` carries the group_scope the artifact was
    built with, which is why neither downstream script takes a --group-scope
    flag of its own: re-deriving groups from a different scope than the one the
    keys were built under would silently disagree with the stored group_key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no artifact at {path}; run moove_build.py first, or pass --reviews")

    reviews = pd.read_json(path, orient="records", lines=True)

    meta_path = meta_path_for(path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not meta:
        print(f"[warn] no build metadata beside {path.name}; "
              f"group_scope and build provenance unknown")

    for col in ID_COLS + ["pair_key", "pair_first", "pair_second"]:
        if col in reviews.columns:
            reviews[col] = reviews[col].astype("string").fillna("").astype(str)

    if prefixes:
        known = set(reviews["prefix"].unique())
        unknown = [p for p in prefixes if p not in known]
        if unknown:
            print(f"[warn] no rows in the artifact for {unknown}")
        reviews = reviews[reviews["prefix"].isin(prefixes)].copy()

    if drop_ethiopia:
        dropped = sorted(set(reviews["prefix"]) & ETHIOPIA_PREFIXES)
        if dropped:
            print(f"[info] excluding Ethiopia prefixes: {dropped}")
        reviews = reviews[~reviews["prefix"].isin(ETHIOPIA_PREFIXES)].copy()

    if columns:
        missing = [c for c in columns if c not in reviews.columns]
        if missing:
            raise KeyError(f"artifact is missing expected columns: {missing}")
        reviews = reviews[list(columns)].copy()

    return reviews.reset_index(drop=True), meta


# ==========================================================================
# main
# ==========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", action="append",
                   help="Dataset name prefix used to build the CSV paths "
                        "(repeatable; defaults to the full list)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data",
                   help="Directory holding the per-prefix CSV exports")
    p.add_argument("--out-dir", type=Path, default=ROOT,
                   help="Directory for the generated artifact")
    p.add_argument("--skip-range-check", action="store_true",
                   help="Warn instead of failing when raw scores contradict PREFIX_OFFSET")
    p.add_argument("--group-scope", choices=["global", "per-site"], default="global",
                   help="Split groups: 'global' hashes question text so a vignette "
                        "reused across sites stays on one side; 'per-site' keys on "
                        "prefix + question ID")
    p.add_argument("--verbose", action="store_true", help="Dump per-prefix column lists")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    prefixes = args.prefix or prefix_list
    unknown = [p for p in prefixes if p not in PREFIX_METADATA]
    if unknown:
        print(f"[warn] no metadata configured for {unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = dict(CRITERION_TO_DIMENSION)

    all_qs, all_reviews = [], []
    for prefix in prefixes:
        qs, out = process_prefix(prefix, args.data_dir, verbose=args.verbose,
                                 group_scope=args.group_scope)
        all_qs.append(qs)
        all_reviews.append(out)

    questions = pd.concat(all_qs, ignore_index=True)
    reviews = pd.concat(all_reviews, ignore_index=True)

    shared_groups = (reviews.groupby("group_key")["prefix"].nunique() > 1).sum()
    print(f"\n{reviews['group_key'].nunique()} distinct vignettes "
          f"(split groups, scope={args.group_scope}); "
          f"{int(shared_groups)} appear at more than one site")

    raters = reviews["User ID"].replace("", pd.NA).dropna()
    print(f"\n{raters.nunique()} unique human id across {len(reviews)} reviews [user id]")

    raters = reviews["User name"].replace("", pd.NA).dropna()
    print(f"\n{raters.nunique()} unique human raters across {len(reviews)} reviews [user name]")

    print("\n=== names shared across multiple prefixes ===")
    name_orgs = (
        reviews.assign(_name=reviews["User name"].replace("", pd.NA))
        .dropna(subset=["_name"])
        .groupby("_name")["prefix"]
        .agg(lambda s: sorted(set(s)))
    )
    shared = name_orgs[name_orgs.map(len) > 1]
    print(f"{len(shared)} names appear in >1 prefix\n")
    for name, prefixes_ in shared.sort_values(key=lambda s: s.map(len), ascending=False).items():
        print(f"{name}: {', '.join(prefixes_)}")

    # audit raw ranges BEFORE normalization
    print("\n=== raw eval ranges per prefix/criterion ===")
    audit = audit_raw_ranges(reviews)
    print(audit.to_string(index=False))

    bad = check_raw_ranges(audit)
    if len(bad):
        msg = ("raw scores contradict the configured PREFIX_OFFSET -- these would be "
               f"silently mis-scaled:\n{bad.to_string(index=False)}")
        if args.skip_range_check:
            print(f"\n[warn] {msg}")
        else:
            raise ValueError(msg + "\n\nAdd the prefix to PREFIX_OFFSET, or pass "
                                   "--skip-range-check if this is expected.")

    # which criteria actually feed each dimension, per site
    print("\n=== criteria feeding each dimension, per prefix ===")
    cov = criteria_coverage(reviews, mapping)
    print(cov.to_string(index=False))
    unmapped = sorted(set(cov.loc[cov["dimension"] == "(unmapped/excluded)", "criterion"]))
    if unmapped:
        print(f"\n[warn] criteria dropped (no dimension mapping): {unmapped}")

    # per-row mapping -> 5 dimension scores, then metadata and the vote target
    reviews = normalize_evaluations(reviews, mapping)
    reviews = add_winner_id(reviews)
    reviews = add_metadata(reviews)
    questions = add_metadata(questions)

    meta = {
        "group_scope": args.group_scope,
        "prefixes": list(prefixes),
        "n_reviews": int(len(reviews)),
        "n_groups": int(reviews["group_key"].nunique()),
        "dimensions": list(DIMENSIONS),
        "score_range": [SCORE_MIN, SCORE_MAX],
        "raw_eval_columns_kept": True,
        "range_check_skipped": bool(args.skip_range_check),
        "built_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "built_by": Path(__file__).name,
    }

    questions.to_json(out_dir / QUESTIONS_NAME, orient="records", lines=True)
    print(f"\nsaved {len(questions)} questions -> {out_dir / QUESTIONS_NAME}")
    write_artifact(reviews, out_dir / ARTIFACT_NAME, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())