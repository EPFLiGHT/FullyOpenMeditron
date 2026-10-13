#!/usr/bin/env python3
# truncation/normalize_schema.py

"""Give every row in a regenerated JSONL the same keys.

Why
---
`regenerate_truncated.py` adds bookkeeping fields (`regen`, `regen_tries`,
`regen_stated_letter`, `label_letter_before_regen`, `finish_reason`) to the
rows it rewrote and to no others. That is fine for a JSONL read line by line,
but HuggingFace `datasets` -- which axolotl uses -- infers an Arrow schema
from the first chunk of a file and then hard-fails when a later chunk carries
columns the schema does not have:

    datasets.exceptions.DatasetGenerationCastError: ... at some point there
    are 5 new columns ({'finish_reason', 'regen_tries', 'regen_stated_letter',
    'regen', 'label_letter_before_regen'})

Caught by the 8B trial run on a debug node (job 3327839, failed 4m40s in).
The 70B run would have failed identically, 5 minutes into an 8-node
reservation.

What it does
------------
Two passes per file. The first collects the union of keys and, for each, the
type of the first non-null value seen. The second rewrites every row with the
full key set, filling absences with a typed default -- False for bool, 0 for
int, None for everything else -- so Arrow infers one nullable column rather
than a schema change mid-file.

Filling rather than stripping is deliberate: the bookkeeping is the record of
which answers were regenerated and how, and it is worth keeping in the file
that gets trained on. Nothing that was present is altered; only absences are
filled. Key order is the union in first-seen order, so a diff stays readable.

Writes to a sibling temp file and renames, so an interrupted run cannot leave
a half-written corpus behind -- these files are the training input for an
8-node job.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Fields regenerate_truncated.py adds. A null in one of these means only "this
# row was not regenerated", never anything a reader needs, so they are always
# coerced to a typed empty regardless of where their first real value sits.
# That removes any dependence on guessing Arrow's inference window -- the
# window is smaller than row 4443, which is all the failed run told us, and
# guessing it exactly is not something this should rest on.
REGEN_FIELDS = {"regen", "regen_tries", "regen_stated_letter",
                "label_letter_before_regen", "finish_reason"}

DEFAULT_INPUTS = [
    "data/fixed/gpt_oss_8retries_v3_fixed.jsonl",
    "data/fixed/meditron_4_synthetic_qa_v3_fixed.jsonl",
    "data/fixed/guidelines_qa_v3_full_fixed.jsonl",
    "data/fixed/synthetic_moove_v2_gpt_oss_no_truncation_fixed.jsonl",
]


def default_for(kind):
    """A same-typed empty value, so Arrow infers the real type from row 0.

    None is NOT usable as a filler for a typed column. Arrow infers from the
    first block of the file: if every value there is null the column becomes
    type `null`, and the first real value later in the file fails with

        TypeError: Couldn't cast array of type string to null

    which is exactly how the second trial (job 3327881) died -- the filled
    columns' first real value sits at row 4443 / 15386 / 45, far past the
    inference window. Empty-but-typed fillers keep the column typed from the
    start. None survives only for a column with no non-null value anywhere,
    where `null` is genuinely the right type.
    """
    if kind is bool:
        return False
    if kind is int:
        return 0
    if kind is float:
        return 0.0
    if kind is str:
        return ""
    if kind is list:
        return []
    if kind is dict:
        return {}
    return None                      # no non-null value anywhere in the file


def scan(path, block=1000):
    """(ordered keys, {key: type}, {late-typed keys}) over the whole file.

    A key is "late-typed" when its first non-null value falls outside the
    leading `block` rows Arrow infers from. Those are the columns that must
    have their nulls coerced to a typed empty; everywhere else a null is left
    alone, because in the pre-existing columns (label_letter, label_text,
    exact_match) it carries meaning and Arrow already types them correctly
    from row 0.
    """
    order, kinds, first_nonnull = [], {}, {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            for k, v in row.items():
                if k not in kinds:
                    order.append(k)
                    kinds[k] = None
                if v is not None:
                    if kinds[k] is None:
                        kinds[k] = type(v)
                    first_nonnull.setdefault(k, i)
    late = {k for k in order
            if first_nonnull.get(k, 10 ** 18) >= block and kinds[k] is not None}
    # Always coerce the fields we introduced, wherever their first value sits.
    late |= {k for k in order if k in REGEN_FIELDS and kinds[k] is not None}
    return order, kinds, late


def normalize(path, dry_run=False):
    order, kinds, late = scan(path)
    fill = {k: default_for(kinds[k]) for k in order}

    def value(row, k):
        """Present value, except that a null in a late-typed column becomes
        the typed empty -- that column is the one Arrow would otherwise infer
        as `null` and then fail to cast a real value into."""
        v = row.get(k)
        if v is None and k in late:
            return fill[k]
        return v if k in row else fill[k]

    src = Path(path)
    tmp = src.with_suffix(src.suffix + ".normalizing")
    n = changed = 0

    if dry_run:
        with open(src, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n += 1
                if len(row) != len(order) or any(
                        row.get(k) is None for k in late):
                    changed += 1
        return n, changed, order, late

    with open(src, encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            if len(row) != len(order) or any(row.get(k) is None for k in late):
                changed += 1
            out = {k: value(row, k) for k in order}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    os.replace(tmp, src)             # atomic; never a half-written corpus
    return n, changed, order, late


def verify(path, block=5000):
    """One schema for the whole file, and no column that is all-null across
    the leading block while carrying a real value later -- the shape that
    makes Arrow infer `null` and then fail to cast into it.

    A direct load with `datasets` would be the better check, but it is not
    installed on the login node, so this reproduces the specific inference
    rule that bit us.
    """
    seen = set()
    head_nonnull, ever_nonnull = set(), set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add(tuple(sorted(row.keys())))
            for k, v in row.items():
                if v is not None:
                    ever_nonnull.add(k)
                    if i < block:
                        head_nonnull.add(k)
    late = sorted(ever_nonnull - head_nonnull)
    return seen, late


def main():
    p = argparse.ArgumentParser(description="Uniform JSONL schema for axolotl.")
    p.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    bad = False
    for path in args.inputs:
        if not Path(path).exists():
            print(f"MISSING {path}")
            bad = True
            continue
        n, changed, order, late = normalize(path, args.dry_run)
        verb = "would fix" if args.dry_run else "fixed"
        print(f"{path}\n  {n} rows, {verb} {changed}, {len(order)} columns"
              + (f", late-typed {sorted(late)}" if late else ""))
        if not args.dry_run:
            schemas, late = verify(path)
            ok = len(schemas) == 1 and not late
            print(f"  schemas after: {len(schemas)}"
                  f"{'' if not late else f'; LATE-TYPED columns {late}'}"
                  f"  {'OK' if ok else 'NOT SAFE TO TRAIN ON'}")
            if not ok:
                bad = True
    if bad:
        sys.exit(1)
    if not args.dry_run:
        print("\nAll files carry one schema. Delete any stale "
              "dataset_prepared_path before retraining -- axolotl caches the "
              "tokenised dataset and will happily reuse the pre-fix one.")


if __name__ == "__main__":
    main()
