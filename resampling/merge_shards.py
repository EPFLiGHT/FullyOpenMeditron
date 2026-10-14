#!/usr/bin/env python3
# resampling/merge_shards.py
"""
Join the per-shard outputs of resample_answers.py back into
one file per dataset, in the input's row order.

Shards are disjoint by construction (row i belongs to shard i % N), and every
row carries `source_row`, so the merge is a sort rather than a guess. The merge
refuses to write a file when a shard is missing or a row appears twice: a
quietly short corpus is worse than no corpus.

No GPU, no model. Run it on the login node after the shards finish.

usage:
python3 resampling/merge_shards.py --outdir $SCRATCH/meditronfo/resampling/answers
    --shards 8        how many shards to expect (default: infer from filenames)
    --keep-source-row leave the source_row column in the merged file
writes, for each dataset found:
    <outdir>/<name>.jsonl        the merged corpus
    <outdir>/merge_report.csv    rows per shard, total, gaps
"""

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, write_jsonl                      # noqa: E402

SHARD = re.compile(r"^(?P<stem>.+)\.shard(?P<i>\d+)of(?P<n>\d+)\.jsonl$")
SKIP = {"checkpoint"}        # resume state, not corpus: never merged


def group_shards(outdir):
    """{stem: {shard index: path}} for every sharded jsonl in outdir."""
    groups = collections.defaultdict(dict)
    counts = {}
    for path in sorted(Path(outdir).glob("*.shard*of*.jsonl")):
        m = SHARD.match(path.name)
        if not m:
            continue
        stem, i, n = m["stem"], int(m["i"]), int(m["n"])
        if stem in SKIP:
            continue
        if counts.setdefault(stem, n) != n:
            raise SystemExit(f"{stem}: shards from two different runs "
                             f"({counts[stem]} and {n}); clean the directory first")
        if i in groups[stem]:
            raise SystemExit(f"{stem}: shard {i} appears twice")
        groups[stem][i] = path
    return groups, counts


def merge(stem, shards, nshards, outdir, keep_source_row, expect_all=True):
    missing = [i for i in range(nshards) if i not in shards]
    if missing and expect_all:
        raise SystemExit(f"{stem}: missing shard(s) {missing} of {nshards}; "
                         f"rerun them or pass --allow-missing to merge anyway")
    rows, per_shard = [], {}
    for i in sorted(shards):
        part = read_jsonl(shards[i])
        per_shard[i] = len(part)
        rows.extend(part)
    # kept rows carry source_row; rejected rows carry the row index as "row"
    key = next((k for k in ("source_row", "row") if all(k in r for r in rows)), None)
    if key:
        seen = collections.Counter(r[key] for r in rows)
        dupes = [k for k, c in seen.items() if c > 1]
        if dupes:
            raise SystemExit(f"{stem}: {len(dupes)} row(s) in more than one shard, "
                             f"e.g. {key}={dupes[:3]}")
        rows.sort(key=lambda r: r[key])
    elif rows:
        print(f"  note: {stem} has no row index; shards concatenated in shard order")
    if not keep_source_row:
        for r in rows:
            r.pop("source_row", None)
    out = Path(outdir) / f"{stem}.jsonl"
    write_jsonl(rows, out)
    return {"file": stem, "shards_found": len(shards), "shards_expected": nshards,
            "missing": ",".join(map(str, missing)), "rows": len(rows),
            "per_shard": json.dumps(per_shard)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir")
    p.add_argument("--shards", type=int, default=0, help="expected shard count")
    p.add_argument("--keep-source-row", action="store_true")
    p.add_argument("--allow-missing", action="store_true",
                   help="merge what is there; the report records the gaps")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()
    if args.selfcheck:
        return _selfcheck()

    if not args.outdir:
        raise SystemExit("--outdir is required")
    groups, counts = group_shards(args.outdir)
    if not groups:
        raise SystemExit(f"no *.shardIofN.jsonl files in {args.outdir}")
    report = []
    for stem, shards in sorted(groups.items()):
        n = args.shards or counts[stem]
        rec = merge(stem, shards, n, args.outdir, args.keep_source_row,
                    expect_all=not args.allow_missing)
        report.append(rec)
        print(f"  {stem}: {rec['rows']} rows from {rec['shards_found']}/{n} shards")
    with open(Path(args.outdir) / "merge_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    print(f"--- wrote {args.outdir}/merge_report.csv ---")


def _selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_jsonl([{"source_row": 0, "v": "a"}, {"source_row": 2, "v": "c"}],
                    d / "set.shard0of2.jsonl")
        write_jsonl([{"source_row": 1, "v": "b"}], d / "set.shard1of2.jsonl")
        groups, counts = group_shards(d)
        rec = merge("set", groups["set"], 2, d, keep_source_row=False)
        rows = read_jsonl(d / "set.jsonl")
        assert [r["v"] for r in rows] == ["a", "b", "c"], rows      # input order restored
        assert "source_row" not in rows[0] and rec["rows"] == 3
        # a missing shard must stop the merge, not shorten the corpus
        try:
            merge("set", {0: d / "set.shard0of2.jsonl"}, 2, d, False)
            raise AssertionError("missing shard was not caught")
        except SystemExit as e:
            assert "missing shard" in str(e)
        # so must a row claimed by two shards
        write_jsonl([{"source_row": 0, "v": "dup"}], d / "set.shard1of2.jsonl")
        groups, _ = group_shards(d)
        try:
            merge("set", groups["set"], 2, d, False)
            raise AssertionError("duplicate row was not caught")
        except SystemExit as e:
            assert "more than one shard" in str(e)
    print("selfcheck ok")


if __name__ == "__main__":
    main()
