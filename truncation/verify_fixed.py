#!/usr/bin/env python3
# truncation/verify_fixed.py
"""
Integrity check on the corpora regenerate_truncated.py wrote.

Run before retraining. Confirms the fix did exactly what it claimed and nothing
else: untouched rows byte-identical, questions preserved, answers actually
replaced and properly terminated, no harmony markers leaked into an assistant
turn, and no row regenerated that the audit had not flagged.

usage: python3 truncation/verify_fixed.py [--outdir data/fixed]
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regenerate_truncated import (DEFAULT_INPUTS, DEFAULT_FLAGS, load_flags, scan,
                                  assistant_slots, chosen_letter, is_mcq, _h)

HARMONY = ("<|channel|>", "<|start|>", "<|message|>", "<|end|>", "<|return|>")
ENDINGS = (".", "!", "?", ":", ")", '"', "*", "`", "—")

# How far a regenerated group's answer-format rate may fall below the rate of
# the untouched rows it sits alongside. Regenerating from the bare question
# once cost meditron_4_synthetic_qa_v3 its MCQ format -- 93.6% of the corpus
# ends on an option letter, the regenerated rows managed 0.6% -- and nothing
# in this file noticed. It does now.
FORMAT_TOLERANCE = 0.15


def walk_pairs(src, dst, drop):
    """Yield (line_no, source row, output row) by streaming both files at once.

    Line numbers follow scan() and apply_fixes(): enumerate() over the raw file
    with `continue` on blank lines, so a blank line still consumes an index.
    Indexing a dense list instead would silently shift every row after the
    first blank line.

    Streamed rather than loaded: guidelines_qa_v3_full alone is 1.18GB on disk
    and several times that once parsed, and this walks four such corpora.
    apply_fixes() writes kept rows in source order, so the two files advance in
    lockstep with dropped rows skipped on the source side only.
    """
    with open(src, encoding="utf-8") as fs, open(dst, encoding="utf-8") as fd:
        dst_iter = (l for l in fd if l.strip())
        for line_no, line in enumerate(fs):
            if not line.strip() or line_no in drop:
                continue
            out = next(dst_iter, None)
            if out is None:
                return                      # output ran short; the count check reports it
            yield line_no, json.loads(line), json.loads(out)


def count_lines(path, drop=frozenset()):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line.strip() and line_no not in drop:
                n += 1
    return n


def check_dataset(src, outdir, flagged, chk):
    name = Path(src).stem
    dst = Path(outdir) / f"{name}_fixed.jsonl"
    print(f"\n=== {name} ===")
    if not dst.exists():
        chk(False, f"{name}: {dst} missing")
        return

    jobs, drop, _, _ = scan(src, flagged[name])
    n_src = count_lines(src)
    n_kept = count_lines(src, drop)
    n_out = count_lines(dst)

    chk(n_out == n_kept,
        f"{name}: output rows {n_out} == kept {n_kept} "
        f"(source {n_src} - dropped {len(drop)})")
    if n_out != n_kept:
        return                      # any positional check below would be noise

    regen = touched = altered_q = unchanged_a = leaked = empty = extra = ends = 0
    old_len, new_len = [], []
    # answer-format rate, split by whether the row answers a multiple choice
    # question and whether we rewrote it
    fmt = {(g, k): [0, 0] for g in ("clean", "regen") for k in ("mcq", "free")}

    for ln, s, d in walk_pairs(src, dst, drop):
        # Judge MCQ-ness from the SOURCE row: regeneration overwrites
        # label_letter and nulls it when the new answer states no letter --
        # exactly the rows this check exists to catch. is_mcq() also respects
        # the is_labeled flag, so open-ended rows are not measured against an
        # MCQ norm they were never meant to meet.
        was_mcq = is_mcq(s)
        grp = "regen" if d.get("regen") else "clean"
        kind = "mcq" if was_mcq else "free"
        for _, _, a in assistant_slots(d["conversations"]):
            fmt[(grp, kind)][1] += 1
            if chosen_letter(a):
                fmt[(grp, kind)][0] += 1

        if not d.get("regen"):
            if d != s:
                touched += 1
            continue
        regen += 1

        s_slots, d_slots = assistant_slots(s["conversations"]), assistant_slots(d["conversations"])
        s_users = [t.get("value") for t in s["conversations"] if t.get("from") in ("human", "user")]
        d_users = [t.get("value") for t in d["conversations"] if t.get("from") in ("human", "user")]
        if s_users != d_users:
            altered_q += 1

        for (_, _, sa), (_, _, da) in zip(s_slots, d_slots):
            old_len.append(len(sa))
            new_len.append(len(da))
            if sa == da:
                unchanged_a += 1
            if any(m in da for m in HARMONY):
                leaked += 1
            if not da.strip():
                empty += 1
            if da.rstrip().endswith(ENDINGS):
                ends += 1

        # the row must be one the audit actually flagged
        if not (s.get("finish_reason") == "length"
                or any(_h(a) in flagged[name] for _, _, a in s_slots)):
            extra += 1

    chk(touched == 0, f"{name}: untouched rows unchanged ({touched} differ)")
    chk(altered_q == 0, f"{name}: questions preserved ({altered_q} altered)")
    chk(unchanged_a == 0, f"{name}: answers actually replaced ({unchanged_a} unchanged)")
    chk(leaked == 0, f"{name}: no harmony markers in answers ({leaked} leaked)")
    chk(empty == 0, f"{name}: no empty answers ({empty})")
    chk(extra == 0, f"{name}: only audit-flagged rows regenerated ({extra} extra)")
    chk(regen == len(jobs), f"{name}: regenerated {regen} == expected {len(jobs)}")

    print("  answer format (fraction ending on an option letter):")
    for kind in ("mcq", "free"):
        ch, ct = fmt[("clean", kind)]
        rh, rt = fmt[("regen", kind)]
        if not rt:
            continue
        clean_rate = ch / ct if ct else None
        regen_rate = rh / rt
        base = f"{clean_rate:.1%} clean (n={ct})" if clean_rate is not None else "no clean rows"
        print(f"    {kind}: {regen_rate:.1%} regenerated (n={rt}) vs {base}")
        if clean_rate is not None:
            # Two-sided on purpose. Too FEW letters means an MCQ row lost its
            # format; too MANY means the MCQ contract was applied to free-form
            # rows that should never state one. Both have happened here, and a
            # one-sided check only ever caught the first.
            chk(abs(regen_rate - clean_rate) <= FORMAT_TOLERANCE,
                f"{name}: {kind} answer format {regen_rate:.1%} within "
                f"{FORMAT_TOLERANCE:.0%} of the corpus norm {clean_rate:.1%}")

    if new_len:
        # A regenerated group far longer than the rows around it means the
        # model rambled instead of answering; that is how the lost MCQ format
        # first showed up.
        chk(statistics.median(new_len) < 4 * statistics.median(old_len) + 2000,
            f"{name}: regenerated length in line with the originals")
        print(f"  answer chars: old median={statistics.median(old_len):.0f} "
              f"new median={statistics.median(new_len):.0f}")
        print(f"  new answers ending in sentence punctuation: "
              f"{ends}/{len(new_len)} ({100 * ends / len(new_len):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/fixed")
    ap.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--flags", default=DEFAULT_FLAGS)
    a = ap.parse_args()

    flagged = load_flags(a.flags)
    fails = []

    def chk(cond, msg):
        print(("  OK   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    for src in a.input:
        check_dataset(src, a.outdir, flagged, chk)

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED:"))
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
