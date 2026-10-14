#!/usr/bin/env python3
"""Compare the judge's view of the old corpus with the clean one, item by item.

    python corpus_audit/compare_clean.py \
        --old corpus_audit/judge_outputs/judged_pro_800.jsonl \
        --new corpus_audit/clean/judged_clean_800.jsonl \
        --items corpus_audit/clean/items_800.jsonl \
        --audit corpus_audit/data/meditronfo_audit_2026-07-25.csv \
        --audit-items corpus_audit/clean/items_audit60.jsonl \
        --audit-old corpus_audit/clean/judged_rejudge_audit60_old.jsonl \
        --audit-new corpus_audit/clean/judged_clean_audit60.jsonl

The clean sample is the old 800 found again in the clean sets (build_clean_items.py), so every
comparison is paired on the question. Unchanged answers measure the judge's run-to-run drift: an
old-vs-clean difference no larger than that drift is not a data effect.

Defect = any criterion <= -1, the screen validated against the clinician. The estimated clinician
reject rate is the Rogan-Gladen correction with the screen's sensitivity and specificity measured
again on the 60 audited items (--audit-old). Its CI resamples the audit as well as the corpus
sample, so the uncertainty of the screen itself is carried through. Stdlib only.
"""

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from math import comb

from analyze_corpus import CRITERIA, THEMES, wilson

BOOT = 2000
TRUNC = THEMES["truncated answer"]
# training rows per set, to weight subset rates into a corpus rate: wc -l of the old files, and the
# rows kept in tech_reports/meditronfo_clean_report.pdf
ROWS = {
    "old": {"Curated QA": 216719, "Synthetic QA": 214654, "Guidelines QA": 145681, "Synthetic MOOVE": 24465},
    "clean": {"Curated QA": 212172, "Synthetic QA": 214589, "Guidelines QA": 145360, "Synthetic MOOVE": 24454},
    # meditronfo_gptoss_v1 (data/corpora.yaml)
    "v1": {"Curated QA": 195538, "Synthetic QA": 184540, "Guidelines QA": 129767, "Synthetic MOOVE": 24043},
}


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "scores" in r:
                    out[r["id"]] = r
    return out


def flag(r):
    return min(r["scores"].values()) <= -1


def mean(r):
    return statistics.fmean(r["scores"].values())


def truncated(r):
    return any(k in (r.get("notes") or "").lower() for k in TRUNC)


def rg(apparent, sens, spec):
    """Rogan-Gladen, clamped at 0. None when the screen carries no information."""
    d = sens + spec - 1
    return None if d <= 0 else min(1.0, max(0.0, (apparent + spec - 1) / d))


def screen(audit):
    """sensitivity, specificity of the screen from (flagged, clinician rejected) pairs."""
    pos = [f for f, rej in audit if rej]
    neg = [f for f, rej in audit if not rej]
    if not pos or not neg:
        return None
    return sum(pos) / len(pos), sum(not f for f in neg) / len(neg)


def rg_ci(flags, audit, rng, reps=BOOT):
    vals = []
    for _ in range(reps):
        ss = screen(rng.choices(audit, k=len(audit)))
        if ss is None:
            continue
        f = rng.choices(flags, k=len(flags))
        p = rg(sum(f) / len(f), *ss)
        if p is not None:
            vals.append(p)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def paired_ci(pairs, rng, reps=BOOT):
    """Percentile CI of mean(clean) - mean(old) over paired 0/1 or score values."""
    d = []
    for _ in range(reps):
        s = rng.choices(pairs, k=len(pairs))
        d.append(statistics.fmean(b - a for a, b in s))
    d.sort()
    return d[int(0.025 * reps)], d[int(0.975 * reps)]


def mcnemar(b, c):
    """Exact two-sided McNemar p on the discordant counts."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} {100 * k / n:5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}]" if n else "n/a"


def section(t):
    print(f"\n{t}\n{'-' * len(t)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="judged_pro_800.jsonl")
    ap.add_argument("--new", required=True, help="the judged clean items")
    ap.add_argument("--items", required=True, help="items_800.jsonl from build_clean_items.py")
    ap.add_argument("--audit", required=True, help="the clinician CSV")
    ap.add_argument("--audit-items", required=True, help="items_audit60.jsonl from build_clean_items.py")
    ap.add_argument("--audit-old", required=True, help="the 60 audited items as audited, judged again")
    ap.add_argument("--audit-new", required=True, help="the 60 audited items as they are in the clean sets")
    ap.add_argument("--seed", type=int, default=2001)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    old, new = load(args.old), load(args.new)
    items = [json.loads(ln) for ln in open(args.items, encoding="utf-8") if ln.strip()]
    subsets = list(dict.fromkeys(i["subset"] for i in items))
    pairs = [(i, old[i["old_id"]], new[i["id"]]) for i in items
             if i["status"] != "removed" and i["old_id"] in old and i["id"] in new]

    section("Coverage")
    st = Counter((i["subset"], i["status"]) for i in items)
    for s in subsets:
        n = sum(1 for i, _, _ in pairs if i["subset"] == s)
        print(f"{s:<16} unchanged {st[s, 'unchanged']:>3}  regenerated {st[s, 'regenerated']:>3}  "
              f"removed {st[s, 'removed']:>3}  | paired and graded {n}")
    miss = sum(1 for i in items if i["status"] != "removed") - len(pairs)
    if miss:
        print(f"WARNING: {miss} kept items have no grade on one side; rerun the judge to retry them")
    print("judges: old " + ", ".join(sorted({r['model'] for r in old.values()}))
          + " | clean " + ", ".join(sorted({r['model'] for r in new.values()})))
    for i in items:
        if i["status"] == "removed" and i["old_id"] in old:
            r = old[i["old_id"]]
            print(f"removed: {i['old_id']:<24} old flag {'yes' if flag(r) else 'no '}  {(r.get('notes') or '')[:90]}")

    section("Judge drift on unchanged answers (same item, same answer, judged twice)")
    same = [(a, b) for i, a, b in pairs if i["status"] == "unchanged"]
    if same:
        n = len(same)
        ident = sum(1 for a, b in same if a["scores"] == b["scores"])
        print(f"n = {n}; all five scores identical in {ident}/{n} ({100 * ident / n:.0f}%)")
        for c in CRITERIA:
            ex = sum(1 for a, b in same if a["scores"][c] == b["scores"][c])
            d = statistics.fmean(b["scores"][c] - a["scores"][c] for a, b in same)
            print(f"  {c:<24} exact {100 * ex / n:4.0f}%   mean shift {d:+.2f}")
        ff = Counter((flag(a), flag(b)) for a, b in same)
        print(f"flag: both {ff[True, True]}, neither {ff[False, False]}, "
              f"old only {ff[True, False]}, clean only {ff[False, True]}  "
              f"(McNemar p = {mcnemar(ff[True, False], ff[False, True]):.2f})")
        for s in subsets:
            ss = [(a, b) for i, a, b in pairs if i["status"] == "unchanged" and i["subset"] == s]
            if ss:
                k0, k1 = sum(flag(a) for a, _ in ss), sum(flag(b) for _, b in ss)
                print(f"  {s:<16} flagged old {rate(k0, len(ss))}   again {rate(k1, len(ss))}")

    section("Defect rate per subset: old run vs clean, paired on the question")
    print(f"{'subset':<16}{'old (all 200)':>27}{'clean':>27}{'paired diff [95% CI]':>26}  flips old->clean  McNemar p")
    for s in subsets:
        sp = [(a, b) for i, a, b in pairs if i["subset"] == s]
        allold = [r for r in old.values() if r["subset"] == s]
        if not sp:
            continue
        k0 = sum(flag(r) for r in allold)
        k1 = sum(flag(b) for _, b in sp)
        fl = [(int(flag(a)), int(flag(b))) for a, b in sp]
        d = statistics.fmean(b - a for a, b in fl)
        lo, hi = paired_ci(fl, rng)
        fixed = sum(1 for a, b in fl if a and not b)
        broke = sum(1 for a, b in fl if b and not a)
        print(f"{s:<16}{rate(k0, len(allold)):>27}{rate(k1, len(sp)):>27}"
              f"{f'{100 * d:+5.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]':>26}"
              f"  -{fixed:<3} +{broke:<3}        {mcnemar(fixed, broke):.3f}")

    section("Regenerated answers only (same question, new answer)")
    for s in subsets:
        rp = [(a, b) for i, a, b in pairs if i["status"] == "regenerated" and i["subset"] == s]
        if not rp:
            continue
        n = len(rp)
        m0, m1 = statistics.fmean(mean(a) for a, _ in rp), statistics.fmean(mean(b) for _, b in rp)
        lo, hi = paired_ci([(mean(a), mean(b)) for a, b in rp], rng)
        print(f"{s:<16} n={n:<4} flagged {sum(flag(a) for a, _ in rp):>3} -> {sum(flag(b) for _, b in rp):<3}"
              f"  mean score {m0:+.2f} -> {m1:+.2f} (diff {m1 - m0:+.2f} [{lo:+.2f}, {hi:+.2f}])"
              f"  notes say truncated {sum(map(truncated, (a for a, _ in rp)))} -> "
              f"{sum(map(truncated, (b for _, b in rp)))}")

    section("Mean score per criterion, old -> clean (paired items)")
    print(f"{'':<24}" + "".join(f"{s[:16]:>18}" for s in subsets))
    for c in CRITERIA:
        cells = []
        for s in subsets:
            sp = [(a, b) for i, a, b in pairs if i["subset"] == s]
            if not sp:
                cells.append("—")
                continue
            cells.append(f"{statistics.fmean(a['scores'][c] for a, _ in sp):+.2f} -> "
                         f"{statistics.fmean(b['scores'][c] for _, b in sp):+.2f}")
        print(f"{c:<24}" + "".join(f"{x:>18}" for x in cells))

    # --- the audit: screen operating point, and what happened to the clinician's rejects ---
    with open(args.audit, encoding="utf-8-sig", newline="") as f:
        audit = [r for r in csv.DictReader(f) if r.get("verdict") in ("accept", "reject")]
    a_items = {i["old_id"]: i for i in map(json.loads, open(args.audit_items, encoding="utf-8")) if i}
    a_old, a_new = load(args.audit_old), load(args.audit_new)
    screen_pairs = [(flag(a_old[r["src_id"]]), r["verdict"] == "reject") for r in audit if r["src_id"] in a_old]
    sens, spec = screen(screen_pairs)

    section("Screen operating point on the 60 audited items (judge run again today)")
    tp = sum(1 for f, rej in screen_pairs if f and rej)
    fp = sum(1 for f, rej in screen_pairs if f and not rej)
    pos = sum(1 for _, rej in screen_pairs if rej)
    print(f"n = {len(screen_pairs)}  TP={tp} FP={fp} FN={pos - tp} TN={len(screen_pairs) - pos - fp}  "
          f"sensitivity {100 * sens:.0f}% [{100 * wilson(tp, pos)[0]:.0f}-{100 * wilson(tp, pos)[1]:.0f}]  "
          f"specificity {100 * spec:.0f}% [{100 * wilson(len(screen_pairs) - pos - fp, len(screen_pairs) - pos)[0]:.0f}-"
          f"{100 * wilson(len(screen_pairs) - pos - fp, len(screen_pairs) - pos)[1]:.0f}]")

    section("Estimated clinician reject rate (Rogan-Gladen, screen and sample uncertainty both resampled)")
    print(f"{'subset':<16}{'old':>26}{'clean':>26}")
    groups = [(s, [s]) for s in subsets] + [("audited three", [s for s in subsets if s != "Curated QA"])]
    for label, ss in groups:
        cells = []
        for rows in ([r for r in old.values() if r["subset"] in ss],
                     [b for i, _, b in pairs if i["subset"] in ss]):
            flags = [flag(r) for r in rows]
            if not flags:
                cells.append("—")
                continue
            p = rg(sum(flags) / len(flags), sens, spec)
            lo, hi = rg_ci(flags, screen_pairs, rng)
            cells.append(f"{100 * p:5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}]")
        print(f"{label:<16}{cells[0]:>26}{cells[1]:>26}")
    print("Curated QA was not in the clinician audit; its corrected rate borrows the screen from the other three.")

    section("Whole corpus: subset rates weighted by training rows (sample and screen resampled)")
    ci = lambda v: (v[int(0.025 * len(v))], v[int(0.975 * len(v))])
    for label, by in (("old", {s: [flag(r) for r in old.values() if r["subset"] == s] for s in subsets}),
                      ("clean", {s: [flag(b) for i, _, b in pairs if i["subset"] == s] for s in subsets})):
        if not all(by.values()):
            print(f"{label}: a subset is not graded yet")
            continue
        w = ROWS[label]
        app = lambda d: sum(w[s] * sum(d[s]) / len(d[s]) for s in subsets) / sum(w[s] for s in subsets)
        boots, rgs = [], []
        for _ in range(BOOT):
            b = app({s: rng.choices(v, k=len(v)) for s, v in by.items()})
            boots.append(b)
            ss = screen(rng.choices(screen_pairs, k=len(screen_pairs)))
            p = rg(b, *ss) if ss else None
            if p is not None:
                rgs.append(p)
        boots.sort()
        rgs.sort()
        a = app(by)
        print(f"{label:<6} flagged {100 * a:5.1f}% [{100 * ci(boots)[0]:.1f}-{100 * ci(boots)[1]:.1f}]   "
              f"estimated clinician reject {100 * rg(a, sens, spec):5.1f}% "
              f"[{100 * ci(rgs)[0]:.1f}-{100 * ci(rgs)[1]:.1f}]")

    section("The 60 audited items in the clean corpus, by clinician verdict")
    for verdict in ("reject", "accept"):
        rows = [r for r in audit if r["verdict"] == verdict]
        st = Counter(a_items[r["src_id"]]["status"] for r in rows)
        k0 = sum(flag(a_old[r["src_id"]]) for r in rows if r["src_id"] in a_old)
        kept = [r for r in rows if a_items[r["src_id"]]["id"] in a_new]
        k1 = sum(flag(a_new[a_items[r["src_id"]]["id"]]) for r in kept)
        print(f"{verdict:<7} n={len(rows):<3} {dict(st)}   judge flags as audited {k0}/{len(rows)}, "
              f"in the clean corpus {k1}/{len(kept)}")
    print("\nclinician-rejected items:")
    for r in audit:
        if r["verdict"] != "reject":
            continue
        it = a_items[r["src_id"]]
        o, n_ = a_old.get(r["src_id"]), a_new.get(it["id"])
        fmt = lambda x: "  —  " if x is None else f"{min(x['scores'].values()):+d}{'F' if flag(x) else ' '} "
        print(f"  #{r['sample_id']:<3} {r['src_id']:<24} {it['status']:<12} min score as audited {fmt(o)}"
              f" clean {fmt(n_)}  {(r.get('comments') or '').strip()[:70]}")


if __name__ == "__main__":
    main()
