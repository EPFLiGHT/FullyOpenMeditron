#!/usr/bin/env python3
# auto_moove/judge_eval_unanimous576_tables.py

"""
The Gemma-4 and thinking numbers of tech_reports/judge_eval_unanimous576.tex,
recomputed from the judge outputs its Reproduce section writes:

.venv/bin/python auto_moove/judge_eval_unanimous576_tables.py

Reads data/moove/moove_eval_unanimous_576.jsonl and
auto_moove/judge_outputs/judged_unanimous576_*.jsonl; a missing output is
skipped. p values are exact McNemar tests on the pairs two settings disagree on.
"""
import json, math, statistics
from pathlib import Path

from auto_moove_check import cohen_kappa, load_pairs, macro_f1, three_way_vote
from auto_moove_common import extract_judge_data, winner_to_moove_vote
from auto_moove_prompt import verdict_model

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "auto_moove" / "judge_outputs"
PAIRS = load_pairs(ROOT / "data" / "moove" / "moove_eval_unanimous_576.jsonl")
KEYS = sorted(PAIRS)
N_RATERS = {k: p["humans"][0]["n_raters"] for k, p in PAIRS.items()}
WINNER = verdict_model("winner")

GEMMA = {  # label -> output tag, all both orders
    "winner / generate": "winner_generate_bidir_gemma4_31b",
    "winner / log-prob": "winner_logprob_bidir_gemma4_31b",
    "moove / generate": "moove_generate_bidir_gemma4_31b",
    "checklist / generate": "checklist_generate_bidir_gemma4_31b",
    "both / generate": "both_generate_bidir_gemma4_31b",
    "preference / generate": "preference_generate_bidir_gemma4_31b",
}
SAMPLED = {"T 1.0": "winner_generate_bidir_k8_t1_gemma4_31b",
           "T 0.7": "winner_generate_bidir_k8_t07_gemma4_31b"}
# label -> (thinking off, thinking on, (start, end) of the reasoning, whether
# the chat template opens the reasoning in the prompt)
THINKING = {
    "Gemma-4-31B": ("winner_generate_bidir_gemma4_31b", "winner_generate_bidir_think_gemma4_31b",
                    ("<|channel>", "<channel|>"), False),
    "Qwen3.8-27B": ("winner_generate_bidir_qwen38_27b", "winner_generate_bidir_think_qwen38_27b",
                    ("<think>", "</think>"), True),
}
GROUPS = {
    "3+ raters": lambda r: N_RATERS[r["pair_key"]] >= 3,
    "2 raters": lambda r: N_RATERS[r["pair_key"]] == 2,
    "Primary": lambda r: r["care_level"].startswith("Primary"),
    "Tertiary": lambda r: r["care_level"].startswith("Tertiary"),
    "HUG": lambda r: r["prefix"].startswith("HUG"),
    "Kenya": lambda r: r["prefix"] == "Kenya ValiDATAthon",   # not "Kenya PHC mini"
}


def load(tag):
    """One record per pair, in KEYS order (every pair has one review row here)."""
    path = J / f"judged_unanimous576_{tag}.jsonl"
    if not path.exists():
        return None
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows.setdefault(r["pair_key"], r)
    assert sorted(rows) == KEYS, path
    return [rows[k] for k in KEYS]


def pct(flags):
    flags = list(flags)
    return 100 * sum(flags) / len(flags)


def votes(rows, field="judge_vote"):
    return [r[field] for r in rows]


def right(rows, vs):
    return [r["human_vote"] == v for r, v in zip(rows, vs)]


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def paired(a, b):
    """Pairs only a gets right, only b gets right, and the McNemar p."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    return only_a, only_b, mcnemar(only_a, only_b)


def metrics(rows, vs):
    """Agreement on the pairs with a verdict; three_all counts the rest as wrong."""
    got = [(r["human_vote"], v) for r, v in zip(rows, vs) if v is not None]
    hv, jv = [h for h, _ in got], [v for _, v in got]
    cons = [r["orientation_consistent"] for r in rows if r.get("orientation_consistent") is not None]
    return {"three": pct(h == v for h, v in got),
            "three_all": 100 * sum(h == v for h, v in got) / len(rows),
            "binary": pct(h == v for h, v in got if h in ("1", "2")),
            "f1": 100 * macro_f1(hv, jv), "kappa": cohen_kappa(hv, jv),
            "flips": pct(not c for c in cons) if cons else float("nan"),
            "n_flips": sum(not c for c in cons), "equal": jv.count("12"),
            "unparsed": len(rows) - len(got)}


def fmt(m):
    return (f"{m['three']:5.1f} {m['binary']:5.1f} {m['f1']:5.1f} {m['kappa']:.3f} "
            f"{m['flips']:5.1f} ({m['n_flips']:2d}) {m['equal']:3d}")


def subgroups(rows, vs):
    """Three-way agreement per group, on the pairs with a verdict (as in metrics)."""
    got = [(r, r["human_vote"] == v) for r, v in zip(rows, vs) if v is not None]
    return " ".join(f"{pct(ok for r, ok in got if f(r)):5.1f}" for f in GROUPS.values())


def revote(r, tags, opens):
    """Re-parse a thinking run's saved outputs. Where the template opened the
    reasoning in the prompt, an output without the end tag never finished it
    and has no verdict; outputs written before auto_moove_common restored the
    opening tag were parsed from a draft inside that reasoning."""
    ab_ba = []
    for field, swap in (("eval_text", False), ("eval_text_ba", True)):
        text = r.get(field) or ""
        if opens and tags[0] not in text:
            text = tags[0] + text
        ab_ba.append(winner_to_moove_vote(getattr(extract_judge_data(text, WINNER), "winner", None), swap))
    ab, ba = ab_ba
    both = ab is not None and ba is not None
    return dict(r, winner_ab=ab, winner_ba=ba, orientation_consistent=(ab == ba) if both else None,
                judge_vote=(ab if ab == ba else "12") if both else None)


def reasoning(text, tags, opens):
    """Words of reasoning and whether it finished; None if there is none."""
    start, end = tags
    if opens and start not in text:
        text = start + text
    if end in text:
        return len(text.rpartition(end)[0].replace(start, " ").split()), True
    return (len(text.split()), False) if start in text else None


JUDGES = {  # label -> output tag, winner rubric, both orders
    "Gemma-4-31B log-prob": "winner_logprob_bidir_gemma4_31b",
    "Gemma-4-31B generate": "winner_generate_bidir_gemma4_31b",
    "Qwen3.8-27B log-prob": "winner_logprob_bidir_qwen38_27b",
    "Qwen3.8-27B generate": "winner_generate_bidir_qwen38_27b",
    "Qwen3-30B-A3B log-prob": "winner_logprob_bidir_qwen3_30b",
    "Qwen3-30B-A3B generate": "winner_generate_bidir_qwen3_30b",
    "GPT-5.6 Sol generate": "winner_generate_bidir_gpt56sol_none",
    # thinking off: Nemotron's empty <think></think>, Muse-Glimmer's answer message
    # opened in the prompt (auto_moove_common.ANSWER_PREFILL); vLLM 0.30, vllm_030.toml
    "Muse-Glimmer-30B log-prob": "winner_logprob_bidir_museglimmer_30b",
    "Muse-Glimmer-30B generate": "winner_generate_bidir_museglimmer_30b",
    "Nemotron-3.5-Lightning-30B-A3B log-prob": "winner_logprob_bidir_nemotron35_30b",
    "Nemotron-3.5-Lightning-30B-A3B generate": "winner_generate_bidir_nemotron35_30b",
}


def robustness():
    """Agreement with the physicians under other tie rules and answer orders
    (tech_reports/iclr_clean_reproduction.tex, judge robustness)."""
    import random
    print("\n== robustness: three-way agreement (%), kappa; log-prob rows only for log-prob runs")
    pick = random.Random(42).choices((0, 1), k=len(KEYS))   # one random order per pair
    for label, tag in JUDGES.items():
        rows = load(tag)
        if rows is None:
            continue
        lp = "prob_left_mean" in rows[0]
        rules = {"default": votes(rows)}
        if lp:
            for t in (0.1, 0.3):
                rules[f"threshold {t}"] = [three_way_vote({"left": r["prob_left_mean"],
                                                           "right": r["prob_right_mean"],
                                                           "equal": r["prob_equal_mean"]}, t) for r in rows]
            rules["flip -> Equal"] = ["12" if r["winner_ab"] != r["winner_ba"] else r["judge_vote"]
                                      for r in rows]
        rules["AB only"] = votes(rows, "winner_ab")
        rules["BA only"] = votes(rows, "winner_ba")
        rules["random order"] = [r["winner_ba" if p else "winner_ab"] for r, p in zip(rows, pick)]
        print(label)
        for name, vs in rules.items():
            m = metrics(rows, vs)
            both = [r["human_vote"] == v for r, v in zip(rows, vs)
                    if v in ("1", "2") and r["human_vote"] in ("1", "2")]
            print(f"  {name:14s} three-way {m['three']:5.1f}  kappa {m['kappa']:.3f}  Equal {m['equal']:3d}  "
                  f"binary {m['binary']:5.1f}  no ties {pct(both):5.1f} (n={len(both)})")


def main():
    robustness()
    gemma = {label: rows for label, tag in GEMMA.items() if (rows := load(tag)) is not None}
    base = gemma["winner / generate"]
    base_right = right(base, votes(base))
    print("group sizes:", {g: sum(f(r) for r in base) for g, f in GROUPS.items()})

    print("\n== Gemma-4-31B in every setting: three-way binary macro-F1 kappa flips% (pairs) Equal"
          " | one order AB / BA | vs winner / generate: only this, only that, p")
    for label, rows in gemma.items():
        ab, ba = (pct(right(rows, votes(rows, f"winner_{s}"))) for s in ("ab", "ba"))
        b, c, p = paired(right(rows, votes(rows)), base_right)
        print(f"{label:24s} {fmt(metrics(rows, votes(rows)))} | {ab:.1f} / {ba:.1f} | {b} {c} p={p:.3f}")

    print("\n== one order vs both orders, winner / generate")
    for side in ("ab", "ba"):
        b, c, p = paired(right(base, votes(base, f"winner_{side}")), base_right)
        print(f"{side.upper()} only: {b} vs {c} pairs, p={p:.3f}")
    if "winner / log-prob" in gemma:
        lp = gemma["winner / log-prob"]
        lp_right = right(lp, votes(lp))
        b, c, p = paired(lp_right, right(base, votes(base, "winner_ab")))
        print(f"log-prob (both orders) vs generate AB only: {b} vs {c}, p={p:.3f}")
        qwen_lp = load("winner_logprob_bidir_qwen38_27b")
        if qwen_lp is not None:
            b, c, p = paired(lp_right, right(qwen_lp, votes(qwen_lp)))
            print(f"Gemma log-prob vs Qwen3.8 log-prob: {b} vs {c}, p={p:.3f}")

        print("\n== log-prob variants: three-way binary macro-F1 kappa flips% (pairs) Equal")
        d2e = ["12" if r["winner_ab"] != r["winner_ba"] else r["judge_vote"] for r in lp]
        print(f"{'disagreement-to-equal':24s} {fmt(metrics(lp, d2e))}")
        for t in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
            vs = [three_way_vote({"left": r["prob_left_mean"], "right": r["prob_right_mean"],
                                  "equal": r["prob_equal_mean"]}, t) for r in lp]
            print(f"{'threshold %.2f' % t:24s} {fmt(metrics(lp, vs))}")

    print("\n== sampling, winner / generate: greedy | one sample mean ± sd | maj@8 | pass@8 | "
          "all 8 agree | flips range")
    greedy = pct(base_right)
    for label, tag in SAMPLED.items():
        rows = load(tag)
        if rows is None:
            continue
        per = [pct(right(rows, vs)) for vs in zip(*votes(rows, "judge_votes"))]
        flips = [pct(not c for c in cs if c is not None)
                 for cs in zip(*votes(rows, "orientation_consistent_samples"))]
        print(f"Gemma-4-31B {label}: {greedy:.1f} | {statistics.mean(per):.1f} ± "
              f"{statistics.stdev(per):.1f} | {pct(right(rows, votes(rows))):.1f} | "
              f"{pct(r['human_vote'] in r['judge_votes'] for r in rows):.1f} | "
              f"{pct(len(set(r['judge_votes'])) == 1 for r in rows):.1f} | "
              f"{min(flips):.1f}-{max(flips):.1f}")

    print("\n== thinking on vs off, winner / generate, both orders")
    sub = {f"Gemma-4-31B, {label.split(' / ')[1]}": gemma[label]
           for label in ("winner / generate", "winner / log-prob") if label in gemma}
    for judge, (off_tag, on_tag, tags, opens) in THINKING.items():
        off, saved = load(off_tag), load(on_tag)
        if off is None or saved is None:
            print(f"{judge}: output missing")
            continue
        on = [revote(r, tags, opens) for r in saved]
        m_off, m_on = metrics(off, votes(off)), metrics(on, votes(on))
        print(f"{judge} off: {fmt(m_off)}")
        print(f"{judge} on:  {fmt(m_on)} | no verdict {m_on['unparsed']}, "
              f"three-way counting those wrong {m_on['three_all']:.1f}")
        b, c, p = paired(right(on, votes(on)), right(off, votes(off)))
        changed = sum(x != y for x, y in zip(votes(on), votes(off)))
        redone = sum(x != y for x, y in zip(votes(on), votes(saved)))
        print(f"  gained {b}, lost {c}, p={p:.4f}; verdict changed on {changed}; "
              f"re-parsing changed {redone} of the run's verdicts")
        print(f"  one order: AB {pct(right(on, votes(on, 'winner_ab'))):.1f} / "
              f"BA {pct(right(on, votes(on, 'winner_ba'))):.1f} (no verdict counted wrong)")
        found = [reasoning(r.get(f) or "", tags, opens) for r in on for f in ("eval_text", "eval_text_ba")]
        done = [n for n, fin in filter(None, found) if fin]
        print(f"  reasoning words: median {statistics.median(done):.0f}, "
              f"p90 {statistics.quantiles(done, n=10)[-1]:.0f}, max {max(done)}; "
              f"unfinished {sum(1 for x in found if x and not x[1])}, "
              f"none {sum(x is None for x in found)} of {len(found)}")
        sub[f"{judge}, thinking"] = on

    print("\n== three-way by subgroup:", " ".join(GROUPS))
    for label, rows in sub.items():
        print(f"{label:28s} {subgroups(rows, votes(rows))}")


if __name__ == "__main__":
    main()
