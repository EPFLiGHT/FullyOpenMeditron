#!/usr/bin/env python3
# tech_reports/figures/healthbench_breakdown_ci.py

"""
HealthBench beyond the overall score, for tech_reports/healthbench_breakdown.tex: every
theme, rubric axis, rubric level, physician-agreed category and consensus cluster (full and
hard), the Professional slices (use case, type, difficulty), penalty criteria, and response
length, for the ICLR report's MeditronFO finetunes, their bases, MedGemma-27B and the 8B
corpus ablations. Gemma-4-31B grader only, as in iclr_clean_reproduction.tex.

Every slice is recomputed from the per-example rubric grades as healthbench.py does it
(per-example score on the slice's rubric items, mean over examples, clipped to [0, 1]), and
checked against the stored metrics. Intervals: 95% bootstrap over the conversations, one
resample per split shared by every model, so a difference between two models is paired.

.venv/bin/python tech_reports/figures/healthbench_breakdown_ci.py   # -> healthbench_breakdown.json
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # login node: 48 BLAS threads fail to spawn

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HB = ROOT / "healthbench/results"
DATA = ROOT / "data/healthbench"
B = 2000
rng = np.random.default_rng(0)

# result names (results_<name>_<split>.json), as HBNAME in iclr_clean_ci.py
HBNAME = {
    "apertus70b": "apertus70b_instruct", "apertus70b_ft": "meditronFO_gptoss_70b_16n",
    "apertus70b_ft_rep": "meditronFO_gptoss_70b_normal",
    "apertus8b": "apertus8b_instruct", "apertus8b_ft": "meditronFO_gptoss_8b",
    "olmo32b": "base_olmo32b_sft", "olmo32b_ft": "meditronFO_gptoss_olmo32b",
    "euro22b": "base_euro22b_instruct", "euro22b_ft": "meditronFO_gptoss_euro22b",
    "euro9b": "base_euro9b_instruct", "euro9b_ft": "meditronFO_gptoss_euro9b",
    "gemma27b": "base_gemma3_27b_it", "gemma27b_ft": "meditronFO_gptoss_gemma3_27b",
    "medgemma": "base_medgemma27b",
    "no_curated_qa": "meditronFO_gptoss_8b_no_curated_qa",
    "no_synthetic_qa": "meditronFO_gptoss_8b_no_synthetic_qa",
    "no_guidelines_qa": "meditronFO_gptoss_8b_no_guidelines_qa",
    "no_synthetic_moove": "meditronFO_gptoss_8b_no_synthetic_moove",
}
# (model, reference): the difference model - reference gets a paired interval
PAIRS = [
    ("apertus70b_ft", "apertus70b"), ("apertus8b_ft", "apertus8b"), ("olmo32b_ft", "olmo32b"),
    ("euro22b_ft", "euro22b"), ("euro9b_ft", "euro9b"), ("gemma27b_ft", "gemma27b"),
    ("medgemma", "gemma27b"), ("gemma27b_ft", "medgemma"),
    ("apertus70b_ft_rep", "apertus70b_ft"),           # training noise: second 70B run
    ("no_curated_qa", "apertus8b_ft"), ("no_synthetic_qa", "apertus8b_ft"),
    ("no_guidelines_qa", "apertus8b_ft"), ("no_synthetic_moove", "apertus8b_ft"),
]
SPLITS = {"full": "oss_eval_full.jsonl", "hard": "hard.jsonl"}
# HealthBench Professional length adjustment (healthbench.py): 2,000 characters unadjusted,
# 0.0147 per 500 characters either way
LEN_CENTER, LEN_PER_500 = 2000.0, 0.0147
SCORE_KINDS = ("overall", "theme:", "physician_agreed_category:", "axis:", "level:", "cluster:",
               "use_case:", "type:", "difficulty:", "score_len_adj", "pos_only")


def example_tags(split):
    """{prompt_id: [example tags]}; Professional's use_case/type/difficulty become tags as in
    healthbench.normalize_professional_example (specialty left out: 2 to 43 conversations each)."""
    out = {}
    for line in open(DATA / SPLITS[split], encoding="utf-8"):
        r = json.loads(line)
        if "rubric_items" in r:
            out[r["id"]] = [f"{k}:{r[k]}" for k in ("use_case", "type", "difficulty")]
        else:
            out[r["prompt_id"]] = r["example_tags"]
    return out


def rubric_score(items):
    possible = sum(i["points"] for i in items if i["points"] > 0)
    if possible == 0:
        return None
    return sum(i["points"] for i in items if i["criteria_met"]) / possible


def per_example(name, split, tags):
    """{metric: {prompt_id: value}} for one model, plus the stored metrics for the check."""
    d = json.load(open(HB / f"{split}_gemma4_31b" / f"results_{name}_{split}.json"))
    m = defaultdict(dict)
    for e in d["metadata"]["example_level_metadata"]:
        pid, items = e["prompt_id"], e["rubric_items"]
        s = float(e["score"])
        m["overall"][pid] = s
        for t in tags[pid]:
            m[t][pid] = s
        by_tag = defaultdict(list)
        for i in items:
            for t in i["tags"]:
                by_tag[t].append(i)
        for t, its in by_tag.items():
            v = rubric_score(its)
            if v is not None:
                m[t][pid] = v
        # penalties: criteria with negative points that the answer met
        possible = sum(i["points"] for i in items if i["points"] > 0)
        neg = [i for i in items if i["points"] < 0]
        lost = sum(-i["points"] for i in neg if i["criteria_met"])
        m["pos_only"][pid] = sum(i["points"] for i in items if i["points"] > 0 and i["criteria_met"]) / possible
        m["penalty_lost"][pid] = lost / possible
        if neg:
            m["neg_any"][pid] = float(any(i["criteria_met"] for i in neg))
            m["neg_rate"][pid] = sum(i["criteria_met"] for i in neg) / len(neg)
        text = e["completion"][0]["content"]
        m["chars"][pid] = float(len(text))
        m["words"][pid] = float(len(text.split()))
        m["score_len_adj"][pid] = s - LEN_PER_500 * (len(text) - LEN_CENTER) / 500.0
    stored = {k: v for k, v in d["metrics"].items() if not k.endswith(("n_samples", "bootstrap_std"))}
    return dict(m), stored


def is_score(metric):
    return metric.startswith(SCORE_KINDS)


def main():
    out = {"models": HBNAME, "pairs": PAIRS, "B": B, "splits": {}}
    for split in SPLITS:
        tags = example_tags(split)
        pids = sorted(tags)
        pos = {p: k for k, p in enumerate(pids)}
        # one resample of the conversations per split, shared by every model and metric
        counts = np.zeros((B, len(pids)), dtype=np.float32)
        for b in range(B):
            counts[b] = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids))
        per_model, boot, points, n = {}, {}, {}, {}
        worst = 0.0
        for key, name in HBNAME.items():
            m, stored = per_example(name, split, tags)
            per_model[key] = m
            for metric, vals in m.items():
                v = np.zeros(len(pids), dtype=np.float32)
                mask = np.zeros(len(pids), dtype=np.float32)
                for p, x in vals.items():
                    v[pos[p]], mask[pos[p]] = x, 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    s = (counts @ (v * mask)) / (counts @ mask)
                pt = float(np.mean(list(vals.values())))
                if is_score(metric):
                    s, pt = np.clip(s, 0, 1), float(np.clip(pt, 0, 1))
                boot[(key, metric)], points[(key, metric)] = s, pt
                n[metric] = len(vals)
                ref = stored.get("overall_score" if metric == "overall" else metric)
                if ref is not None:
                    worst = max(worst, abs(ref - pt))
        assert worst < 1e-6, f"{split}: recomputed slice differs from stored metric by {worst}"
        print(f"{split}: {len(pids)} conversations, slices match the stored metrics (max diff {worst:.1e})")

        scale = lambda metric: 1.0 if metric in ("chars", "words") else 100.0
        res = {"n": n, "scores": {}, "diffs": {}}
        for (key, metric), s in boot.items():
            f = scale(metric)
            lo, hi = np.nanpercentile(s, [2.5, 97.5])
            res["scores"].setdefault(key, {})[metric] = [f * points[(key, metric)], f * float(lo), f * float(hi)]
        for a, b in PAIRS:
            for metric in per_model[a]:
                if (b, metric) not in boot:
                    continue
                f = scale(metric)
                d = boot[(a, metric)] - boot[(b, metric)]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                pt = points[(a, metric)] - points[(b, metric)]
                res["diffs"].setdefault(f"{a}-{b}", {})[metric] = [f * pt, f * float(lo), f * float(hi)]
        out["splits"][split] = res
    json.dump(out, open(HERE / "healthbench_breakdown.json", "w"), indent=1)
    print("wrote", HERE / "healthbench_breakdown.json")


if __name__ == "__main__":
    main()
