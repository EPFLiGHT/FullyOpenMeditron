#!/usr/bin/env python3
# tech_reports/figures/iclr_clean_ci.py

"""
95% bootstrap intervals for the draft figures of iclr_clean_reproduction.tex,
from the saved per-item outputs (login node, no GPU). Writes iclr_clean_ci.json
next to this file; iclr_clean_figures.py reads it.

.venv/bin/python tech_reports/figures/iclr_clean_ci.py
.venv/bin/python tech_reports/figures/iclr_clean_ci.py robustness_humans   # only that key (or judge_majority)

Every interval resamples the items of the evaluation (questions, conversations,
physician pairs). A difference between two models on the same items resamples
the items once for both, so its interval is paired. The MCQA average resamples
each of its four tasks on its own.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # login node: 48 BLAS threads fail to spawn

import glob, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B = 2000
rng = np.random.default_rng(0)

EVAL = ROOT / "eval_results"
HB = ROOT / "healthbench/results"
AM = ROOT / "auto_moove/results/unanimous576"      # AutoMOOVE runs (auto_moove.py outputs)
TAG = ""
TASKS = ("medmcqa_g", "medqa_g", "pubmedqa_g", "medxpertqa_g")
MOOVE = ("clinical_soundness", "safety", "contextual_fit", "appropriate_confidence", "clarity_completeness")
SCORE = {"1": 1.0, "12": 0.5, "2": 0.0}

# eval-gen run directories behind the report's tables (checked against their numbers)
RUN = {
    "apertus70b": "apertus-70b-instruct-2509_3478465",
    "apertus70b_gptoss": "apertus_70b_meditronfo_gptoss_v1_16n_3465533",
    "apertus70b_gptoss_rep": "apertus_70b_meditronfo_gptoss_v1_normal_3465341",
    "apertus70b_qwen": "apertus_70b_meditronfo_qwen38_v3_16n_3474504",
    "apertus70b_qwen_rep": "apertus_70b_meditronfo_qwen38_v3_normal_3478453",
    "apertus8b": "apertus-8b-instruct-2509_3478462",
    "apertus8b_gptoss": "apertus_8b_meditronfo_gptoss_v1_3464365",
    "apertus8b_qwen": "apertus_8b_meditronfo_qwen38_v3_3464367",
    "no_curated_qa": "apertus_8b_meditronfo_gptoss_v1_no_curated_qa_3467724",
    "no_synthetic_qa": "apertus_8b_meditronfo_gptoss_v1_no_synthetic_qa_3467727",
    "no_guidelines_qa": "apertus_8b_meditronfo_gptoss_v1_no_guidelines_qa_3467732",
    "no_synthetic_moove": "apertus_8b_meditronfo_gptoss_v1_no_synthetic_moove_3467734",
    # the same four ablations on Apertus-70B (normal, 8 nodes), added 26 September
    "no_curated_qa_70b": "apertus_70b_meditronfo_gptoss_v1_no_curated_qa_3523363",
    "no_synthetic_qa_70b": "apertus_70b_meditronfo_gptoss_v1_no_synthetic_qa_3523365",
    "no_guidelines_qa_70b": "apertus_70b_meditronfo_gptoss_v1_no_guidelines_qa_3523919",
    "no_synthetic_moove_70b": "apertus_70b_meditronfo_gptoss_v1_no_synthetic_moove_3523369",
    "olmo32b": "olmo-2-0325-32b-sft_3485004",
    "olmo32b_ft": "olmo_2_32b_meditronfo_gptoss_v1_3467269",
    "euro22b": "eurollm-22b-instruct-2512_3484596",
    "euro22b_ft": "euro_llm_22b_meditronfo_gptoss_v1_3467274",
    "euro9b": "eurollm-9b-instruct_3484597",
    "euro9b_ft": "euro_llm_9b_meditronfo_gptoss_v1_3467276",
    "gemma27b": "gemma-3-27b-it_3484598",
    "gemma27b_ft": "gemma_3_27b_meditronfo_gptoss_v1_3474465",
    "medgemma": "medgemma-27b-text-it_3489833",
}
# HealthBench result names (results_<name>_<full|hard>.json)
HBNAME = {
    "apertus70b": "apertus70b_instruct", "apertus70b_gptoss": "meditronFO_gptoss_70b_16n",
    "apertus70b_gptoss_rep": "meditronFO_gptoss_70b_normal", "apertus70b_qwen": "meditronFO_qwen38_70b_16n",
    "apertus70b_qwen_rep": "meditronFO_qwen38_70b_normal",
    "apertus8b": "apertus8b_instruct", "apertus8b_gptoss": "meditronFO_gptoss_8b",
    "apertus8b_qwen": "meditronFO_qwen38_8b",
    "no_curated_qa": "meditronFO_gptoss_8b_no_curated_qa", "no_synthetic_qa": "meditronFO_gptoss_8b_no_synthetic_qa",
    "no_guidelines_qa": "meditronFO_gptoss_8b_no_guidelines_qa",
    "no_synthetic_moove": "meditronFO_gptoss_8b_no_synthetic_moove",
    "no_curated_qa_70b": "meditronFO_gptoss_70b_no_curated_qa", "no_synthetic_qa_70b": "meditronFO_gptoss_70b_no_synthetic_qa",
    "no_guidelines_qa_70b": "meditronFO_gptoss_70b_no_guidelines_qa",
    "no_synthetic_moove_70b": "meditronFO_gptoss_70b_no_synthetic_moove",
    "olmo32b": "base_olmo32b_sft", "olmo32b_ft": "meditronFO_gptoss_olmo32b",
    "euro22b": "base_euro22b_instruct", "euro22b_ft": "meditronFO_gptoss_euro22b",
    "euro9b": "base_euro9b_instruct", "euro9b_ft": "meditronFO_gptoss_euro9b",
    "gemma27b": "base_gemma3_27b_it", "gemma27b_ft": "meditronFO_gptoss_gemma3_27b",
    "medgemma": "base_medgemma27b",
    "meditron3_70b": "ref_meditron3_70b", "aloe_70b": "ref_aloe_beta_70b",
}
HBDIR = {"full_gemma4": "full_gemma4_31b", "hard_gemma4": "hard_gemma4_31b",
         "professional_gemma4": "professional_gemma4_31b"}


def interval(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


# --- per-item loaders ---------------------------------------------------------------------
def medbench(key):
    """{task: {doc_id: 0/1}} from the rescored samples."""
    out = {}
    for t in TASKS:
        # a directory evaluated twice keeps both; the latest is the one in the tables. Runs
        # evaluated after the filter fix have no _rescored copy.
        d = EVAL / f"eval-gen_{RUN[key]}" / "*"
        f = sorted(glob.glob(str(d / f"samples_{t}_*_rescored.jsonl")) or glob.glob(str(d / f"samples_{t}_*.jsonl")))[-1]
        out[t] = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[t][r["doc_id"]] = float(r["exact_match"])
    return out


def ifeval(key):
    f = sorted(glob.glob(str(EVAL / f"eval-gen_{RUN[key]}" / "*" / "samples_ifeval_*.jsonl")))[-1]
    return {json.loads(l)["doc_id"]: float(json.loads(l)["prompt_level_strict_acc"]) for l in open(f)}


def mmlu_pro(key, unparsed=False):
    """{subject:doc_id: 0/1} over the 14 subjects; its mean is the harness's size-weighted mmlu_pro.
    unparsed=True: 0/1 for an answer the harness's filter ("the answer is (X)") could not read."""
    out = {}
    for f in glob.glob(str(EVAL / f"eval-gen_{RUN[key]}" / "*" / "samples_mmlu_pro_*.jsonl")):
        subj = Path(f).name[len("samples_mmlu_pro_"):].rsplit("_", 1)[0]
        if f != sorted(glob.glob(str(Path(f).parent / f"samples_mmlu_pro_{subj}_*.jsonl")))[-1]:
            continue                      # evaluated twice: keep the latest, as for medbench
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[f"{subj}:{r['doc_id']}"] = (float(r["filtered_resps"][0] not in "ABCDEFGHIJ")
                                            if unparsed else float(r["exact_match"]))
    return out


def arc(key):
    """ARC-Challenge acc_norm, the metric of the paper's appendix table."""
    f = sorted(glob.glob(str(EVAL / f"eval-gen_{RUN[key]}" / "*" / "samples_arc_challenge_*.jsonl")))[-1]
    return {json.loads(l)["doc_id"]: float(json.loads(l)["acc_norm"]) for l in open(f)}


def healthbench(key, split):
    name = HBNAME[key]
    kind = split.split("_")[0]
    path = HB / HBDIR[split] / f"results_{name}_{kind}.json"
    d = json.load(open(path))
    ex = d["metadata"]["example_level_metadata"]
    return {e["prompt_id"]: float(e["score"]) for e in ex}, float(d["score"])


def auto_moove(model_1, model_2, rubric, judge="gemma-4-31B-it"):
    # a preempted and requeued job writes one file per attempt; the latest is complete
    f = sorted(glob.glob(str(AM / f"*_auto_moove_{model_1}{TAG}_{model_2}{TAG}_{judge}_{rubric}_bidir.jsonl")))[-1]
    last = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        last[r["prompt"]] = r        # last copy of each question, as in unanimous576_h2h_tables.py
    return last


# --- bootstrap machinery ------------------------------------------------------------------
def mcqa_boot(runs):
    """B x len(runs) MCQA averages (%), each task resampled once for every run."""
    out = np.zeros((B, len(runs)))
    for t in TASKS:
        ids = sorted(set.intersection(*(set(r[t]) for r in runs)))
        m = np.array([[r[t][i] for i in ids] for r in runs])          # runs x docs
        idx = rng.integers(0, len(ids), (B, len(ids)))
        out += np.stack([m[k][idx].mean(axis=1) for k in range(len(runs))], axis=1)
    return 100 * out / len(TASKS), 100 * np.array(
        [np.mean([np.mean(list(r[t].values())) for t in TASKS]) for r in runs])


def item_boot(dicts, stat=np.mean, clip=False):
    """B x len(dicts) of stat over shared, resampled items; point estimates on all items of each."""
    ids = sorted(set.intersection(*(set(d) for d in dicts)))
    m = np.array([[d[i] for i in ids] for d in dicts])
    idx = rng.integers(0, len(ids), (B, len(ids)))
    s = np.stack([m[k][idx].mean(axis=1) for k in range(len(dicts))], axis=1)
    point = np.array([np.mean(list(d.values())) for d in dicts])
    if clip:                      # simple-evals clips the HealthBench mean to [0, 1]
        s, point = np.clip(s, 0, 1), np.clip(point, 0, 1)
    return 100 * s, 100 * point


def pair_ci(boot, point, i, j=None):
    """Interval of column i, or of column i minus column j."""
    if j is None:
        return interval(boot[:, i], point[i])
    return interval(boot[:, i] - boot[:, j], point[i] - point[j])


# --- figures ------------------------------------------------------------------------------
def medbench_fig(cache):
    pairs = [("apertus70b", "apertus70b_gptoss"), ("olmo32b", "olmo32b_ft"),
             ("euro22b", "euro22b_ft"), ("apertus8b", "apertus8b_gptoss"), ("euro9b", "euro9b_ft"),
             ("gemma27b", "gemma27b_ft")]
    out = {}
    for base, ft in pairs:
        rec = {}
        bs, pt = mcqa_boot([cache.mb(base), cache.mb(ft)])
        rec["mcqa"] = {"base": pair_ci(bs, pt, 0), "ft": pair_ci(bs, pt, 1), "delta": pair_ci(bs, pt, 1, 0)}
        bs, pt = item_boot([cache.hb(base, "full_gemma4"), cache.hb(ft, "full_gemma4")], clip=True)
        rec["hb"] = {"base": pair_ci(bs, pt, 0), "ft": pair_ci(bs, pt, 1), "delta": pair_ci(bs, pt, 1, 0)}
        out[ft] = rec
    bs, pt = mcqa_boot([cache.mb("medgemma")])
    hs, hp = item_boot([cache.hb("medgemma", "full_gemma4")], clip=True)
    out["medgemma"] = {"mcqa": pair_ci(bs, pt, 0), "hb": pair_ci(hs, hp, 0)}
    return out


GENERAL = {"mmlu_pro": mmlu_pro, "ifeval": ifeval, "arc": arc}


def general_fig():
    """tab:general: MMLU-Pro, IFEval, ARC-Challenge and their mean, per model and as the gain over the base."""
    global rng
    rng = np.random.default_rng(1)        # own seed: its draws do not depend on the steps before it
    pairs = [("apertus70b", "apertus70b_gptoss"), ("olmo32b", "olmo32b_ft"), ("euro22b", "euro22b_ft"),
             ("apertus8b", "apertus8b_gptoss"), ("euro9b", "euro9b_ft"), ("gemma27b", "gemma27b_ft"),
             ("gemma27b", "medgemma")]
    data = {k: {b: fn(k) for b, fn in GENERAL.items()} for k in dict.fromkeys(k for p in pairs for k in p)}
    out = {}
    for base, ft in pairs:
        rec, avg = {}, np.zeros((B, 2))
        for b in GENERAL:
            bs, pt = item_boot([data[ft][b], data[base][b]])   # each benchmark resampled once for both
            avg += bs / len(GENERAL)
            rec[b] = {"ft": pair_ci(bs, pt, 0), "base": pair_ci(bs, pt, 1), "delta": pair_ci(bs, pt, 0, 1)}
        pt = np.array([np.mean([rec[b][s][0] for b in GENERAL]) for s in ("ft", "base")])
        rec["avg"] = {"ft": pair_ci(avg, pt, 0), "base": pair_ci(avg, pt, 1), "delta": pair_ci(avg, pt, 0, 1)}
        out[ft] = rec
    out["mmlu_pro_unparsed"] = {k: 100 * float(np.mean(list(mmlu_pro(k, unparsed=True).values()))) for k in data}
    return out


def teacher_fig(cache):
    keys = ["apertus8b", "apertus8b_gptoss", "apertus8b_qwen", "apertus70b", "apertus70b_gptoss",
            "apertus70b_gptoss_rep", "apertus70b_qwen", "apertus70b_qwen_rep"]
    out = {k: {} for k in keys}
    for k in keys:
        bs, pt = mcqa_boot([cache.mb(k)])
        out[k]["mcqa"] = pair_ci(bs, pt, 0)
        for split in ("full_gemma4", "hard_gemma4", "professional_gemma4"):
            bs, pt = item_boot([cache.hb(k, split)], clip=True)
            out[k][split] = pair_ci(bs, pt, 0)
        bs, pt = item_boot([ifeval(k)])
        out[k]["ifeval"] = pair_ci(bs, pt, 0)
    # teacher gaps, paired on the same items
    for size, g, q in (("8B", "apertus8b_gptoss", "apertus8b_qwen"), ("70B", "apertus70b_gptoss", "apertus70b_qwen")):
        gap = {}
        bs, pt = mcqa_boot([cache.mb(q), cache.mb(g)])
        gap["mcqa"] = pair_ci(bs, pt, 0, 1)
        for split in ("full_gemma4", "hard_gemma4", "professional_gemma4"):
            bs, pt = item_boot([cache.hb(q, split), cache.hb(g, split)], clip=True)
            gap[split] = pair_ci(bs, pt, 0, 1)
        bs, pt = item_boot([ifeval(q), ifeval(g)])
        gap["ifeval"] = pair_ci(bs, pt, 0, 1)
        out[f"gap_{size}"] = gap          # Qwen3.8 minus gpt-oss
    return out


def criterion_scores(model_1, model_2, judge="gemma-4-31B-it"):
    d = auto_moove(model_1, model_2, "moove", judge)
    rows = [r for r in d.values() if all(r.get(f"m1_{c}") is not None and r.get(f"m2_{c}") is not None
                                         for c in MOOVE)]
    return {r["prompt"]: {c: (float(r[f"m1_{c}"]), float(r[f"m2_{c}"])) for c in MOOVE} for r in rows}


# MeditronFO models against their base, on top of the two of the paper's radar; they run last, so
# the two radar models and every other figure keep their bootstrap draws
MORE_PAIRS = (("apertus8b", "apertus_8b_meditronFO_gptoss_v1", "Apertus-8B-Instruct-2509"),
              ("olmo32b", "olmo_2_32b_meditronFO_gptoss_v1", "OLMo-2-0325-32B-SFT"),
              ("euro22b", "euro_llm_22b_meditronFO_gptoss_v1", "EuroLLM-22B-Instruct-2512"),
              ("euro9b", "euro_llm_9b_meditronFO_gptoss_v1", "EuroLLM-9B-Instruct"))


RADAR_PAIRS = (("apertus70b", "apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-Instruct-2509"),
               ("gemma27b", "gemma_3_27b_meditronFO_gptoss_v1", "gemma-3-27b-it"))


def criteria_fig(pairs=RADAR_PAIRS, judge="gemma-4-31B-it"):
    out = {}
    for name, m1, m2 in pairs:
        s = criterion_scores(m1, m2, judge)
        ids = list(s)
        idx = rng.integers(0, len(ids), (B, len(ids)))
        rec = {"n": len(ids)}
        deltas = []
        for c in MOOVE:
            ft = np.array([s[i][c][0] for i in ids])
            base = np.array([s[i][c][1] for i in ids])
            rec[c] = {"ft": interval(ft[idx].mean(1), ft.mean()), "base": interval(base[idx].mean(1), base.mean()),
                      "delta": interval((ft - base)[idx].mean(1), (ft - base).mean())}
            deltas.append(ft - base)
        md = np.mean(deltas, axis=0)
        rec["mean_delta"] = interval(md[idx].mean(1), md.mean())
        out[name] = rec
    return out


def ablations_fig(cache):
    configs = ["no_curated_qa", "no_synthetic_qa", "no_guidelines_qa", "no_synthetic_moove"]
    am_name = {"full": "apertus_8b_meditronFO_gptoss_v1"} | {
        c: f"apertus_8b_meditronFO_gptoss_v1_{c}" for c in configs}
    runs = {"full": "apertus8b_gptoss"} | {c: c for c in configs}
    out = {c: {} for c in configs}
    # MCQA average and HealthBench: each configuration minus the full corpus, paired
    order = ["full"] + configs
    bs, pt = mcqa_boot([cache.mb(runs[k]) for k in order])
    hs, hp = item_boot([cache.hb(runs[k], "full_gemma4") for k in order], clip=True)
    for i, c in enumerate(configs, 1):
        out[c]["mcqa"] = pair_ci(bs, pt, i, 0)
        out[c]["hb"] = pair_ci(hs, hp, i, 0)
    # AutoMOOVE win rate and Likert delta against the base: configuration minus full, paired by question
    win, lik = {}, {}
    for k in order:
        d = auto_moove(am_name[k], "Apertus-8B-Instruct-2509", "winner_logprob")
        win[k] = {p: 100 * SCORE[r["judge_vote"]] for p, r in d.items() if r["judge_vote"] in SCORE}
        try:
            s = criterion_scores(am_name[k], "Apertus-8B-Instruct-2509")
            lik[k] = {p: np.mean([v[c][0] - v[c][1] for c in MOOVE]) for p, v in s.items()}
        except IndexError:
            # no Likert run of the ablations: draw the same resample of the questions anyway, so every
            # interval after this one keeps its draws (automoove_only() replaces these fields)
            lik[k] = {p: 0.0 for p in win[k]}
    ws, wp = item_boot([win[k] for k in order])
    ls, lp = item_boot([lik[k] for k in order])
    ws, wp, ls, lp = ws / 100, wp / 100, ls / 100, lp / 100
    for i, c in enumerate(configs, 1):
        out[c]["automoove"] = pair_ci(ws, wp, i, 0)
        out[c]["likert"] = pair_ci(ls, lp, i, 0)
    out["levels"] = {"automoove": {k: float(wp[i]) for i, k in enumerate(order)},
                     "likert": {k: float(lp[i]) for i, k in enumerate(order)}}
    return out


def ablations_abs_fig():
    """The paper's ablation figure: each Apertus-8B model's own score with its interval on MCQA average,
    HealthBench full, AutoMOOVE win rate against Apertus-8B-Instruct (Gemma-4, winner log-prob, ties
    half) and IFEval (prompt-level strict). Seeded on its own so it adds to the json without moving
    the draws of the other keys."""
    global rng
    rng = np.random.default_rng(2)
    configs = ["no_curated_qa", "no_synthetic_qa", "no_guidelines_qa", "no_synthetic_moove"]
    runs = {"base": "apertus8b", "full": "apertus8b_gptoss"} | {c: c for c in configs}
    am_name = {"full": "apertus_8b_meditronFO_gptoss_v1"} | {c: f"apertus_8b_meditronFO_gptoss_v1_{c}"
                                                            for c in configs}
    out = {k: {} for k in runs}
    for k, r in runs.items():
        bs, pt = mcqa_boot([medbench(r)])
        out[k]["mcqa"] = pair_ci(bs, pt, 0)
        bs, pt = item_boot([healthbench(r, "full_gemma4")[0]], clip=True)
        out[k]["hb"] = pair_ci(bs, pt, 0)
        bs, pt = item_boot([ifeval(r)])
        out[k]["ifeval"] = pair_ci(bs, pt, 0)
        if k in am_name:
            d = auto_moove(am_name[k], "Apertus-8B-Instruct-2509", "winner_logprob")
            bs, pt = item_boot([{p: SCORE[x["judge_vote"]] for p, x in d.items() if x["judge_vote"] in SCORE}])
            out[k]["automoove"] = pair_ci(bs, pt, 0)
    # Apertus-70B-MeditronFO (gpt-oss, first run) against its base, for the teacher rows of the figure
    d = auto_moove("apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-Instruct-2509", "winner_logprob")
    bs, pt = item_boot([{p: SCORE[x["judge_vote"]] for p, x in d.items() if x["judge_vote"] in SCORE}])
    out["apertus70b_gptoss"] = {"automoove": pair_ci(bs, pt, 0)}
    return out


def ablations70_abs_fig():
    """ablations_abs_fig for the Apertus-70B ablations: each model's own MCQA average, HealthBench full
    and IFEval with its interval (their AutoMOOVE rates come from ablations_am_fig).
    Seeded on its own."""
    global rng
    rng = np.random.default_rng(3)
    out = {}
    for c in ("no_curated_qa", "no_synthetic_qa", "no_guidelines_qa", "no_synthetic_moove"):
        r = f"{c}_70b"
        out[c] = {}
        bs, pt = mcqa_boot([medbench(r)])
        out[c]["mcqa"] = pair_ci(bs, pt, 0)
        bs, pt = item_boot([healthbench(r, "full_gemma4")[0]], clip=True)
        out[c]["hb"] = pair_ci(bs, pt, 0)
        bs, pt = item_boot([ifeval(r)])
        out[c]["ifeval"] = pair_ci(bs, pt, 0)
    return out


def robustness_fig(judges=("gemma-4-31B-it", "Qwen3.8-27B")):
    """unanimous576_h2h_tables.robustness, with an interval on every row."""
    sys.path.insert(0, str(ROOT / "auto_moove"))
    from unanimous576_h2h_tables import three_way
    out = {}
    for m1 in ("apertus_70b_meditronFO_gptoss_v1_16n", "gemma_3_27b_meditronFO_gptoss_v1"):
        for judge in judges:
            d = auto_moove(m1, "medgemma-27b-text-it", "winner_logprob", judge)
            dec = [r for r in d.values() if r["judge_vote"] in SCORE]
            v = [r["verdict"] for r in dec]
            pick = np.random.default_rng(42).integers(0, 2, len(v))
            rows = {
                "default": [SCORE[r["judge_vote"]] for r in dec],
                "thr0.1": [SCORE[three_way(x, 0.1)] for x in v],
                "thr0.3": [SCORE[three_way(x, 0.3)] for x in v],
                "flip_equal": [SCORE["12" if x["winner_ab"] != x["winner_ba"] else x["averaged_judge_vote"]]
                               for x in v],
                "ties_dropped": [SCORE[x["judge_vote"]] for x in v if x["judge_vote"] != "12"],
                # ties dropped under the flip rule: the questions both orders give the same winner, 25 September
                "flip_ties_dropped": [SCORE[x["averaged_judge_vote"]] for x in v if x["winner_ab"] == x["winner_ba"]
                                      and x["averaged_judge_vote"] != "12"],
                "ab": [SCORE[x["winner_ab"]] for x in v],
                "ba": [SCORE[x["winner_ba"]] for x in v],
                "random": [SCORE[x["winner_ba"] if p else x["winner_ab"]] for x, p in zip(v, pick)],
            }
            rec = {}
            for k, xs in rows.items():
                xs = 100 * np.asarray(xs)
                # seeded per row, so the default row repeats tab:wtl's interval for the same votes
                idx = np.random.default_rng(0).integers(0, len(xs), (B, len(xs)))
                rec[k] = interval(xs[idx].mean(1), xs.mean())
            out[f"{m1}|{judge}"] = rec
    return out


def macro_f1_boot(h, j, classes):
    """auto_moove_check.macro_f1 over the given classes, one value per row of the (resamples, n) label arrays."""
    f1 = []
    for v in classes:
        tp = ((h == v) & (j == v)).sum(1)
        wrong = ((h == v) != (j == v)).sum(1)   # false positives plus false negatives
        f1.append(np.where(tp + wrong > 0, 2 * tp / np.maximum(2 * tp + wrong, 1), 0.0))
    return np.mean(f1, axis=0)


def robustness_humans_fig():
    """judge_eval_unanimous576_tables.robustness, with intervals: three-way agreement and kappa with the
    physicians' unanimous verdict under each tie rule and answer order, resampling the 576 pairs."""
    sys.path.insert(0, str(ROOT / "auto_moove"))
    import random
    import judge_eval_unanimous576_tables as jt
    from auto_moove_check import cohen_kappa, three_way_vote
    pick = random.Random(42).choices((0, 1), k=len(jt.KEYS))   # the table's random order
    out, votes = {}, {}
    for tag in jt.JUDGES.values():
        rows = jt.load(tag)
        if rows is None:
            continue
        rules = {"default": jt.votes(rows)}
        if "prob_left_mean" in rows[0]:
            for t, k in ((0.1, "thr0.1"), (0.3, "thr0.3")):
                rules[k] = [three_way_vote({"left": r["prob_left_mean"], "right": r["prob_right_mean"],
                                            "equal": r["prob_equal_mean"]}, t) for r in rows]
            rules["flip_equal"] = ["12" if r["winner_ab"] != r["winner_ba"] else r["judge_vote"] for r in rows]
            rules["flip_ties_dropped"] = rules["flip_equal"]      # ties dropped under the flip rule, 25 September
        rules["ties_dropped"] = rules["default"]
        rules["ab"] = jt.votes(rows, "winner_ab")
        rules["ba"] = jt.votes(rows, "winner_ba")
        rules["random"] = [r["winner_ba" if p else "winner_ab"] for r, p in zip(rows, pick)]
        votes[tag] = (rows, rules)
    # majority of the three judges of the main results under each rule (wtl_fig's majority: the verdict
    # two of them share, a tie when all three differ), added 25 September
    for scoring in ("logprob", "generate"):
        tags = [f"winner_{scoring}_bidir_{j}" for j in ("gemma4_31b", "qwen38_27b", "museglimmer_30b")]
        if all(t in votes for t in tags):
            rules = {k: [max(("1", "2", "12"), key=vs.count) if len(set(vs)) < 3 else "12"
                         for vs in zip(*(votes[t][1][k] for t in tags))] for k in votes[tags[0]][1]}
            votes[f"winner_{scoring}_majority"] = (votes[tags[0]][0], rules)
    for tag, (rows, rules) in votes.items():
        rec = {}
        for k, vs in rules.items():
            # as jt.metrics: unparsed verdicts left out; ties dropped keeps the pairs both called a winner
            got = [(r["human_vote"], v) for r, v in zip(rows, vs) if v is not None]
            if k in ("ties_dropped", "flip_ties_dropped"):
                got = [(h, v) for h, v in got if h in ("1", "2") and v in ("1", "2")]
            h, j = np.array([g[0] for g in got]), np.array([g[1] for g in got])
            idx = np.random.default_rng(0).integers(0, len(h), (B, len(h)))   # seeded per row
            agree = 100 * (h == j)
            # ties dropped has no Equal class left, so its macro-F1 averages the two winner classes
            classes = ("1", "2") if k in ("ties_dropped", "flip_ties_dropped") else ("1", "2", "12")
            rec[k] = {"agree": interval(agree[idx].mean(1), agree.mean()),
                      "kappa": interval([cohen_kappa(list(h[i]), list(j[i])) for i in idx],
                                        cohen_kappa(list(h), list(j))),
                      "f1": interval(macro_f1_boot(h[idx], j[idx], classes),
                                     float(macro_f1_boot(h[None], j[None], classes)[0])),
                      "equal": int((j == "12").sum()), "n": len(h)}
        out[tag] = rec
    return out


def length_fig(judge="gemma-4-31B-it"):
    d = auto_moove("gemma_3_27b_meditronFO_gptoss_v1", "medgemma-27b-text-it", "winner_logprob", judge)
    rows = [r for r in d.values() if r["judge_vote"] in SCORE]
    words = lambda s: len((s or "").split())
    lr = np.log([words(r["model_1_answer"]) / max(1, words(r["model_2_answer"])) for r in rows])
    m = np.array([r["verdict"]["prob_left_mean"] - r["verdict"]["prob_right_mean"] for r in rows])
    idx = rng.integers(0, len(rows), (B, len(rows)))
    rs = [np.corrcoef(m[i], lr[i])[0, 1] for i in idx]
    return {"r": interval(rs, np.corrcoef(m, lr)[0, 1])}


def judge_validation_fig():
    """Per-criterion kappa of moove_criteria_agreement.py, resampling physician pairs."""
    sys.path.insert(0, str(ROOT / "auto_moove"))
    import argparse
    import moove_criteria_agreement as mca
    args = argparse.Namespace(test=str(ROOT / "data/moove/moove_eval_unanimous_576.jsonl"),
                              reviews=str(ROOT / "moove/all_reviews.jsonl"), rounded=False, prefix=None)
    cwd = os.getcwd()
    os.chdir(ROOT)
    reviews = mca.human_reviews(args)
    os.chdir(cwd)
    judged = mca.load_judged(str(ROOT / "auto_moove/judge_outputs/"
                                 "judged_unanimous576_moove_pref_generate_bidir_gemma4_31b.jsonl"))
    kappa = mca.cohen_kappa
    groups = {pk: g for pk, g in reviews.groupby("pair_key")}
    keys = sorted(groups)
    out = {}
    for c in MOOVE:
        jv = {k: r.get(f"j_pref_{c}") for k, r in judged.items()}
        lab = {k: v for k, v in mca.pair_labels(reviews, c, "unanimous").items() if jv.get(k)}
        # per pair: unanimous label, the reviews' (human, judge) rows, and the rater-rater pairs
        per = {}
        for k in keys:
            g = groups[k]
            prefs = [p for p in g[c] if isinstance(p, str)]
            rv = [(p, jv[k]) for p in prefs if jv.get(k)]
            hh = [(x, y) for a, b in __import__("itertools").combinations(prefs, 2) for x, y in ((a, b), (b, a))]
            per[k] = (lab.get(k), rv, hh)

        def stats(sample):
            u = [(per[k][0], jv[k]) for k in sample if per[k][0] is not None]
            r = [x for k in sample for x in per[k][1]]
            h = [x for k in sample for x in per[k][2]]
            f = lambda xs: kappa([a for a, _ in xs], [b for _, b in xs])
            return f(u), f(r), f(h)

        point = stats(keys)
        boots = np.array([stats([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(1000)])
        out[c] = {name: interval(boots[:, i], point[i]) for i, name in enumerate(("unanimous", "review", "hh"))}
    return out


def tables(cache):
    """The paired intervals the report's tables carry: gains, teacher gaps and replicate differences."""
    out = {}
    # tab:timeline, one interval per open point
    out["timeline"] = {}
    for k in ("gemma27b_ft", "gemma27b", "medgemma", "apertus70b_gptoss", "apertus8b_gptoss", "apertus70b",
              "aloe_70b", "apertus8b", "meditron3_70b"):
        bs, pt = item_boot([cache.hb(k, "full_gemma4")], clip=True)
        out["timeline"][k] = pair_ci(bs, pt, 0)
    # tab:medbench, MCQA gain over the base (MedGemma over Gemma-3-27B-IT)
    out["medbench_gain"] = {}
    for base, ft in (("apertus70b", "apertus70b_gptoss"), ("olmo32b", "olmo32b_ft"),
                     ("euro22b", "euro22b_ft"), ("apertus8b", "apertus8b_gptoss"), ("euro9b", "euro9b_ft"),
                     ("gemma27b", "gemma27b_ft"), ("gemma27b", "medgemma")):
        bs, pt = mcqa_boot([cache.mb(ft), cache.mb(base)])
        out["medbench_gain"][ft] = pair_ci(bs, pt, 0, 1)
    # tab:teacher, Qwen3.8 minus gpt-oss and run 2 minus run 1, every metric of the table
    out["teacher"] = {}
    for name, a, b in (("gap_8B", "apertus8b_qwen", "apertus8b_gptoss"), ("gap_70B", "apertus70b_qwen", "apertus70b_gptoss"),
                       ("rep_gptoss", "apertus70b_gptoss_rep", "apertus70b_gptoss"),
                       ("rep_qwen", "apertus70b_qwen_rep", "apertus70b_qwen")):
        rec = {}
        bs, pt = mcqa_boot([cache.mb(a), cache.mb(b)])
        rec["mcqa"] = pair_ci(bs, pt, 0, 1)
        for t in TASKS:
            x, y = cache.mb(a)[t], cache.mb(b)[t]
            bs, pt = item_boot([x, y])
            rec[t] = pair_ci(bs, pt, 0, 1)
        for split in ("full_gemma4", "hard_gemma4", "professional_gemma4"):
            bs, pt = item_boot([cache.hb(a, split), cache.hb(b, split)], clip=True)
            rec[split] = pair_ci(bs, pt, 0, 1)
        bs, pt = item_boot([ifeval(a), ifeval(b)])
        rec["ifeval"] = pair_ci(bs, pt, 0, 1)
        out["teacher"][name] = rec
    return out


def wtl_fig():
    """tab:wtl: win / tie / loss counts and the win rate (ties half) of every AutoMOOVE pair."""
    out = {}
    # Muse-Glimmer-30B, the third judge, added 24 September (log-prob, thinking off)
    judges = ("gemma-4-31B-it", "Qwen3.8-27B", "Muse-Glimmer-30B")
    for name, m1, m2 in (
            ("apertus70b", "apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-Instruct-2509"),
            ("apertus8b", "apertus_8b_meditronFO_gptoss_v1", "Apertus-8B-Instruct-2509"),
            ("olmo32b", "olmo_2_32b_meditronFO_gptoss_v1", "OLMo-2-0325-32B-SFT"),
            ("euro22b", "euro_llm_22b_meditronFO_gptoss_v1", "EuroLLM-22B-Instruct-2512"),
            ("euro9b", "euro_llm_9b_meditronFO_gptoss_v1", "EuroLLM-9B-Instruct"),
            ("gemma27b", "gemma_3_27b_meditronFO_gptoss_v1", "gemma-3-27b-it"),
            ("gemma27b_medgemma", "gemma_3_27b_meditronFO_gptoss_v1", "medgemma-27b-text-it"),
            ("apertus70b_medgemma", "apertus_70b_meditronFO_gptoss_v1_16n", "medgemma-27b-text-it")):
        rec, by_judge, flip = {}, {}, {}
        for judge in judges:
            try:
                d = auto_moove(m1, m2, "winner_logprob", judge)
            except IndexError:        # no run of this judge on this pair yet
                continue
            by_judge[judge] = {p: r["judge_vote"] for p, r in d.items() if r["judge_vote"] in SCORE}
            # the judge's verdict with an answer-order flip counted as a tie (robustness_fig's flip_equal)
            flip[judge] = {p: "12" if r["verdict"]["winner_ab"] != r["verdict"]["winner_ba"]
                           else r["verdict"]["averaged_judge_vote"]
                           for p, r in d.items() if r["judge_vote"] in SCORE}
        # majority of the three judges per question, added 24 September: the verdict two of them
        # share, a tie when all three differ; only questions every judge decided.
        # majority_flip: the same over the flip-as-tie verdicts, for the win / tie / loss bars
        def majority(v):
            common = sorted(set.intersection(*(set(v[j]) for j in judges)))
            return {p: max(("1", "2", "12"), key=[v[j][p] for j in judges].count)
                    if len({v[j][p] for j in judges}) < 3 else "12" for p in common}
        if len(by_judge) == len(judges):
            by_judge["majority"] = majority(by_judge)
            by_judge["majority_flip"] = majority(flip)
        # each judge alone with its order flips as ties, for the per-judge win / tie / loss bars
        by_judge.update({f"{j}|flip": v for j, v in flip.items()})
        for judge, vs in by_judge.items():
            votes = list(vs.values())
            xs = 100 * np.array([SCORE[v] for v in votes])
            idx = np.random.default_rng(0).integers(0, len(xs), (B, len(xs)))   # as in robustness_fig
            rec[judge] = {"win": votes.count("1"), "tie": votes.count("12"), "loss": votes.count("2"),
                          "rate": interval(xs[idx].mean(1), xs.mean())}
        out[name] = rec
    return out


# judge runs added after the others, drawn last so the earlier intervals keep their draws
LATE_JUDGES = ("winner_logprob_bidir_qwen3_30b",)   # 24 September


def judge_overall_fig(late=False):
    """tab:judge_validation, left: each judge's kappa with the physicians' unanimous verdict on the 576
    pairs, with ties and on the pairs both called a winner, resampling pairs; and the agreement.
    late: only the LATE_JUDGES, else all the others."""
    sys.path.insert(0, str(ROOT / "auto_moove"))
    import judge_eval_unanimous576_tables as jt
    from auto_moove_check import cohen_kappa
    out = {}
    for label, tag in jt.JUDGES.items():
        rows = jt.load(tag) if (tag in LATE_JUDGES) == late else None
        if rows is None:
            continue
        h = np.array([r["human_vote"] for r in rows])
        j = np.array([r["judge_vote"] for r in rows])
        rec = {}
        for name, keep in (("ties", np.ones(len(h), bool)),
                           ("noties", np.isin(h, ["1", "2"]) & np.isin(j, ["1", "2"]))):
            hh, jj = h[keep], j[keep]
            idx = rng.integers(0, len(hh), (B, len(hh)))
            ks = [cohen_kappa(list(hh[i]), list(jj[i])) for i in idx]
            rec[name] = {"kappa": interval(ks, cohen_kappa(list(hh), list(jj))),
                         "agree": float(100 * (hh == jj).mean()), "n": int(keep.sum())}
        out[tag] = rec
    return out


def judge_majority_fig(judges=("gemma4_31b", "qwen38_27b", "museglimmer_30b")):
    """judge_overall_fig for the majority of the three judges of the main results (wtl_fig's majority: the
    verdict two of them share, a tie when all three differ), once over log-prob and once over generated
    verdicts. Seeded on its own, so the other rows keep their draws."""
    sys.path.insert(0, str(ROOT / "auto_moove"))
    import judge_eval_unanimous576_tables as jt
    from auto_moove_check import cohen_kappa
    out = {}
    for scoring in ("logprob", "generate"):
        runs = [jt.load(f"winner_{scoring}_bidir_{j}") for j in judges]
        h = np.array([r["human_vote"] for r in runs[0]])
        j = np.array([max(("1", "2", "12"), key=vs.count) if len(set(vs)) < 3 else "12"
                      for vs in zip(*([r["judge_vote"] for r in run] for run in runs))])
        rec, g = {}, np.random.default_rng(0)
        for name, keep in (("ties", np.ones(len(h), bool)),
                           ("noties", np.isin(h, ["1", "2"]) & np.isin(j, ["1", "2"]))):
            hh, jj = h[keep], j[keep]
            idx = g.integers(0, len(hh), (B, len(hh)))
            ks = [cohen_kappa(list(hh[i]), list(jj[i])) for i in idx]
            rec[name] = {"kappa": interval(ks, cohen_kappa(list(hh), list(jj))),
                         "agree": float(100 * (hh == jj).mean()), "n": int(keep.sum())}
        out[f"winner_{scoring}_majority"] = rec
    return out


class Cache:
    def __init__(self):
        self._mb, self._hb = {}, {}

    def mb(self, key):
        if key not in self._mb:
            self._mb[key] = medbench(key)
        return self._mb[key]

    def hb(self, key, split):
        if (key, split) not in self._hb:
            self._hb[(key, split)] = healthbench(key, split)[0]
        return self._hb[(key, split)]


def judge_overall_flip_fig(judges=("gemma4_31b", "qwen38_27b", "museglimmer_30b")):
    """judge_overall_fig and judge_majority_fig for log-prob scoring with an answer-order flip counted as a
    tie (wtl_fig's |flip rule, the rule of the paper's win rates), keyed as judge_overall and judge_majority.
    Generated verdicts already count a flip as a tie. Each row seeded on its own. Added 25 September."""
    sys.path.insert(0, str(ROOT / "auto_moove"))
    import judge_eval_unanimous576_tables as jt
    from auto_moove_check import cohen_kappa
    flip = lambda r: "12" if r["winner_ab"] != r["winner_ba"] else r["judge_vote"]

    def rec(h, j):
        out, g = {}, np.random.default_rng(0)
        for name, keep in (("ties", np.ones(len(h), bool)),
                           ("noties", np.isin(h, ["1", "2"]) & np.isin(j, ["1", "2"]))):
            hh, jj = h[keep], j[keep]
            idx = g.integers(0, len(hh), (B, len(hh)))
            ks = [cohen_kappa(list(hh[i]), list(jj[i])) for i in idx]
            out[name] = {"kappa": interval(ks, cohen_kappa(list(hh), list(jj))),
                         "agree": float(100 * (hh == jj).mean()), "n": int(keep.sum())}
        return out
    out = {}
    for tag in jt.JUDGES.values():
        rows = jt.load(tag) if "logprob" in tag else None
        if rows is not None:
            out[tag] = rec(np.array([r["human_vote"] for r in rows]), np.array([flip(r) for r in rows]))
    runs = [jt.load(f"winner_logprob_bidir_{j}") for j in judges]
    h = np.array([r["human_vote"] for r in runs[0]])
    j = np.array([max(("1", "2", "12"), key=vs.count) if len(set(vs)) < 3 else "12"
                  for vs in zip(*([flip(r) for r in run] for run in runs))])
    out["winner_logprob_majority"] = rec(h, j)
    return out


def length_base_fig(judges=("gemma-4-31B-it", "Qwen3.8-27B"), similar=(0.8, 1.25)):
    """tab:length_base: each finetune against its base on the unanimous set, by answer length (words). Win rate
    under the flip rule of tab:wtl on all questions, on similar-length answers (word ratio within `similar`,
    as unanimous576_h2h_tables.py) and where the finetune is >25% longer; the median word ratio, the share the
    finetune writes longer, and r between the probability margin and the log word ratio with a bootstrap
    interval over questions. Seeded per cell.
    Added 25 September."""
    words = lambda s: len((s or "").split())
    flip = lambda v: "12" if v["winner_ab"] != v["winner_ba"] else v["averaged_judge_vote"]
    pairs = (("apertus70b", "apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-Instruct-2509"),
             ("apertus8b", "apertus_8b_meditronFO_gptoss_v1", "Apertus-8B-Instruct-2509"),
             ("olmo32b", "olmo_2_32b_meditronFO_gptoss_v1", "OLMo-2-0325-32B-SFT"),
             ("euro22b", "euro_llm_22b_meditronFO_gptoss_v1", "EuroLLM-22B-Instruct-2512"),
             ("euro9b", "euro_llm_9b_meditronFO_gptoss_v1", "EuroLLM-9B-Instruct"),
             ("gemma27b", "gemma_3_27b_meditronFO_gptoss_v1", "gemma-3-27b-it"),
             ("gemma27b_medgemma", "gemma_3_27b_meditronFO_gptoss_v1", "medgemma-27b-text-it"))
    out = {}
    for name, m1, m2 in pairs:
        rec = {}
        for judge in judges:
            rows = [r for r in auto_moove(m1, m2, "winner_logprob", judge).values() if r["judge_vote"] in SCORE]
            q = np.array([words(r["model_1_answer"]) / max(1, words(r["model_2_answer"])) for r in rows])
            s = 100 * np.array([SCORE[flip(r["verdict"])] for r in rows])
            m = np.array([r["verdict"]["prob_left_mean"] - r["verdict"]["prob_right_mean"] for r in rows])
            lq = np.log(np.maximum(q, 1e-3))
            ridx = np.random.default_rng(0).integers(0, len(q), (B, len(q)))
            cell = {"median_ratio": float(np.median(q)), "ft_longer": float(100 * (q > 1).mean()),
                    "r": interval([np.corrcoef(m[i], lq[i])[0, 1] for i in ridx], np.corrcoef(m, lq)[0, 1])}
            for k, keep in (("all", np.ones(len(q), bool)), ("similar", (q >= similar[0]) & (q <= similar[1])),
                            ("ft_longer25", q > similar[1])):
                xs = s[keep]
                idx = np.random.default_rng(0).integers(0, len(xs), (B, len(xs)))
                cell[k] = {"rate": interval(xs[idx].mean(1), xs.mean()), "n": int(keep.sum())}
            rec[judge] = cell
        out[name] = rec
    return out


def ablations_am_fig(size="8b"):
    """The AutoMOOVE part of ablations_fig: each corpus ablation minus the full-corpus model, win rate
    against the base (Gemma-4, winner log-prob, ties half, an order flip a tie; ablations_fig counted the
    averaged vote) and, where the Likert runs exist, the Likert
    delta; paired by question. size="70b": the Apertus-70B ablations against Apertus-70B-Instruct, their
    full-corpus model the first (16-node) run. Seeded on its own."""
    global rng
    rng = np.random.default_rng(4 if size == "70b" else 0)
    configs = ["no_curated_qa", "no_synthetic_qa", "no_guidelines_qa", "no_synthetic_moove"]
    full, base = (("apertus_70b_meditronFO_gptoss_v1_16n", "Apertus-70B-Instruct-2509") if size == "70b" else
                  ("apertus_8b_meditronFO_gptoss_v1", "Apertus-8B-Instruct-2509"))
    prefix = f"apertus_{size}_meditronFO_gptoss_v1"
    am_name = {"full": full} | {c: f"{prefix}_{c}" for c in configs}
    order = ["full"] + configs
    win, lik = {}, {}
    for k in order:
        d = auto_moove(am_name[k], base, "winner_logprob")
        # an answer-order flip counts as a tie, as in tab:wtl and the table's own level rows
        win[k] = {p: 100 * SCORE["12" if r["verdict"]["winner_ab"] != r["verdict"]["winner_ba"]
                                 else r["verdict"]["averaged_judge_vote"]]
                  for p, r in d.items() if r["judge_vote"] in SCORE}
        try:
            s = criterion_scores(am_name[k], base)
            lik[k] = {p: np.mean([v[c][0] - v[c][1] for c in MOOVE]) for p, v in s.items()}
        except IndexError:          # no Likert run of this model (the ablations)
            pass
    ws, wp = item_boot([win[k] for k in order])
    ws, wp = ws / 100, wp / 100
    out = {c: {"automoove": pair_ci(ws, wp, i, 0)} for i, c in enumerate(configs, 1)}
    out["levels"] = {"automoove": {k: float(wp[i]) for i, k in enumerate(order)}}
    abs_rng = np.random.default_rng(5)
    for i, k in enumerate(order):           # each model's own win rate with its interval
        xs = np.array(list(win[k].values()))
        idx = abs_rng.integers(0, len(xs), (B, len(xs)))
        out.setdefault("abs", {})[k] = interval(xs[idx].mean(1), xs.mean())
    if len(lik) == len(order):
        ls, lp = item_boot([lik[k] for k in order])
        ls, lp = ls / 100, lp / 100
        for i, c in enumerate(configs, 1):
            out[c]["likert"] = pair_ci(ls, lp, i, 0)
        out["levels"]["likert"] = {k: float(lp[i]) for i, k in enumerate(order)}
    return out


def automoove_only():
    """iclr_clean_ci.py automoove: recompute every key that reads AutoMOOVE runs and keep the others
    (MCQA, HealthBench, judge validation, ...) as they are, so their draws cannot move. Keys on the
    global generator restart it at seed 0. Mixed keys (ablations, ablations_abs) only get their
    AutoMOOVE fields replaced."""
    global rng
    res = json.load(open(HERE / "iclr_clean_ci.json"))

    def fresh(fn):
        global rng
        rng = np.random.default_rng(0)
        return fn()

    res["wtl"] = wtl_fig()
    res["robustness"] = {**robustness_fig(), **robustness_fig(
        ("Muse-Glimmer-30B", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))}
    res["length"] = fresh(length_fig)
    res["length_q38"] = fresh(lambda: length_fig("Qwen3.8-27B"))
    res["length_base"] = length_base_fig()
    res["criteria"] = fresh(lambda: {**criteria_fig(), **criteria_fig(MORE_PAIRS)})
    res["criteria_q38"] = fresh(lambda: criteria_fig(RADAR_PAIRS + MORE_PAIRS, "Qwen3.8-27B"))
    ab = ablations_am_fig("8b")
    for c, v in ab.items():
        if c in ("levels", "abs"):
            continue
        res["ablations"][c]["automoove"] = v["automoove"]
        res["ablations"][c]["likert"] = v.get("likert")     # None: no Likert run of the ablations
    res["ablations"]["levels"] = ab["levels"] | {"likert": ab["levels"].get("likert")}
    for k, v in ab["abs"].items():
        res["ablations_abs"][k]["automoove"] = v
    res["ablations70_am"] = ablations_am_fig("70b")
    # the 70B full-corpus model's own win rate, for the teacher rows of the ablation figure
    res["ablations_abs"]["apertus70b_gptoss"] = {"automoove": res["ablations70_am"]["abs"]["full"]}
    write(res)


def write(res):
    """iclr_clean_ci.json, the keys the paper reads. The Likert criteria and HealthBench professional are
    computed only for their place in the random stream: dropping the computation would move the draws of
    every interval after it."""
    for k in ("criteria", "criteria_q38"):
        res.pop(k, None)
    for rec in list(res["teacher"].values()) + list(res["tables"]["teacher"].values()):
        rec.pop("professional_gemma4", None)
    json.dump(res, open(HERE / "iclr_clean_ci.json", "w"), indent=1)


def main():
    only = {"length_base": length_base_fig,                   # added 25 September"robustness_humans": robustness_humans_fig,     # added 25 September; refresh only that key
            "judge_majority": judge_majority_fig,
            "judge_overall_flip": judge_overall_flip_fig,  # added 25 September
            # both judge sets of robustness_fig, as main() merges them (rows seeded on their own)
            "robustness": lambda: {**robustness_fig(), **robustness_fig(
                ("Muse-Glimmer-30B", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))},
            "ablations_abs": ablations_abs_fig,                # added 25 September, the paper's ablation figure
            "ablations70_abs": ablations70_abs_fig}            # added 26 September, its 70B rows
    if sys.argv[1:] == ["automoove"]:
        return automoove_only()
    if len(sys.argv) == 2 and sys.argv[1] in only:
        res = json.load(open(HERE / "iclr_clean_ci.json"))
        res[sys.argv[1]] = only[sys.argv[1]]()
        write(res)
        return
    cache = Cache()
    res = {}
    for name, fn in (("judge_validation", judge_validation_fig), ("robustness", robustness_fig),
                     ("length", length_fig), ("criteria", criteria_fig),
                     ("medbench", lambda: medbench_fig(cache)), ("teacher", lambda: teacher_fig(cache)),
                     ("ablations", lambda: ablations_fig(cache)), ("tables", lambda: tables(cache)),
                     ("wtl", wtl_fig), ("criteria_more", lambda: criteria_fig(MORE_PAIRS)),
                     ("judge_overall", judge_overall_fig),
                     # the judges added on 23 September, last so the rows above keep their draws
                     ("robustness_more", lambda: robustness_fig(
                         ("Muse-Glimmer-30B", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))),
                     # Qwen3.8-27B as second judge of the six base pairs, added 24 September
                     ("criteria_q38", lambda: criteria_fig(RADAR_PAIRS + MORE_PAIRS, "Qwen3.8-27B")),
                     ("length_q38", lambda: length_fig("Qwen3.8-27B")),
                     # Qwen3-30B-A3B log-prob, added 24 September
                     ("judge_overall_late", lambda: judge_overall_fig(late=True)),
                     # MMLU-Pro, IFEval, ARC-Challenge, added 24 September (reseeds, so last)
                     ("general", general_fig),
                     # tab:judge_robustness_humans, added 25 September (seeded per row)
                     ("robustness_humans", robustness_humans_fig),
                     # three-judge majority against the physicians, added 25 September (seeded on its own)
                     ("judge_majority", judge_majority_fig),
                     # log-prob verdicts with an order flip as a tie, added 25 September (seeded on its own)
                     ("judge_overall_flip", judge_overall_flip_fig)):
        print(f"{name} ...", flush=True)
        res[name] = fn()
    res["criteria"].update(res.pop("criteria_more"))
    res["robustness"].update(res.pop("robustness_more"))
    res["judge_overall"].update(res.pop("judge_overall_late"))
    write(res)


if __name__ == "__main__":
    main()
