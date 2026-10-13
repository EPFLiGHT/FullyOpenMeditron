"""gpt-oss-120b HealthBench under our graders, rescored on OpenAI's Hard and Consensus subsets.

The Hard and Consensus subsets are not separate generations: Hard is 1000 of the 5000 full conversations with
the same rubrics, and Consensus keeps 3671 conversations but only their cluster-level (consensus) criteria.
So both are recomputed here from the per-conversation grades of the full run, with the scoring of
healthbench.py and simple-evals (met points over positive points per conversation, averaged, the mean
clipped to [0, 1]) and a 95% bootstrap interval.
Checks that every subset rubric is present in the full run before scoring it.

Usage: python gptoss_official_compare.py <dir with hard_*.jsonl and consensus_*.jsonl>
Writes gptoss_official_compare.json next to this script.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "healthbench", "results")
RUNS = {
    "gemma4_31b": os.path.join(RESULTS, "full_gemma4_31b", "results_ref_gptoss_120b_full.json"),
    "qwen30b": os.path.join(RESULTS, "full_qwen30b", "results_ref_gptoss_120b_full.json"),
}
N_BOOT = 1000


def load_subset(path):
    rows = [json.loads(line) for line in open(path)]
    return {r["prompt_id"]: {(c["criterion"], c["points"]) for c in r["rubrics"]} for r in rows}


def example_score(items):
    possible = sum(i["points"] for i in items if i["points"] > 0)
    if possible == 0:
        return None
    got = sum(i["points"] for i in items if i["criteria_met"])
    return got / possible


def summarise(scores, rng):
    s = np.array(scores)
    boot = np.clip(rng.choice(s, size=(N_BOOT, len(s))).mean(axis=1), 0, 1)
    return {"score": float(np.clip(s.mean(), 0, 1)), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": len(s)}


def main(subset_dir):
    hard = load_subset(glob.glob(os.path.join(subset_dir, "hard_*.jsonl"))[0])
    consensus = load_subset(glob.glob(os.path.join(subset_dir, "consensus_*.jsonl"))[0])
    out = {}
    for grader, path in RUNS.items():
        examples = {e["prompt_id"]: e for e in json.load(open(path))["metadata"]["example_level_metadata"]}
        rng = np.random.default_rng(0)
        res = {"full": summarise([example_score(e["rubric_items"]) for e in examples.values()], rng)}

        scores = []
        for pid, rubric in hard.items():
            items = examples[pid]["rubric_items"]
            assert {(i["criterion"], i["points"]) for i in items} == rubric, pid
            scores.append(example_score(items))
        res["hard"] = summarise(scores, rng)

        scores = []
        for pid, rubric in consensus.items():
            items = [i for i in examples[pid]["rubric_items"] if (i["criterion"], i["points"]) in rubric]
            assert len(items) == len(rubric), pid
            score = example_score(items)
            if score is not None:
                scores.append(score)
        res["consensus"] = summarise(scores, rng)
        out[grader] = res
        print(grader, {k: round(100 * v["score"], 1) for k, v in res.items()})
    json.dump(out, open(os.path.join(HERE, "gptoss_official_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
