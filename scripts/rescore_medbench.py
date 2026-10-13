#!/usr/bin/env python3
# scripts/rescore_medbench.py
"""
Rescore saved medbench generations with the answer filters now in the task
YAMLs (lm-evaluation-harness/lm_eval/tasks/medbench_task/).

The strict-final-answer filters of medqa_g, medmcqa_g, pubmedqa_g and
medxpertqa_g used to be (?i)Answer[^A-X]*...([A-X])\\b, run on the first
"answer" in the response. Two failures follow:

  * A-J (MedXpertQA): the pattern stops at the "i" of "is", so it never parsed
    "The answer is X." -- the ending the resampled MeditronFO corpora teach --
    and scored those answers [invalid].
  * A-D: case-insensitive, so "To answer this question, we need ..." scored as
    "d". Apertus-8B-Instruct's PubMedQA was 0.292 for that reason.

Generation is greedy and the filter runs after it, so rescoring the saved
samples gives what a fresh evaluation would, without the run-to-run noise a
regeneration adds.

For every results_<ts>.json, writes next to the originals (which are left
untouched):
    samples_<task>_<ts>_rescored.jsonl   filtered_resps / exact_match redone
    results_<ts>_rescored.json           the four tasks and the medbench
                                         group's strict-final-answer redone

    scripts/rescore_medbench.py eval_results/eval-gen_<run>_<job> [...]
"""

import argparse
import glob
import json
import math
import os
import string
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lm-evaluation-harness"))

from lm_eval.filters.extraction import RegexFilter  # noqa: E402

TASK_DIR = os.path.join(ROOT, "lm-evaluation-harness/lm_eval/tasks/medbench_task")
TASKS = ["medmcqa_g", "medqa_g", "pubmedqa_g", "medxpertqa_g"]
FILTER, GROUP = "strict-final-answer", "medbench"
KEY, SE_KEY = f"exact_match,{FILTER}", f"exact_match_stderr,{FILTER}"


class _Loader(yaml.SafeLoader):
    pass


# task YAMLs may carry "!function module.fn"; only the filter and metric matter here
_Loader.add_multi_constructor("!", lambda loader, suffix, node: None)


# Copies of lm_eval.api.metrics (exact_match_hf_evaluate, mean_stderr,
# aggregate_subtask_metrics, pooled_sample_stderr); importing that module needs
# sacrebleu and friends, which the login-node .venv does not have.
def exact_match(pred, ref, ignore_case=False, ignore_punctuation=False):
    if ignore_case:
        pred, ref = pred.lower(), ref.lower()
    if ignore_punctuation:
        table = str.maketrans("", "", string.punctuation)
        pred, ref = pred.translate(table), ref.translate(table)
    return float(pred == ref)


def mean_stderr(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def aggregate_subtask_metrics(metrics, sizes):
    return sum(m * n for m, n in zip(metrics, sizes)) / sum(sizes)


def pooled_sample_stderr(stderrs, sizes):
    var = sum((n - 1) * se ** 2 * n for n, se in zip(sizes, stderrs)) / (sum(sizes) - len(sizes))
    return math.sqrt(var / sum(sizes))


def build_filter(filter_list):
    (flt,) = [f for f in filter_list if f["name"] == FILTER]
    regex, take_first = flt["filter"]
    assert regex["function"] == "regex" and take_first["function"] == "take_first"
    return RegexFilter(**{k: v for k, v in regex.items() if k != "function"})


def load_task(task):
    with open(os.path.join(TASK_DIR, f"{task}.yaml")) as fh:
        cfg = yaml.load(fh, Loader=_Loader)
    assert cfg["task"] == task
    (metric,) = cfg["metric_list"]
    match_kwargs = {k: metric[k] for k in ("ignore_case", "ignore_punctuation") if k in metric}
    return cfg["filter_list"], build_filter(cfg["filter_list"]), match_kwargs


def rescore_samples(path, flt, match_kwargs):
    rows = [json.loads(line) for line in open(path)]
    preds = [p[0] for p in flt.apply([r["resps"][0] for r in rows], [r["doc"] for r in rows])]
    for r, p in zip(rows, preds):
        r["filtered_resps"] = [p]
        r["exact_match"] = exact_match(p, r["target"], **match_kwargs)
    return rows


def rescore_group(res, sizes):
    """Redo the medbench strict-final-answer aggregate the way lm-eval builds it."""
    subtasks = [t for t in res["group_subtasks"][GROUP] if KEY in res["results"].get(t, {})]
    ns = [sizes[t] for t in subtasks]
    res["results"][GROUP][KEY] = aggregate_subtask_metrics([res["results"][t][KEY] for t in subtasks], ns)
    res["results"][GROUP][SE_KEY] = pooled_sample_stderr([res["results"][t][SE_KEY] for t in subtasks], ns)


def rescore_run(res_path, tasks):
    stem = os.path.basename(res_path)[len("results_"):-len(".json")]
    res = json.load(open(res_path))
    sizes = {t: v["effective"] for t, v in res["n-samples"].items()}

    # self-check: the stored group score must come back from the stored subtask scores
    old_group = dict(res["results"][GROUP])
    rescore_group(res, sizes)
    assert abs(res["results"][GROUP][KEY] - old_group[KEY]) < 1e-9, (res_path, old_group[KEY])
    assert abs(res["results"][GROUP][SE_KEY] - old_group[SE_KEY]) < 1e-9, (res_path, old_group[SE_KEY])

    original, line = {}, []
    for task in TASKS:
        samples = os.path.join(os.path.dirname(res_path), f"samples_{task}_{stem}.jsonl")
        if task not in res["results"] or not os.path.exists(samples):
            continue
        filter_list, flt, match_kwargs = tasks[task]
        old = res["results"][task][KEY]

        # self-check: the filter the run used must give back its stored score
        replay = [r["exact_match"] for r in
                  rescore_samples(samples, build_filter(res["configs"][task]["filter_list"]), match_kwargs)]
        assert abs(sum(replay) / len(replay) - old) < 1e-9, (samples, old, sum(replay) / len(replay))

        rows = rescore_samples(samples, flt, match_kwargs)
        assert len(rows) == sizes[task], (samples, len(rows), sizes[task])
        scores = [r["exact_match"] for r in rows]
        res["results"][task][KEY] = sum(scores) / len(scores)
        res["results"][task][SE_KEY] = mean_stderr(scores)
        res["configs"][task]["filter_list"] = filter_list
        original[task] = old
        with open(samples[:-len(".jsonl")] + "_rescored.jsonl", "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        invalid = sum(r["filtered_resps"][0] == "[invalid]" for r in rows)
        line.append(f"{task[:-2]} {old:.3f}->{res['results'][task][KEY]:.3f} ({invalid} inv)")

    rescore_group(res, sizes)
    original[GROUP] = old_group[KEY]
    res["rescored"] = {"filters_from": os.path.relpath(TASK_DIR, ROOT),
                       "original_results": os.path.basename(res_path), "original": original}
    with open(res_path[:-len(".json")] + "_rescored.json", "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    line.append(f"{GROUP} {old_group[KEY]:.3f}->{res['results'][GROUP][KEY]:.3f}")
    return stem, line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dirs", nargs="+")
    args = ap.parse_args()
    tasks = {t: load_task(t) for t in TASKS}

    for d in args.eval_dirs:
        for res_path in sorted(glob.glob(os.path.join(d, "*", "results_*.json"))):
            if res_path.endswith("_rescored.json"):
                continue
            stem, line = rescore_run(res_path, tasks)
            print(f"{os.path.basename(d)}  {stem}\n    " + "  ".join(line))


if __name__ == "__main__":
    main()
