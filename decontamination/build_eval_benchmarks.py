#!/usr/bin/env python3
# decontamination/build_eval_benchmarks.py
"""
The benchmark prompts decontamination.py checks the training corpus against:
every benchmark MeditronFO is evaluated on, read from the split the
evaluation reads.

  medbench (lm-evaluation-harness/lm_eval/tasks/medbench_task/medbench.yaml)
    medqa_test          GBaker/MedQA-USMLE-4-options-hf, test       medqa_g
    medmcqa_val         openlifescienceai/medmcqa, validation       medmcqa_g
    medxpertqa_test     TsinghuaC3I/MedXpertQA Text, test           medxpertqa_g
    pubmedqa_labeled    qiaojin/PubMedQA pqa_labeled, train         pubmedqa_g (all 1000)
    mmlu_pro_test       TIGER-Lab/MMLU-Pro, test                    mmlu_pro
    arc_easy_test       allenai/ai2_arc ARC-Easy, test              ai2_arc
    arc_challenge_test  allenai/ai2_arc ARC-Challenge, test         ai2_arc
    ifeval              google/IFEval, train                        ifeval
  healthbench           data/healthbench/oss_eval_full.jsonl, one row per user
                        turn (all 5000 conversations; hard is a subset)
  moove_chuv            the questions of the three CHUV validatathons
  moove_phc             the questions of Kenya, Tanzania and Malawi PHC
  moove_unanimous576    the questions of the judge test set
                        (data/moove/moove_eval_unanimous_576.jsonl); 172 of its
                        576 pairs come from events outside CHUV and PHC

decontamination.py counts a training prompt as contaminated when blocks of
5+ tokens it shares with one benchmark row cover half of that row, so each row
is one question: stem plus options, as the model sees it.

python3 decontamination/build_eval_benchmarks.py \
    --out decontamination/data/decontam_prompts/meditronfo_eval
python3 decontamination/build_eval_benchmarks.py --only moove_unanimous576 \
    --out decontamination/data/decontam_prompts/unanimous576
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "auto_moove"))
from moove_meta import extract_meta, question_of  # noqa: E402

CHUV = ["CHUV - Sprint Testing", "CHUV (Centre Hospitalier Universitaire Vaudois)",
        "CHUV Research Space"]
PHC = ["Kenya PHC", "Tanzania PHC", "Malawi PHC"]


def letters(options):
    return "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))


def medqa(ex):
    return f"{ex['sent1']}\n" + letters([ex[f"ending{i}"] for i in range(4)])


def medmcqa(ex):
    return f"{ex['question']}\n" + letters([ex["opa"], ex["opb"], ex["opc"], ex["opd"]])


def medxpertqa(ex):
    # the question already lists "Answer Choices: (A) ..."; options repeats them
    return ex["question"]


def pubmedqa(ex):
    return f"{' '.join(ex['context']['contexts'])}\n{ex['question']}"


def mmlu_pro(ex):
    return f"{ex['question']}\n" + letters(ex["options"])


def arc(ex):
    return f"{ex['question']}\n" + letters(ex["choices"]["text"])


def hf(path, name, split, fmt):
    ds = load_dataset(path, name, split=split)
    return [fmt(ex) for ex in ds]


def healthbench(path):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for m in json.loads(line)["prompt"]:
                if m["role"] == "user":
                    prompts.append(m["content"])
    return prompts


def moove_questions(path, events):
    """Distinct questions of the named MOOVE events, via moove_meta."""
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = extract_meta(row).get("source_set")
            src = ", ".join(map(str, src)) if isinstance(src, (list, tuple)) else (src or "")
            q = question_of(row)
            if src in events and q:
                seen.setdefault(q.strip(), None)
    return list(seen)


def unanimous_questions(path):
    """Distinct questions of the unanimous judge test set (its own schema, not a review export)."""
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seen.setdefault(json.loads(line)["question"].strip(), None)
    return list(seen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--healthbench", default=str(ROOT / "data/healthbench/oss_eval_full.jsonl"))
    p.add_argument("--reviews", default=str(ROOT / "moove/all_reviews_no_dup.jsonl"))
    p.add_argument("--unanimous", default=str(ROOT / "data/moove/moove_eval_unanimous_576.jsonl"))
    p.add_argument("--only", nargs="+", help="build only these benchmarks")
    args = p.parse_args()

    builders = {
        "medqa_test": lambda: hf("GBaker/MedQA-USMLE-4-options-hf", None, "test", medqa),
        "medmcqa_val": lambda: hf("openlifescienceai/medmcqa", None, "validation", medmcqa),
        "medxpertqa_test": lambda: hf("TsinghuaC3I/MedXpertQA", "Text", "test", medxpertqa),
        "pubmedqa_labeled": lambda: hf("qiaojin/PubMedQA", "pqa_labeled", "train", pubmedqa),
        "mmlu_pro_test": lambda: hf("TIGER-Lab/MMLU-Pro", None, "test", mmlu_pro),
        "arc_easy_test": lambda: hf("allenai/ai2_arc", "ARC-Easy", "test", arc),
        "arc_challenge_test": lambda: hf("allenai/ai2_arc", "ARC-Challenge", "test", arc),
        "ifeval": lambda: hf("google/IFEval", None, "train", lambda ex: ex["prompt"]),
        "healthbench": lambda: healthbench(args.healthbench),
        "moove_chuv": lambda: moove_questions(args.reviews, set(CHUV)),
        "moove_phc": lambda: moove_questions(args.reviews, set(PHC)),
        "moove_unanimous576": lambda: unanimous_questions(args.unanimous),
    }
    unknown = set(args.only or []) - set(builders)
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)}")
    benches = {k: build() for k, build in builders.items() if not args.only or k in args.only}
    for name, prompts in benches.items():
        if not prompts:
            raise SystemExit(f"{name}: no prompts")
    DatasetDict({k: Dataset.from_dict({"prompt": v}) for k, v in benches.items()}).save_to_disk(args.out)
    print(f"saved to {args.out}")
    for name, prompts in benches.items():
        print(f"\n=== {name} ({len(prompts)} prompts)\n{prompts[0][:300]}")


if __name__ == "__main__":
    main()
