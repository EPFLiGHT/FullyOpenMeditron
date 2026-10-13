#!/usr/bin/env python3
"""LM-judge the MeditronFO synthetic training sets on the MOOVE criteria.

    pip install google-genai
    export GEMINI_API_KEY=...
    python judge_moove.py                                       # 20/subset = 60 items
    python judge_moove.py --model gemini-3.6-flash --out judged_pro.jsonl
    python judge_moove.py --model gemini-3.6-flash --out judged_pro_800.jsonl -n 200
    python judge_moove.py --items clean/items_800.jsonl --out clean/judged_clean_800.jsonl
    python judge_moove.py --corpus ../data/corpora/meditronfo_gptoss_v1 -n 200 --sample-to v1/items_sample800.jsonl
    python judge_moove.py --categories --items v1/items_sample800.jsonl --out v1/gemini-3.6-flash/judged_sample800_v1.jsonl

Resumable: rerun the same command, only missing/failed ids are sent.
Same SEED as the sampling script, so ids line up with samples.json.
--items judges fixed lists (build_clean_items.py) instead of sampling, one --out per file, and
leaves samples.json alone. --corpus samples from a data/corpora/<name>/ directory instead of SOURCES,
and --sample-to writes that sample as an --items file and stops. --categories adds the reviewers' five
error categories (CATEGORIES) to the verdict; without it the rubric and output are the July ones.

--backend vllm runs a local judge on a GPU node with the same rubric and output rows; one job loads
the model once for every --items file:
    sbatch --partition=debug --time=01:25:00 new_launch.sh --requirements none \
        corpus_audit/judge_moove.py --backend vllm --model google/gemma-4-31B-it [--thinking] \
        --items a.jsonl b.jsonl --out judged_a.jsonl judged_b.jsonl
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 2001

# USD per 1M tokens, standard (interactive) tier, July 2026. Batch is 50% of this.
# ponytail: no >200k long-context tier here, a judged item is a few thousand tokens.
# Thinking tokens bill at the output rate, so `out` covers thoughts too.
PRICING = {
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash": (0.50, 3.00),
}

ROOT = Path(__file__).resolve().parent.parent  # the repo; data/ has the same layout on the laptop and the cluster

# the corpus the clinician audited and judged_pro_800 sampled from
SOURCES = {
    "Curated QA": ROOT / "data/curated_qa/gpt_oss_8retries_v3.jsonl",
    "Synthetic QA": ROOT / "data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl",
    "Guidelines QA": ROOT / "data/guidelines_qa/guidelines_qa_v3_full.jsonl",
    "Synthetic MOOVE": ROOT / "data/synthetic_moove/synthetic_moove_v2_distillation_1try.jsonl",
}

# the same four sets after the truncation fix, the MedExpQA removal and decontamination
# (tech_reports/meditronfo_clean_report.pdf): same prompts, some answers regenerated
CLEAN_SOURCES = {
    "Curated QA": ROOT / "data/decontaminated/gpt_oss_8retries_v3_fixed_no_medexqa_decontam.jsonl",
    "Synthetic QA": ROOT / "data/decontaminated/meditron_4_synthetic_qa_v3_fixed_decontam.jsonl",
    "Guidelines QA": ROOT / "data/decontaminated/guidelines_qa_v3_full_fixed_decontam.jsonl",
    "Synthetic MOOVE": ROOT / "data/decontaminated/synthetic_moove_v2_gpt_oss_no_truncation_fixed_decontam.jsonl",
}

# file names inside a data/corpora/<name>/ directory (resampling/corpora.py)
CORPUS_FILES = {"Curated QA": "curated_qa", "Synthetic QA": "synthetic_qa",
                "Guidelines QA": "guidelines_qa", "Synthetic MOOVE": "synthetic_moove"}


def corpus_sources(d):
    return {s: Path(d) / f"{f}.jsonl" for s, f in CORPUS_FILES.items()}


CRITERIA = [
    "Clinical Soundness",
    "Safety",
    "Contextual Fit",
    "Appropriate Confidence",
    "Clarity & Completeness",
]

RUBRIC = """You are an experienced attending physician screening a single question-answer pair \
for inclusion in a medical instruction-tuning corpus. You are grading the PAIR as a training item, \
not the answer in isolation.
 
STEP 1 - Is the question answerable as written?
Before grading anything else, check the question for these defects:
  - multiple-choice stem with the answer options missing or incomplete
  - a referent that is never specified ("the antibiotic", "this agent", "the above")
  - required clinical data absent (no dose, no timepoint, no patient parameter the answer depends on)
  - the stem presupposes facts it never states
If any apply, the item is defective regardless of how well-written the answer is. Set
"question_defect": true, and score Contextual Fit at -1 or below. An answer that confidently resolves
an under-determined question is not a good answer; it is a fluent guess, and training on it teaches
the model to guess. Do not silently repair the question, do not assume the missing information, and
do not grade the answer against the question you think was intended.
 
STEP 2 - Score five criteria, integers from -2 to +2:
  Clinical Soundness      factual and clinical correctness of the content.
  Safety                  no advice that could harm a patient; red flags and the threshold for
                          escalation are recognised.
  Contextual Fit          addresses the question actually asked, at the right scope and for the
                          implied setting and audience.
  Appropriate Confidence  certainty matches the evidence. Hedging where the evidence is genuinely
                          mixed is correct. Committing to one option when the stem does not
                          determine one is NOT appropriate confidence, however fluent the
                          justification.
  Clarity & Completeness  organised, unambiguous, no material omission.
 
Anchors, identical for every criterion:
 +2 excellent, would pass unchanged in a clinical reference
 +1 good, minor nitpicks only
  0 acceptable but unremarkable, or the criterion barely applies
 -1 flawed, would need revision before training on it
 -2 seriously deficient or dangerous, should be dropped from the corpus
 
Grade what is written. Do not reward length, citation-like language, or confident tone.
 
In "notes", at most 40 words: justify every score of -1 or -2, and if question_defect is true, name
the defect first."""

# --categories: the item-level error categories the ICLR reviewers asked for, flagged per item
CATEGORIES = ["factual_error", "unsafe_recommendation", "ambiguous_question", "hallucinated_rationale",
              "label_mismatch"]

CATEGORY_STEP = """
 
STEP 3 - Flag error categories, true or false, in "defects". Flag what is present in the item, however
minor; the scores say how much it matters.
  factual_error           the answer states something medically false.
  unsafe_recommendation   the answer recommends or endorses an action that could harm a patient.
  ambiguous_question      the question has a STEP 1 defect.
  hallucinated_rationale  the reasoning rests on a mechanism, study, guideline, number or source that
                          does not exist or does not support the conclusion drawn from it.
  label_mismatch          multiple choice only: the option the answer selects is not the best option
                          given. Always false for a question without options."""

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {c: {"type": "integer"} for c in CRITERIA},
            "required": CRITERIA,
            "propertyOrdering": CRITERIA,
        },
        "notes": {"type": "string"},
    },
    "required": ["scores", "notes"],
}

# what --items keeps from a row; scores and notes of an earlier run are dropped
ITEM_KEYS = ("id", "subset", "question", "answer", "old_id", "status")


def rubric(categories):
    return RUBRIC + CATEGORY_STEP if categories else RUBRIC


def schema(categories):
    if not categories:
        return SCHEMA
    props = {**SCHEMA["properties"], "defects": {
        "type": "object", "properties": {c: {"type": "boolean"} for c in CATEGORIES},
        "required": CATEGORIES, "propertyOrdering": CATEGORIES}}
    return {"type": "object", "properties": props, "required": ["scores", "defects", "notes"],
            "propertyOrdering": ["scores", "defects", "notes"]}


def extract(conv):
    """First user turn and first assistant turn following it."""
    q = a = ""
    for m in conv:
        role, val = m.get("from"), m.get("value", "")
        if role in ("user", "human") and not q:
            q = val
        elif role in ("assistant", "gpt") and q and not a:
            a = val
    return q, a


def unescape(s):
    return (s or "").replace("\\n", "\n").replace("\\t", "\t")


def load(subset, path, n, rng):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    if n > len(lines):
        print(f"warning: {subset} has only {len(lines)} rows")
    items = []
    for i in sorted(rng.sample(range(len(lines)), min(n, len(lines)))):
        q, a = extract(json.loads(lines[i])["conversations"])
        q, a = unescape(q), unescape(a)
        if q and a:
            # id is the line number, so resume and the human review sheet point at the same row
            items.append({"id": f"{subset}:{i}", "subset": subset, "question": q, "answer": a})
    return items


def write_samples(items, path, done=None):
    """samples.json for the annotator: sequential int ids (the HTML puts them in DOM ids),
    subsets interleaved so reviewer fatigue does not correlate with data source.
    Keeps src_id, so this file is also the join key back to the judge output.
    The shuffle is seeded and independent of the scores, so ids are stable across reruns."""
    done = done or {}
    rows = list(items)
    random.Random(SEED).shuffle(rows)
    rows = [
        {
            "id": n,
            "src_id": r["id"],
            "subset": r["subset"],
            "question": r["question"],
            "answer": r["answer"],
            "lm": {k: v for k, v in done.get(r["id"], {}).items() if k in ("model", "scores", "notes")} or None,
        }
        for n, r in enumerate(rows, 1)
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} items to {path}")
    return rows


def parse(text):
    out = json.loads(text)
    scores = out["scores"]
    if set(scores) != set(CRITERIA):
        raise ValueError(f"criteria mismatch: {sorted(scores)}")
    for c, v in scores.items():
        if not isinstance(v, int) or not -2 <= v <= 2:
            raise ValueError(f"bad score for {c}: {v!r}")
    res = {"scores": scores, "notes": out.get("notes", "")}
    if "defects" in out:
        d = out["defects"]
        if set(d) != set(CATEGORIES) or not all(isinstance(v, bool) for v in d.values()):
            raise ValueError(f"bad defects: {d!r}")
        res["defects"] = d
    return res


def user_prompt(item):
    return f"<question>\n{item['question']}\n</question>\n\n<answer>\n{item['answer']}\n</answer>"


def judge(client, model, item, categories=False):
    prompt = user_prompt(item)
    err = None
    for _ in range(3):  # ponytail: fixed retries, no backoff; the SDK already retries transport errors
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": rubric(categories),
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": schema(categories),
                },
            )
            return {**parse(resp.text), "usage": usage(resp)}
        except Exception as e:  # bad JSON, out-of-range score, 429, 5xx
            err = e
    return {"error": f"{type(err).__name__}: {err}"}


def usage(resp):
    u = resp.usage_metadata
    return {
        "in": u.prompt_token_count or 0,
        "out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
    }


def cost(model, tok_in, tok_out):
    """USD, or None if the model is not in PRICING."""
    p = PRICING.get(model)
    return None if p is None else tok_in / 1e6 * p[0] + tok_out / 1e6 * p[1]


def report(rows):
    ok = [r for r in rows if "scores" in r]
    for subset in SOURCES:
        sub = [r for r in ok if r["subset"] == subset]
        if not sub:
            continue
        print(f"\n{subset}  (n={len(sub)})")
        for c in CRITERIA:
            print(f"  {c:<24} {sum(r['scores'][c] for r in sub) / len(sub):+.2f}")
        flagged = [r for r in sub if min(r["scores"].values()) <= -1]
        print(f"  items with any score <= -1: {len(flagged)} ({100 * len(flagged) / len(sub):.0f}%)")
        for r in flagged[:5]:
            print(f"    {r['id']}: {r['notes'][:100]}")
    bad = len(rows) - len(ok)
    if bad:
        print(f"\n{bad} items failed to grade, rerun to retry them")

    tokens = {}  # model -> [in, out]; rows carry their own model, so mixed files price correctly
    for r in rows:
        if "usage" in r:
            t = tokens.setdefault(r["model"], [0, 0])
            t[0] += r["usage"]["in"]
            t[1] += r["usage"]["out"]
    print()
    for model, (ti, to) in tokens.items():
        c = cost(model, ti, to)
        price = "unpriced model" if c is None else f"${c:.4f} (${c / 2:.4f} on batch)"
        print(f"{model}: in={ti:,} out={to:,} -> {price}")


def verdict_model(categories=False):
    """schema() as a pydantic model, for vLLM's grammar. The criterion names have spaces, so they
    are aliases; the scores are an enum, which the grammar enforces."""
    from typing import Literal

    from pydantic import Field, create_model

    score = Literal[-2, -1, 0, 1, 2]
    scores = create_model("Scores", **{f"c{i}": (score, Field(alias=c)) for i, c in enumerate(CRITERIA)})
    if categories:
        defects = create_model("Defects", **{c: (bool, ...) for c in CATEGORIES})
        return create_model("Verdict", scores=(scores, ...), defects=(defects, ...), notes=(str, ...))
    return create_model("Verdict", scores=(scores, ...), notes=(str, ...))


def grade_gemini(args, todo, write):
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY

    def run(job):
        item, out = job
        write(out, {**item, "model": args.model, **judge(client, args.model, item, args.categories)})

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, todo))


def grade_vllm(args, todo, write):
    """A local judge, greedy and seeded, through auto_moove's vLLM loader. The rubric is the
    system turn, as for Gemini. With --thinking the model reasons first and the JSON grammar
    applies once its reasoning ends; the reasoning is not kept."""
    import sys

    sys.path.insert(0, str(ROOT / "auto_moove"))
    from auto_moove_common import extract_judge_data, generate_answers

    schema = verdict_model(args.categories)
    prompts = [[{"role": "system", "content": rubric(args.categories)},
                {"role": "user", "content": user_prompt(item)}]
               for item, _ in todo]
    texts = generate_answers(args.model, prompts, schema=schema, thinking=args.thinking,
                             max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
                             tp=args.tp, utilization=0.85, temperature=0.0, seed=args.seed)
    for (item, out), text in zip(todo, texts):
        v = extract_judge_data(text, schema)
        res = ({"scores": v.scores.model_dump(by_alias=True), "notes": v.notes} if v
               else {"error": f"no valid verdict: {(text or '')[-200:]!r}"})
        if v and args.categories:
            res["defects"] = v.defects.model_dump()
        write(out, {**item, "model": args.model, "thinking": args.thinking, **res})


def read_items(path):
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [{k: r[k] for k in ITEM_KEYS if k in r} for r in rows if r.get("answer")]


def read_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=20, help="items per subset")
    p.add_argument("--model", default="gemini-3.6-flash")
    p.add_argument("--backend", choices=["gemini", "vllm"], default="gemini",
                   help="vllm: a local judge such as google/gemma-4-31B-it, on a GPU node")
    p.add_argument("--out", type=Path, nargs="+", default=[Path("judged.jsonl")],
                   help="one file per --items file")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--samples", type=Path, default=Path("samples.json"), help="annotator input")
    p.add_argument("--items", type=Path, nargs="+", help="judge the rows of these jsonl files (id, subset, "
                                                         "question, answer) instead of sampling; rows "
                                                         "without an answer are skipped")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--corpus", type=Path, help="sample from this data/corpora/<name>/ directory, not SOURCES")
    p.add_argument("--sample-to", type=Path, help="write the sample as an --items file and stop")
    p.add_argument("--categories", action="store_true", help="also flag the five error CATEGORIES")
    p.add_argument("--thinking", action="store_true", help="vllm: reason before the verdict")
    p.add_argument("--max-new-tokens", type=int, default=1024, help="vllm; raise it with --thinking")
    p.add_argument("--max-model-len", type=int, default=32768, help="vllm")
    p.add_argument("--tp", type=int, default=0, help="vllm: tensor parallel size, 0 = every GPU")
    p.add_argument("--seed", type=int, default=42, help="vllm")
    args = p.parse_args()
    if len(args.out) != len(args.items or [None]):
        p.error("--out takes one file per --items file")

    if args.items:
        jobs = [(read_items(path), out) for path, out in zip(args.items, args.out)]
    else:
        rng = random.Random(SEED)
        sources = corpus_sources(args.corpus) if args.corpus else SOURCES
        items = [i for s, path in sources.items() for i in load(s, path, args.n, rng)]
        if args.sample_to:
            args.sample_to.parent.mkdir(parents=True, exist_ok=True)
            args.sample_to.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                                      encoding="utf-8")
            print(f"wrote {len(items)} items to {args.sample_to}")
            return
        jobs = [(items, args.out[0])]
    jobs = [(items, out, read_done(out)) for items, out in jobs]

    if not args.report_only:
        todo = [(i, out) for items, out, done in jobs for i in items
                if i["id"] not in done or "error" in done[i["id"]]]
        print(f"{sum(len(items) for items, _, _ in jobs)} items, {len(todo)} to grade with {args.model}")
        lock = threading.Lock()
        dones, files = {out: done for _, out, done in jobs}, {}
        for _, out, _ in jobs:
            out.parent.mkdir(parents=True, exist_ok=True)
            files[out] = out.open("a", encoding="utf-8")

        def write(out, row):
            with lock:
                files[out].write(json.dumps(row, ensure_ascii=False) + "\n")
                files[out].flush()
                dones[out][row["id"]] = row
                print(".", end="", flush=True)

        (grade_vllm if args.backend == "vllm" else grade_gemini)(args, todo, write)
        for f in files.values():
            f.close()

    for items, out, done in jobs:
        if not args.items:  # samples.json is the annotator's sheet, not this run's
            write_samples(items, args.samples, done)
        if len(jobs) > 1:
            print(f"\n=== {out}")
        report([done[i["id"]] for i in items if i["id"] in done])


def selftest():
    conv = [{"from": "human", "value": "a\\nb"}, {"from": "gpt", "value": "c"}, {"from": "gpt", "value": "d"}]
    q, a = extract(conv)
    assert (unescape(q), a) == ("a\nb", "c"), (q, a)
    good = '{"scores": {%s}, "notes": "ok"}' % ", ".join(f'"{c}": 1' for c in CRITERIA)
    assert parse(good)["scores"]["Safety"] == 1
    assert abs(cost("gemini-3.6-flash", 1_000_000, 2_000_000) - 16.50) < 1e-9
    assert cost("gemini-9-nope", 1, 1) is None
    import tempfile

    items = [{"id": f"S:{i}", "subset": "S", "question": "q", "answer": "a"} for i in range(5)]
    with tempfile.TemporaryDirectory() as d:
        rows = write_samples(items, Path(d) / "s.json")
    assert [r["id"] for r in rows] == [1, 2, 3, 4, 5]
    assert sorted(r["src_id"] for r in rows) == sorted(i["id"] for i in items)
    graded = {"S:0": {"model": "m", "scores": {c: 0 for c in CRITERIA}, "notes": "n", "subset": "S"}}
    with tempfile.TemporaryDirectory() as d:
        rows = write_samples(items, Path(d) / "s.json", graded)
    by_src = {r["src_id"]: r for r in rows}
    assert set(by_src["S:0"]["lm"]) == {"model", "scores", "notes"}  # no leaked question/answer keys
    assert by_src["S:1"]["lm"] is None
    withd = good[:-1] + ', "defects": {%s}}' % ", ".join(f'"{c}": false' for c in CATEGORIES)
    assert parse(withd)["defects"]["label_mismatch"] is False and "defects" not in parse(good)
    assert rubric(False) == RUBRIC and schema(False) is SCHEMA  # the July judge is untouched
    assert schema(True)["properties"]["defects"]["required"] == CATEGORIES
    for bad in ['{"scores": {"Safety": 1}}', good.replace('"Safety": 1', '"Safety": 3'),
                withd.replace('"label_mismatch": false', '"label_mismatch": 1')]:
        try:
            parse(bad)
            raise AssertionError("should have rejected: " + bad[:40])
        except (ValueError, KeyError):
            pass
    try:
        m = verdict_model()
    except ImportError:  # pydantic is in the container and ../.venv, not the login node's python3
        m = None
    if m:
        assert list(m.model_json_schema()["$defs"]["Scores"]["properties"]) == CRITERIA
        assert m.model_validate(json.loads(good)).scores.model_dump(by_alias=True)["Safety"] == 1
        assert verdict_model(True).model_validate(json.loads(withd)).defects.label_mismatch is False
        for bad in ['{"scores": {"Safety": 1}, "notes": ""}', good.replace('"Safety": 1', '"Safety": 3')]:
            try:
                m.model_validate(json.loads(bad))
                raise AssertionError("should have rejected: " + bad[:40])
            except ValueError:
                pass
    print("ok")


if __name__ == "__main__":
    import sys

    selftest() if "--selftest" in sys.argv else main()