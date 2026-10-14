# Corpus audit

The clinician audit of 60 items of an earlier version of the corpus (exported 25 July 2026), and the LLM
screen of the paper's corpus (`meditronfo_gptoss_v1`) with two judges, Gemini 3.6 Flash and Gemma-4-31B.
Paper: Section 2.3 and appendix tables `audit_clin`, `audit_v1` and `labels`.

| File | What it is |
|---|---|
| `sample_synthetic_data_for_audit.py` | 20 items per subset into `data/samples.json`, the clinician's input |
| `data/meditronfo_audit_2026-07-25.csv` | the clinician's export: scores on the five MOOVE criteria, error flags, verdict and comments on the 60 items |
| `judge_moove.py` | the LLM judge: five MOOVE criteria, -2 to +2, one QA pair at a time; `--categories` adds five yes/no error flags (factual error, unsafe recommendation, ambiguous question, hallucinated rationale, label mismatch) |
| `judge_outputs/judged_v4.jsonl` | the 60 audited items with the judge rubric of the paper |
| `judge_outputs/judged_pro_800.jsonl` | 200 items from each training set of the earlier corpus |
| `build_clean_items.py` | finds the items of a judged run again in another corpus |
| `v1/` | the screen of `meditronfo_gptoss_v1`: the items (`items_*.jsonl`), one folder of judge outputs per judge, and the printed results (`*.txt`) |
| `audit_categories.py` | the error categories against the clinician's flags, their rates in v1, the clinician's rejects in v1 (`v1/audit_categories.txt`) |
| `compare_judges.py` (uses `compare_clean.py`) | the two judges side by side, against the clinician and on the earlier vs v1 corpus (`v1/compare_judges.txt`) |
| `analyze_audit.py`, `analyze_corpus.py` | clinician vs judge agreement per criterion; defect rates of a judged run corrected with the audit's sensitivity and specificity |
| `label_mismatch.py` | multiple-choice rows whose label the answering model never reached in 8 samples (`v1/label_mismatch.txt`) |

## Reproduce

Run from the repository root. The Gemini judge needs `google-genai` (in `requirements-local.txt`) and
`GEMINI_API_KEY` in `.env`; the analysis scripts need only the standard library and Python 3.12.

```bash
V1=data/corpora/meditronfo_gptoss_v1
python3 corpus_audit/build_clean_items.py corpus_audit/judge_outputs/judged_pro_800.jsonl --corpus $V1 --out corpus_audit/v1/items_800.jsonl
python3 corpus_audit/build_clean_items.py corpus_audit/judge_outputs/judged_v4.jsonl --corpus $V1 --out corpus_audit/v1/items_audit60.jsonl
.venv/bin/python corpus_audit/judge_moove.py --corpus $V1 -n 200 --sample-to corpus_audit/v1/items_sample800.jsonl

ITEMS="corpus_audit/judge_outputs/judged_v4.jsonl corpus_audit/v1/items_audit60.jsonl \
       corpus_audit/judge_outputs/judged_pro_800.jsonl corpus_audit/v1/items_800.jsonl corpus_audit/v1/items_sample800.jsonl"
outs() { for k in audit60_old audit60_v1 800_old 800_v1 sample800_v1; do echo $1/judged_$k.jsonl; done; }

# Gemma-4-31B on a GPU node (about 10 minutes)
sbatch --time=01:25:00 new_launch.sh --requirements none \
    corpus_audit/judge_moove.py --backend vllm --model google/gemma-4-31B-it --categories \
    --items $ITEMS --out $(outs corpus_audit/v1/gemma4_31b)
# Gemini 3.6 Flash from a login node
set -a; source .env; set +a
.venv/bin/python corpus_audit/judge_moove.py --categories --items $ITEMS --out $(outs corpus_audit/v1/gemini-3.6-flash)

.venv/bin/python corpus_audit/audit_categories.py --judge "Gemini 3.6 Flash=corpus_audit/v1/gemini-3.6-flash" \
    --judge "Gemma-4-31B=corpus_audit/v1/gemma4_31b" > corpus_audit/v1/audit_categories.txt
.venv/bin/python corpus_audit/compare_judges.py --new v1 --items corpus_audit/v1/items_800.jsonl \
    --audit-items corpus_audit/v1/items_audit60.jsonl \
    --judge "Gemini 3.6 Flash=corpus_audit/v1/gemini-3.6-flash" --judge "Gemma-4-31B=corpus_audit/v1/gemma4_31b" \
    > corpus_audit/v1/compare_judges.txt
.venv/bin/python corpus_audit/label_mismatch.py \
    --answers "gpt-oss-120b=data/resampling/answers_gptoss_full" \
    --answers "Qwen3.8-27B=data/resampling/answers_qwen38_full" > corpus_audit/v1/label_mismatch.txt
```

The judge is resumable: rerunning a command sends only the items that are missing or failed.
`label_mismatch.py` reads the `*_resampled.jsonl` and `*_rejected.jsonl` files that
`resampling/resample_answers.py` writes (see the top-level README for the data).
