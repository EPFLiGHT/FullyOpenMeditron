# Fully Open Meditron

**An Auditable Pipeline for Clinical LLMs**

This repository contains the code, data generation pipelines, and evaluation infrastructure for *Fully Open Meditron*, the first end-to-end fully open pipeline for building Clinical Decision Support LLMs (LLM-CDSS). Every stage — corpus construction, synthetic data generation, training, and evaluation — is reproducible from this repo.

If you only want to reproduce a specific stage, jump to the relevant section. If you want the full pipeline, follow the sections in order.

---

## What's in this repo

- `data_gen/` — synthetic data generation (Curated QA, Guidelines QA, Synthetic MOOVE) with rejection-sampling-based gold-label resampling
- `auto_moove/` — Auto-MOOVE: our LLM-as-a-judge pairwise clinical evaluation protocol, validated against 204 human raters
- `healthbench/` — HealthBench evaluation runner
- `meditron_train.sh` / `meditron_eval.sh` — training and benchmark evaluation entry points
- `new_launch.sh` — Slurm launcher used throughout

---

## Models

We release MeditronFO finetunes of five fully open base models plus one open-weight control:

| Base | Size | Type |
|---|---|---|
| Apertus-Instruct-2509 | 70B / 8B | Fully open |
| OLMo-2-SFT | 32B | Fully open |
| EuroLLM-Instruct | 22B / 9B | Fully open |
| Gemma-3-IT | 27B | Open-weight (control vs. MedGemma) |

## Setup

The pipeline targets a Slurm + vLLM environment (originally run on CSCS Clariden). All long-running jobs are launched via `new_launch.sh`, which handles container setup, GPU allocation, and vLLM server lifecycle.

```bash
bash new_launch.sh <script.py> [args...]
```

Paths in the examples below reflect the Clariden layout (`/capstor/...`, `/users/theimer/...`). Replace with your own paths when reproducing.

---

## 1. Data Generation

The Fully Open Meditron corpus has three synthetic components, each seeded from a different source pool. All generators use `gpt-oss-120b` as the default teacher. Items with labeled answers are resampled up to 8 times at temperature 0.7 until the extracted letter matches the gold label.

### Curated QA distillation (rejection sampling on existing QA)

```bash
bash new_launch.sh data_gen/distill_with_retries_v4.py \
  --input  data/curated_qa/meditron_4_cleaned.jsonl \
  --output data/curated_qa/gpt_oss_8retries_format_v4.jsonl \
  --model  openai/gpt-oss-120b
```

Swap `--model` for `Qwen/Qwen3-30B-A3B-Instruct-2507` or `google/medgemma-27b-text-it` to reproduce the teacher-choice ablation. Use `--max-tries 1` to disable rejection sampling.

### Synthetic Curated QA (novel exam-style QA)

```bash
bash new_launch.sh data_gen/distill_synthetic_qa_v3.py \
  --input  data/curated_qa/gpt_oss_8retries_v3.jsonl \
  --output data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl \
  --model  openai/gpt-oss-120b
```

### Guidelines QA (grounded in clinical practice guidelines)

Seeded from the GUIDELINES corpus (46,469 articles across 16 institutions).

```bash
bash new_launch.sh data_gen/distill_guidelines_v3.py \
  --input  data/guidelines/guidelines_epfl_llm_no_wikidoc_final.jsonl \
  --output data/guidelines_qa/guidelines_qa_v3_full.jsonl \
  --model  openai/gpt-oss-120b
```

### Synthetic MOOVE (open-ended clinical vignettes)

Two-step: generate vignette prompts, then generate teacher answers.

```bash
# Step 1: generate vignettes from MOOVE training split
bash new_launch.sh data_gen/distill_moove_v2.py \
  --input  data/moove/full_moove.jsonl \
  --output data/synthetic_moove/synthetic_moove_v2.jsonl \
  --model  openai/gpt-oss-120b

# Step 2: generate teacher answers
bash new_launch.sh data_gen/distill_with_retries_v3.py \
  --input  data/synthetic_moove/synthetic_moove_v2.jsonl \
  --output data/synthetic_moove/synthetic_moove_v2_gpt_oss_1try.jsonl \
  --model  openai/gpt-oss-120b \
  --max_tries 1
```

> **Decontamination.** Before training, the full corpus is decontaminated against every evaluation reference (MedQA, MedMCQA, PubMedQA, MedXpertQA, MOOVE eval split, HealthBench Hard, MMLU-Pro, IFEval, ARC-Challenge) using the two-stage n-gram + token-alignment pipeline from Apertus. See `decontamination/` for thresholds and per-benchmark removal counts.

---

## 2. Training

Training uses Axolotl. Per-base configs live under `configs/`; instruction-tuning formats native to each base are preserved.

```bash
bash meditron_train.sh path/to/axolotl_config.yaml
```

Per-model hyperparameters and the optional 10% Tülu replay mixture (for instruction-following retention) are documented in `configs/README.md`.

---

## 3. Evaluation

### Medical and general benchmarks

```bash
bash meditron_eval.sh path/to/weights
```

Runs MedQA, MedMCQA, PubMedQA, MedXpertQA, MMLU-Pro, IFEval, and ARC-Challenge at temperature 0.0.

### HealthBench

```bash
bash new_launch.sh healthbench/healthbench.py \
    --model   /path/to/your/checkpoint \
    --grader  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp      4
```

### Auto-MOOVE (pairwise LLM-as-a-judge)

Auto-MOOVE compares two models on MOOVE clinical vignettes across nine criteria (question comprehension, logical reasoning, relevance & completeness, harmlessness, fairness, contextual awareness, communication, clarity, alignment with guidelines), with random answer-order swapping to mitigate positional bias.

**Base vs. MeditronFO finetune:**

```bash
bash new_launch.sh auto_moove/auto_moove.py \
  --input  data/moove/full_moove.jsonl \
  --output results/auto_moove_apertus70b.jsonl \
  --model1 /path/to/Apertus-70B-Instruct-2509 \
  --model2 /path/to/apertus_70b_meditron_fo \
  --judge  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
```

**Cross-model comparison (Gemma-MeditronFO vs. MedGemma):**

```bash
bash new_launch.sh auto_moove/auto_moove.py \
  --input  data/moove/full_moove.jsonl \
  --output results/medgemma_vs_meditron.jsonl \
  --model1 google/medgemma-27b-text-it \
  --model2 /path/to/gemma_3_27b_meditron_fo \
  --judge  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
```

### Validating the judge against human raters

To reproduce the judge-vs-human Cohen's κ analysis:

```bash
bash new_launch.sh auto_moove/auto_moove_check_distinguishable.py \
  --input  data/moove/full_moove.jsonl \
  --output data/auto-moove-dpo-eval-fullmoove.jsonl \
  --judge  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
```

---

## 4. Reproducing the ablations

| Ablation | How to reproduce |
|---|---|
| Remove Curated QA / Synthetic Curated QA / Guidelines QA / Synthetic MOOVE | Train Apertus-70B-Instruct on the corresponding corpus subset using configs in `configs/ablations/` |
| Tülu replay (0% → 10%) | Set `replay_fraction: 0.1` in the Axolotl config |
| Teacher choice (gpt-oss-120b vs. Qwen3-30B) | Re-run `distill_with_retries_v4.py` and `distill_synthetic_qa_v3.py` with `--model Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Judge sensitivity | Re-run Auto-MOOVE with the alternative judge (see above) |

All ablation results in Table 2 and Table 12 of the paper are reproducible from these knobs alone.

---

## Citation

```bibtex
TODO
```

---

## License and intended use

The corpus is released under a research-use license. Models and data are intended for research on auditable clinical LLMs. **They are not approved for clinical deployment.** Downstream users should perform domain-specific safety evaluation before any deployment-adjacent use, and should be aware of the limitations documented in the paper.