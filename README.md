# MeditronFO: code and results for *Fully Open Meditron*

This repository holds what is needed to reproduce the paper
[*Fully Open Meditron: An Auditable Pipeline for Clinical LLMs*](https://arxiv.org/abs/2605.16215): the data pipeline that builds
the training corpus, the training configs, the evaluations, and the scripts that turn the saved per-item
outputs of every reported run into the paper's figures and tables, together with those scripts' outputs
(every number in the paper can be checked against them).

The training corpus is on Hugging Face as
[`EPFLiGHT/fully-open-meditron`](https://huggingface.co/datasets/EPFLiGHT/fully-open-meditron), and the model
weights are in the [MeditronFO collection](https://huggingface.co/collections/EPFLiGHT/meditronfo). The
intermediate files and per-item evaluation outputs (tens of GB) are not released yet; [Data](#data) lists
what goes where.

Clone with the evaluation harness submodule: `git clone --recursive https://github.com/EPFLiGHT/FullyOpenMeditron.git`.

Two levels of reproduction:

1. **From the saved outputs (CPU, ~15 min).** Re-derive every figure and every number of the
   paper from the per-item results of the runs it reports ([Quick reproduction](#1-quick-reproduction-from-the-saved-outputs)).
2. **From scratch (GPU cluster, Slurm).** Rebuild the corpus, retrain the models, rerun every evaluation
   ([Full pipeline](#2-full-pipeline)).

## Layout

| Path | What |
|---|---|
| `papers/sept_25th_iclr_paper/figures/` | the paper's figures, where the figure scripts write them |
| `data_gen/` | question generation (Synthetic Curated QA, Guidelines QA, synthetic vignettes) and the first distillation of Curated QA |
| `truncation/` | regeneration of answers that hit the token cap (`regenerate_truncated.py`) |
| `decontamination/` | MedExpQA drop, two-stage n-gram decontamination, `decontam_table.py` (appendix table) |
| `resampling/` | the eval-free MOOVE seed pool, pass@8 rejection-sampling distillation of every answer, corpus registry |
| `data/corpora.yaml` | registry of the corpora with row counts and sha256; the paper's corpus is `meditronfo_gptoss_v1`, the second teacher's `meditronfo_qwen38_v3` |
| `axolotl_config/teacher_ablation/` | the training config of every reported model, and `launch.sh`, which trains and evaluates them |
| `meditron_train.sh`, `meditron_eval.sh`, `new_launch.sh` | Slurm launchers: Axolotl training, the lm-eval medbench suite, any Python script in the vLLM container |
| `lm-evaluation-harness/` | submodule: [our fork of the harness](https://github.com/Xkrilandar/lm-evaluation-harness) at the commit the evaluations ran with (task group `medbench`: MedMCQA, MedQA, PubMedQA, MedXpertQA, MMLU-Pro, IFEval, ARC) |
| `healthbench/` | HealthBench answering and grading (`healthbench.py`), Figure 1 (`plot_healthbench_figures.py`) |
| `auto_moove/` | AutoMOOVE: pairwise judging (`auto_moove.py`), judge validation against clinicians (`auto_moove_check.py`), tables |
| `scripts/build_unanimous_eval_set.py` | builds the unanimous-576 AutoMOOVE test set from the clinicians' reviews |
| `corpus_audit/` | clinician audit data, LLM screen of the corpus, label-mismatch table |
| `tech_reports/figures/` | the scripts that compute every interval and draw the figures, and their JSON/TeX outputs |
| `container/` | the container specs (EDF) the jobs run in |

## Data

The corpus is released as [`EPFLiGHT/fully-open-meditron`](https://huggingface.co/datasets/EPFLiGHT/fully-open-meditron)
(Parquet, OpenAI `messages` format); the rest is not released yet. The MOOVE vignettes (`data/moove/`,
`moove/all_reviews.jsonl`, and the MOOVE sets that `decontamination/build_eval_benchmarks.py` writes) are
evaluation material and are not public. The scripts expect the data at these paths:

| Path | Content |
|---|---|
| `data/corpora/meditronfo_gptoss_v1/`, `data/corpora/meditronfo_qwen38_v3/` | the training corpora, one JSONL per training set (rows and sha256 in `data/corpora.yaml`; `python3 resampling/corpora.py check`) |
| `data/curated_qa/`, `data/guidelines/`, `data/moove/full_moove.jsonl` | pipeline inputs: the public QA sets, the public clinical guidelines corpus, the MOOVE vignette pool |
| `data/synthetic_qa/`, `data/guidelines_qa/`, `data/fixed/`, `data/decontaminated/`, `data/decontam_moove_v4/` | each pipeline stage's output for the paper corpus, with the decontamination reports |
| `data/resampling/` | rejected rows and reports of the pass@8 distillation (read by `label_mismatch.py`) |
| `data/moove/moove_eval_unanimous_576.jsonl`, `moove/all_reviews.jsonl` | the AutoMOOVE test set (576 pairs, 516 vignettes) and the clinicians' reviews it is built from |
| `data/healthbench/` | HealthBench full and hard, and OpenAI's hard and consensus subsets (`openai_subsets/`) |
| `eval_results/` | lm-eval per-sample outputs of the 25 medbench runs listed in `RUN` in `tech_reports/figures/iclr_clean_ci.py` |
| `healthbench/results/` | HealthBench answers and per-conversation grades (labels in `HBNAME` in the same file) |
| `auto_moove/results/unanimous576/` | AutoMOOVE per-question judgements |
| `auto_moove/judge_outputs/` | judge-validation runs on the 576 clinician pairs |
| `truncation/results/train/truncated.jsonl` | answers flagged as truncated (input of `regenerate_truncated.py`) |

Model weights: MeditronFO checkpoints are written to `$SCRATCH/meditronfo/models/<config>`; the released
weights are in the [MeditronFO collection](https://huggingface.co/collections/EPFLiGHT/meditronfo).

## Setup

**Login-node Python** (analysis, figures, the Gemini audit judge):

```bash
bash setup_local_env.sh          # uv venv in .venv, Python 3.12, requirements-local.txt
```

`.venv` is tied to the machine's CPU architecture; build it on the machine you run from.

**Cluster jobs.** Copy `.env.example` to `.env` and fill in `PROJECT_ROOT`, `SCRATCH`, `STORAGE_ROOT`
(base-model weights), `HF_TOKEN` and WandB. Jobs run in the containers of `container/`, referenced by
path from the repository root, so submit every job from this folder:

| Spec | Image | Used by |
|---|---|---|
| `container/new_axolotl.toml` | Axolotl (2026-01-15 build), with DeepSpeed and the cut-cross-entropy plugin | `meditron_train.sh` |
| `container/inference_3.toml` | vLLM + lm-eval | `meditron_eval.sh` |
| `container/inference_4.toml` | vLLM 0.20.1 | `new_launch.sh`: data generation, HealthBench, AutoMOOVE |
| `container/vllm_030.toml` | upstream `vllm/vllm-openai:v0.30.0-aarch64` | the Muse-Glimmer-30B judge only |

The specs are enroot/pyxis EDF files for GH200 (aarch64) nodes; set `image`, `mounts` and `workdir` for
your cluster. Import the vLLM 0.30 image with `enroot import -o $SCRATCH/containers/vllm_v0.30.0.sqsh
docker://vllm/vllm-openai:v0.30.0-aarch64`; `vllm_030.toml` expects a `python` shim in
`$SCRATCH/containers/vllm_v0.30.0_bin` and pandas, dateutil and pytz in `$SCRATCH/containers/vllm_v0.30.0_extra`.

Slurm: the account comes from `--account` or your default; `RESERVATION` (optional) adds a reservation.
The paper's jobs ran on a preemptable partition with `--requeue` (`launch.sh` does this). A training job
that loses its node is resubmitted by `axolotl_config/teacher_ablation/resubmit.sh`, or unattended by
`scripts/job_watchdog.sh`.

## 1. Quick reproduction from the saved outputs

```bash
PY=.venv/bin/python
$PY tech_reports/figures/iclr_clean_ci.py            # 95% bootstrap intervals -> iclr_clean_ci.json (~10 min)
for k in ablations_abs ablations70_abs length_base automoove; do   # keys the full run does not compute,
    $PY tech_reports/figures/iclr_clean_ci.py $k                   # then every AutoMOOVE key, on its own seeds
done
$PY tech_reports/figures/iclr_pref.py                 # per-criterion AutoMOOVE -> iclr_pref.json, Figure 6, tab:pref
$PY tech_reports/figures/iclr_clean_figures.py        # every other paper figure, from iclr_clean_ci.json
$PY tech_reports/figures/healthbench_breakdown_ci.py  # HealthBench slices -> healthbench_breakdown.json
(cd tech_reports/figures && ../../$PY healthbench_breakdown_figures.py)   # heatmap, length figure, hb_tables/*.tex
$PY tech_reports/figures/gptoss_official_compare.py data/healthbench/openai_subsets
$PY healthbench/plot_healthbench_figures.py           # Figure 1
$PY decontamination/decontam_table.py                 # decontamination table
$PY corpus_audit/audit_categories.py --judge "Gemini 3.6 Flash=corpus_audit/v1/gemini-3.6-flash" \
    --judge "Gemma-4-31B=corpus_audit/v1/gemma4_31b"   # audit tables (see corpus_audit/README.md)
$PY corpus_audit/label_mismatch.py --answers "gpt-oss-120b=data/resampling/answers_gptoss_full" \
    --answers "Qwen3.8-27B=data/resampling/answers_qwen38_full"
# two figures are drawn into tech_reports/figures/ and copied into the paper's figures by hand:
cp tech_reports/figures/{iclr_medbench,iclr_judge_robustness}.pdf papers/sept_25th_iclr_paper/figures/
```

The paper's table cells are typed from these outputs; they are not generated.

### Where each result comes from

| Paper | Script | Output read |
|---|---|---|
| Fig. 1, HealthBench vs. size | `healthbench/plot_healthbench_figures.py` | `healthbench_size_paper.pdf` |
| Fig. 2, pipeline | drawing | `MeditronFO fig1 v5.pdf` |
| Fig. 3, medical benchmarks; Tab. `medbenchtable`, `general` | `iclr_clean_ci.py` (`medbench`, `general`, `tables`) → `iclr_clean_figures.py` | `iclr_medbench.pdf`, `iclr_clean_ci.json` |
| Fig. 4, judge vs. clinicians; Tab. `judge_validation` | `iclr_clean_ci.py` (`judge_validation`, `judge_overall*`, `judge_majority`) → `iclr_clean_figures.py` | `judge_validation_paper.pdf` |
| Fig. 5, HealthBench gain per theme and axis; Tab. `hb_consensus`, `hb_length` | `healthbench_breakdown_ci.py` → `healthbench_breakdown_figures.py` | `hb_heatmap_full_paper.pdf`, `hb_length_paper.pdf`, `hb_tables/*.tex` |
| Fig. 6, AutoMOOVE win/tie/loss; Tab. `wtl`, `pref` | `iclr_clean_ci.py` (`wtl`) and `iclr_pref.py` | `iclr_automoove_combined.pdf`, `iclr_wtl_{gemma4,qwen38,muse}.pdf`, `iclr_pref_paper.pdf` |
| Fig. 7, ablations and length-adjusted gain; Tab. `corpus_ablations`, `teacher` | `iclr_clean_ci.py` (`ablations`, `ablations_abs`, `ablations70_abs`, `ablations70_am`, `teacher`, `tables`) and `iclr_pref.py` → `iclr_clean_figures.py` | `iclr_ablations_teacher.pdf`, `iclr_teacher.pdf` |
| Tab. `judge_robustness`, `judge_robustness_humans` | `iclr_clean_ci.py` (`robustness`, `robustness_humans`) → `iclr_clean_figures.py` | `iclr_judge_robustness*.pdf` |
| Tab. `length`, `length_base` | `iclr_clean_ci.py` (`length`, `length_q38`, `length_base`) | `iclr_clean_ci.json` |
| Tab. `hb_calibration` | `gptoss_official_compare.py` | `gptoss_official_compare.json` |
| Tab. `decontam`, Section 2.3 counts | `decontamination/decontam_table.py` | stdout |
| Tab. `audit_clin`, `audit_v1` | `corpus_audit/audit_categories.py` | `corpus_audit/v1/audit_categories.txt` |
| Tab. `labels` | `corpus_audit/label_mismatch.py` | `corpus_audit/v1/label_mismatch.txt` |
| Tab. `training_datasets` | `resampling/corpora.py list` | `data/corpora.yaml` |
| Tab. `training_hparams` | `axolotl_config/teacher_ablation/*.yaml` | |
| Tab. `compute_main` | `sacct -j <job> --format=JobID,Elapsed,NNodes` on the training jobs listed in `axolotl_config/teacher_ablation/README.md` | |
| Unanimous-576 test set | `scripts/build_unanimous_eval_set.py --input moove/all_reviews.jsonl --output data/moove/moove_eval_unanimous_576.jsonl` | |

Which run feeds which key: `RUN` (medbench directories) and `HBNAME` (HealthBench labels) at the top of
`tech_reports/figures/iclr_clean_ci.py`, and `axolotl_config/teacher_ablation/README.md`.

## 2. Full pipeline

Every GPU step runs in the vLLM container through `new_launch.sh` (`--requirements none`: the container
file system is read-only). The commands below are the ones that produced the paper's files; each stage's
output is kept in `data/`, so any stage can be rerun alone.

### 2.1 Test sets

```bash
python3 scripts/build_unanimous_eval_set.py --input moove/all_reviews.jsonl \
    --output data/moove/moove_eval_unanimous_576.jsonl           # 576 pairs, 516 questions
python3 decontamination/build_eval_benchmarks.py --out decontamination/data/decontam_prompts/meditronfo_eval
```

### 2.2 Questions and first answers (gpt-oss-120b)

```bash
# Curated QA: the seven public QA sets, answered with 8 tries
bash new_launch.sh --requirements none data_gen/distill_with_retries_v3.py \
    --input data/curated_qa/meditron_4_cleaned.jsonl --output data/curated_qa/gpt_oss_8retries_v3.jsonl \
    --model openai/gpt-oss-120b
# Synthetic Curated QA: 5 random Curated QA examples in, one new question out
bash new_launch.sh --requirements none data_gen/distill_synthetic_qa_v3.py \
    --input data/curated_qa/gpt_oss_8retries_v3.jsonl --output data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl \
    --model openai/gpt-oss-120b
# Guidelines QA: ten vignettes per guideline
bash new_launch.sh --requirements none data_gen/distill_guidelines_v3.py \
    --input data/guidelines/guidelines_epfl_llm_no_wikidoc_final.jsonl \
    --output data/guidelines_qa/guidelines_qa_v3_full.jsonl --model openai/gpt-oss-120b
# Synthetic vignettes (v4): seed pool without any evaluation question, then generation
python3 resampling/moove_seed_pool.py --pool data/moove/full_moove.jsonl --out data/moove/full_moove_eval_free.jsonl
bash new_launch.sh --requirements none data_gen/distill_moove_v2.py \
    --input data/moove/full_moove_eval_free.jsonl --output data/moove/synthetic_moove_v4_eval_free.jsonl \
    --model openai/gpt-oss-120b
```

The generation prompts are printed verbatim in the paper's appendix.

### 2.3 Truncated answers, MedExpQA, decontamination

```bash
bash new_launch.sh --requirements none truncation/regenerate_truncated.py \
    --model openai/gpt-oss-120b --outdir data/fixed \
    --input data/curated_qa/gpt_oss_8retries_v3.jsonl data/synthetic_qa/meditron_4_synthetic_qa_v3.jsonl \
            data/guidelines_qa/guidelines_qa_v3_full.jsonl
python3 decontamination/drop_sources.py --source /medexpqa/ \
    --input data/fixed/gpt_oss_8retries_v3_fixed.jsonl --output data/fixed/gpt_oss_8retries_v3_fixed_no_medexqa.jsonl
sbatch decontamination/decontaminate_corpus.sh data/decontaminated \
    data/fixed/gpt_oss_8retries_v3_fixed_no_medexqa.jsonl data/fixed/meditron_4_synthetic_qa_v3_fixed.jsonl \
    data/fixed/guidelines_qa_v3_full_fixed.jsonl
```

See `truncation/README.md` for why and what `regenerate_truncated.py` changes. The v4 vignettes are
checked against the evaluation set and, separately, the unanimous-576 set. The two report
directories are merged into `data/decontam_moove_v4/combined_reports` and applied to the answered file
in 2.4:

```bash
sbatch decontamination/decontaminate_corpus.sh <scratch>/decontam_moove_v4/meditronfo_eval \
    data/moove/synthetic_moove_v4_eval_free.jsonl
sbatch --export=ALL,BENCH=decontamination/data/decontam_prompts/unanimous576,BENCH_ONLY=moove_unanimous576 \
    decontamination/decontaminate_corpus.sh <scratch>/decontam_moove_v4/unanimous576 \
    data/moove/synthetic_moove_v4_eval_free.jsonl
```

### 2.4 Answers: pass@8 rejection sampling

`resampling/README.md` documents the method. Every answer is regenerated; a multiple-choice row keeps a
randomly chosen sample that reaches the label, in waves of 1, 3 and 4 samples (pass@8, temperature
0.7); rows that never do are dropped to `*_rejected.jsonl`. Eight one-node shards per set:

```bash
bash new_launch.sh --requirements none --array 8 resampling/resample_answers.py --model openai/gpt-oss-120b \
    --input data/decontaminated/gpt_oss_8retries_v3_fixed_no_medexqa_decontam.jsonl \
    --input data/decontaminated/meditron_4_synthetic_qa_v3_fixed_decontam.jsonl \
    --input data/decontaminated/guidelines_qa_v3_full_fixed_decontam.jsonl \
    --outdir <scratch>/resampling/answers_gptoss_full --shard auto
bash new_launch.sh --requirements none --array 4 resampling/resample_answers.py --model openai/gpt-oss-120b \
    --input data/moove/synthetic_moove_v4_eval_free.jsonl --outdir <scratch>/resampling/answers_gptoss_moove_v4 --shard auto
python3 resampling/merge_shards.py --outdir <scratch>/resampling/answers_gptoss_full    # and the moove dir
# drop the flagged vignettes (2.3) from the answered set
python3 decontamination/apply_reports.py \
    --input <scratch>/resampling/answers_gptoss_moove_v4/synthetic_moove_v4_eval_free_resampled.jsonl \
    --reports data/decontam_moove_v4/combined_reports \
    --output data/decontam_moove_v4/gptoss_synthetic_moove_v4_decontam.jsonl
python3 resampling/corpora.py register meditronfo_gptoss_v1 --model openai/gpt-oss-120b \
    --file curated_qa=... --file synthetic_qa=... --file guidelines_qa=... --file synthetic_moove=...
```

The second teacher (`meditronfo_qwen38_v3`) is the same with `--model Qwen/Qwen3.8-27B`.

### 2.5 Training

```bash
bash axolotl_config/teacher_ablation/launch.sh teacher       # Apertus-8B/70B on both teachers (70B: 16 nodes)
bash axolotl_config/teacher_ablation/launch.sh replicates    # second 70B run per teacher (training noise)
bash axolotl_config/teacher_ablation/launch.sh components    # Apertus-8B minus one training set, x4
bash axolotl_config/teacher_ablation/launch.sh components70  # the same on Apertus-70B
bash axolotl_config/teacher_ablation/launch.sh bases         # OLMo-2-32B, Gemma-3-27B, EuroLLM-22B/9B
```

Each mode trains its configs (seed 42, sequence length 8192, packing) and chains medbench after training
and HealthBench after that. Give every config its own `dataset_prepared_path`: Axolotl silently reuses a
stale tokenised cache. Gemma-3-27B-MeditronFO needs a `generation_config.json` with
`"eos_token_id": [1, 106]` in its output directory, or vLLM never stops at `<end_of_turn>`.

### 2.6 Evaluation

```bash
# medbench (temperature 0, seed 42); Gemma models with --chat-template none
bash meditron_eval.sh <model>
python3 scripts/rescore_medbench.py <eval_results dir>     # only runs made before the 2026-09-22 filter fix
# untrained bases, reference models, Figure 1 points, HealthBench (answers, then Gemma-4-31B grade)
bash axolotl_config/teacher_ablation/launch.sh references
bash axolotl_config/teacher_ablation/launch.sh timeline
bash axolotl_config/teacher_ablation/launch.sh openweights
# medbench and HealthBench of the untrained bases, MedGemma-27B and the 8B ablations
bash axolotl_config/teacher_ablation/launch.sh iclr
# AutoMOOVE on the unanimous-576 set: every pair, all judges; then the corpus ablations
bash axolotl_config/teacher_ablation/launch_automoove.sh
bash axolotl_config/teacher_ablation/launch_automoove.sh ablations8b
AFTER=<last job of the Apertus-70B chain> bash axolotl_config/teacher_ablation/launch_automoove.sh ablations70b
```

The judge validation runs on the 576 clinician pairs (outputs in `auto_moove/judge_outputs/`):

```bash
bash new_launch.sh --requirements none auto_moove/auto_moove_check.py \
    --input data/moove/moove_eval_unanimous_576.jsonl \
    --output auto_moove/judge_outputs/judged_unanimous576_winner_logprob_bidir_gemma4_31b.jsonl \
    --judge google/gemma-4-31B-it --rubric winner --scoring logprob --bidirectional --utilization 0.5
#   --scoring generate                                   -> *_winner_generate_bidir_<judge>.jsonl
#   --rubric moove_pref --scoring generate --max-model-len 16384 -> *_moove_pref_generate_bidir_gemma4_31b.jsonl
#   judges: Qwen/Qwen3.8-27B, Qwen/Qwen3-30B-A3B-Instruct-2507, nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16,
#   and meta-models/Muse-Glimmer-30B (sbatch --environment=container/vllm_030.toml)
```

The corpus audit (Gemini 3.6 Flash from the login node, Gemma-4-31B on a GPU node) is described step by
step in `corpus_audit/README.md`.

## AutoMOOVE answer generation

The answering models get the vignette alone, with no system prompt: temperature 0.7, seed 42, up to 8192
new tokens (3072 for the two models with a 4096-token context).

## Checked on 2026-09-26

Section 1 was rerun from this folder, with the data in place, on a fresh Python 3.12 environment
(`requirements-local.txt`):

- all 16 generated figures of the paper regenerate pixel-identical;
- `iclr_clean_ci.json`, `iclr_pref.json`, `healthbench_breakdown.json` and `gptoss_official_compare.json`
  match the paper's values (differences below 1e-3, i.e. floating-point noise);
- `decontam_table.py`, `audit_categories.py` and `label_mismatch.py` reproduce the
  saved tables exactly;
- `build_unanimous_eval_set.py` rebuilds the AutoMOOVE test set byte for byte from the clinicians' reviews.

## Known gaps

- **Figure 2** is a drawing, not generated.
- **Table cells are typed** into the paper from the JSON and text outputs above, not generated.
- **Intermediate data, per-item evaluation outputs and container images** are not part of this repository
  (see [Data](#data) and [Setup](#setup)), so section 1 cannot be rerun from a fresh clone yet.
- vLLM's continuous batching makes regenerated answers differ slightly from run to run even with a fixed
  seed, so a rerun from scratch reproduces the numbers within their intervals, not to the digit.
