# Training and evaluation of the reported models

Corpora: `meditronfo_gptoss_v1` (gpt-oss-120b answers, all reported MeditronFO models) and
`meditronfo_qwen38_v3` (Qwen3.8-27B answers, second-teacher ablation) in `data/corpora.yaml`.

Everything is launched by `launch.sh <mode>` from the repository root: it trains the mode's configs
(`meditron_train.sh`, which chains medbench), then runs HealthBench and, in mode `iclr`, AutoMOOVE.
`EVAL_ONLY=1` reruns a mode's evaluations without retraining; `SKIP_TRAINED=1` trains only the configs
that have no final weights yet. `resubmit.sh` replaces a training job that died and repoints its
dependents.

## Models

Models are written to `$SCRATCH/meditronfo/models/<config>`. "Job" is the training job whose
wall-clock time is reported in the appendix compute table.

| Model | Config | Nodes | Mode | Job | HealthBench label |
|---|---|---|---|---|---|
| 8B, gpt-oss | `apertus_8b_meditronFO_gptoss_v1` | 2 | teacher | 3466501 | `meditronFO_gptoss_8b` |
| 8B, Qwen3.8 | `apertus_8b_meditronFO_qwen38_v3` | 2 | teacher | 3467265 | `meditronFO_qwen38_8b` |
| 70B, gpt-oss | `apertus_70b_meditronFO_gptoss_v1_16n` | 16 | teacher | 3468163 | `meditronFO_gptoss_70b_16n` |
| 70B, Qwen3.8 | `apertus_70b_meditronFO_qwen38_v3_16n` | 16 | teacher | 3468537 | `meditronFO_qwen38_70b_16n` |
| 70B, gpt-oss, replicate | `apertus_70b_meditronFO_gptoss_v1_normal` | 8 | replicates | 3465340 | `meditronFO_gptoss_70b_normal` |
| 70B, Qwen3.8, replicate | `apertus_70b_meditronFO_qwen38_v3_normal` | 8 | replicates | 3465342 | `meditronFO_qwen38_70b_normal` |
| 8B without Curated QA | `apertus_8b_meditronFO_gptoss_v1_no_curated_qa` | 2 | components | 3467723 | |
| 8B without Synthetic QA | `apertus_8b_meditronFO_gptoss_v1_no_synthetic_qa` | 2 | components | 3467725 | |
| 8B without Guidelines QA | `apertus_8b_meditronFO_gptoss_v1_no_guidelines_qa` | 2 | components | 3467729 | |
| 8B without synthetic MOOVE | `apertus_8b_meditronFO_gptoss_v1_no_synthetic_moove` | 2 | components | 3467733 | |
| 70B without Curated QA | `apertus_70b_meditronFO_gptoss_v1_no_curated_qa` | 8 | components70 | 3523362 | `meditronFO_gptoss_70b_no_curated_qa` |
| 70B without Synthetic QA | `apertus_70b_meditronFO_gptoss_v1_no_synthetic_qa` | 8 | components70 | 3523364 | `meditronFO_gptoss_70b_no_synthetic_qa` |
| 70B without Guidelines QA | `apertus_70b_meditronFO_gptoss_v1_no_guidelines_qa` | 8 | components70 | 3523918 | `meditronFO_gptoss_70b_no_guidelines_qa` |
| 70B without synthetic MOOVE | `apertus_70b_meditronFO_gptoss_v1_no_synthetic_moove` | 8 | components70 | 3523368 | `meditronFO_gptoss_70b_no_synthetic_moove` |
| OLMo-2-32B | `olmo_2_32b_meditronFO_gptoss_v1` | 16 | bases | 3468644 | `meditronFO_gptoss_olmo32b` |
| Gemma-3-27B | `gemma_3_27b_meditronFO_gptoss_v1` | 16 | bases | 3470197 | `meditronFO_gptoss_gemma3_27b` |
| EuroLLM-22B | `euro_llm_22b_meditronFO_gptoss_v1` | 8 | bases | 3467273 | `meditronFO_gptoss_euro22b` |
| EuroLLM-9B | `euro_llm_9b_meditronFO_gptoss_v1` | 2 | bases | 3467275 | `meditronFO_gptoss_euro9b` |

## Results

- **medbench**: `eval_results/eval-gen_<config, lowercased>_<eval job>/`; the directory of each reported
  run is in `RUN` at the top of `tech_reports/figures/iclr_clean_ci.py`. Gemma models are evaluated with
  `--chat-template none`. Runs made before the answer-filter fix of the generative MCQA tasks are read
  from the `*_rescored` files written by `scripts/rescore_medbench.py`.
- **HealthBench**: `healthbench/results/{full,hard}_gemma4_31b/results_<label>_<subset>.json`, answers
  generated with the Qwen3-30B grade first and regraded by Gemma-4-31B (`launch.sh hb`); labels in
  `HBNAME` in `iclr_clean_ci.py`. `healthbench.py` drops the oldest exchanges of a conversation that does
  not leave `--max-tokens` of room in a model's context.
- **AutoMOOVE** (`launch_automoove.sh`): `auto_moove/results/unanimous576/`, the unanimous-576 set, no
  system prompt on the answer models, winner rubric with log-prob scoring and per-criterion preference
  (`moove_pref`), both answer orders; judges Gemma-4-31B, Qwen3.8-27B and Muse-Glimmer-30B (Nemotron as
  well against MedGemma). OLMo-2-32B-SFT and EuroLLM-9B have a 4096-token context, so their pairs cap
  answers at 3072 tokens.

**Gemma-3-27B stop tokens.** Axolotl writes `gemma_3_27b_meditronFO_gptoss_v1` without a
`generation_config.json`, and its tokenizer's only EOS is `<eos>`, so vLLM would not stop at
`<end_of_turn>`. Add a `generation_config.json` with `"eos_token_id": [1, 106]` to the output directory
before the AutoMOOVE runs.
