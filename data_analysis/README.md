# Data analysis

Source vs synthetic comparison for medical QA corpora (MOOVE, Curated QA,
Guidelines). Produces composition pies and per-field divergence statistics
with bootstrap CIs.

For environment setup, container images, and parent-repo conventions, see the
top-level repository README. This subdirectory is self-contained — all imports
resolve relative to `data_analysis/`.

## Pipeline

| Stage | Script | GPU | Purpose |
|---|---|---|---|
| 0 | `scripts/00b_dataset_composition.py` | no | Record + token counts; pie charts |
| 1 | `scripts/01_extract_metadata.py`     | yes | Tag every record with clinical metadata via Qwen3-32B |
| 2 | `scripts/02_divergence.py`           | no | JSD / TV / Wasserstein per (pair, field) with 95% bootstrap CIs |

Diagnostic: `scripts/00_sanity.py` runs basic per-file checks on the raw inputs.

## Layout

```
data_analysis/
├── README.md
├── configs/
│   └── default.yaml                  # data paths (override via --config or CLI flags)
├── scripts/
│   ├── 00_sanity.py                  # diagnostic
│   ├── 00b_dataset_composition.py    # stage 0
│   ├── 01_extract_metadata.py        # stage 1 (GPU)
│   └── 02_divergence.py              # stage 2
├── src/
│   ├── config.py                     # YAML config loader (CLI > config > default)
│   ├── schema.py                     # Record dataclass
│   ├── loader.py                     # per-dataset loaders
│   ├── sampling.py                   # deterministic stratified sampling
│   ├── metadata_extract.py           # prompts + JSON parser
│   ├── divergence.py                 # JSD / TV / Wasserstein math
│   └── plotting.py                   # figure helpers
└── outputs/                          # produced by the pipeline
    ├── composition/                  # stage 0 output
    ├── metadata/                     # stage 1 output (large)
    └── divergence/                   # stage 2 output
```

## Input data

Data paths are resolved by `src/config.py` with priority: CLI flag >
`--config` YAML > built-in default. Defaults target the FullyOpenMeditron
data layout, so running from the parent repo root with no flags just works:

```
<root>/                                                   # FullyOpenMeditron repo root
├── data/
│   ├── moove/full_moove.jsonl                            # MOOVE source
│   ├── synthetic_moove/synthetic_moove_v2_gpt_oss_1try.jsonl  # MOOVE synthetic
│   ├── curated_qa/meditron_4_cleaned.jsonl               # Curated QA source
│   ├── synthetic_qa/meditron_4_synthetic_qa_v3.jsonl     # Curated QA synthetic
│   ├── guidelines/*.jsonl                                # Guidelines source (multiple files)
│   └── guidelines_qa/guidelines_qa_v3_full.jsonl         # Guidelines synthetic
└── data_analysis/...
```

To run on a different layout, either:
1. Edit `configs/default.yaml` once, or
2. Create a new YAML and pass `--config path/to/yours.yaml`, or
3. Override individual paths via CLI flags (`--moove-source`,
   `--meditron-source`, `--guidelines-dir`, etc.)

The `--meditron-source` / `--meditron-synthetic` CLI flags refer to the
Curated QA pair (legacy naming — Curated QA is the Meditron-derived corpus).
The canonical YAML keys use `curated_*`.

Schema details for each loader: see `src/loader.py`. MOOVE source uses a
nested Firestore-style format; the other QA files use sharegpt
(`{conversations: [{from, value}]}`); guideline source files have multiple
body-field schemas (`text`, `clean_text`, `content`, `raw_text`) auto-detected
by the loader.

## Dependencies

In addition to whatever the parent repo requires, this subdirectory needs:

```
vllm==0.8.5.post1               # stage 1 only
torch>=2.6.0,<2.9.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0                     # config loader
```

Add anything missing to the parent's `requirements.txt`.

## Running

All commands run from the parent repo root (`FullyOpenMeditron/`).

### Stage 1 — Metadata extraction (GPU, ~2-3 hours full size on 2× H100)

```bash
python3 data_analysis/scripts/01_extract_metadata.py \
  --root . \
  --out-dir data_analysis/outputs \
  --model Qwen/Qwen3-32B \
  --tp 2 \
  --max-model-len 16384 \
  --max-tokens 1024 \
  --temperature 0.1 \
  --n-per-group -1
```

`--n-per-group -1` runs on the full datasets. Pass a positive integer
(e.g. `100`) for a smoke test. Resume is automatic and ID-keyed.

For a no-GPU dry-run that verifies path resolution only:

```bash
python3 data_analysis/scripts/01_extract_metadata.py \
  --root . --out-dir data_analysis/outputs --n-per-group 5 \
  --model Qwen/Qwen3-32B --dry-run
```

For the full GPU run on a Slurm cluster, use the parent repo's launcher:

```bash
bash new_launch.sh data_analysis/scripts/01_extract_metadata.py \
  --root . \
  --out-dir data_analysis/outputs \
  --model Qwen/Qwen3-32B \
  --n-per-group -1
```

Set `HF_HOME` before running so the ~64GB Qwen3-32B model isn't redownloaded
into ephemeral storage:

```bash
export HF_HOME=/path/to/persistent/hf_cache
```

### Stages 0 and 2 — Analysis (CPU only)

After stage 1 produces `outputs/metadata/`:

```bash
python3 data_analysis/scripts/00b_dataset_composition.py \
  --root . --out-dir data_analysis/outputs/composition

python3 data_analysis/scripts/02_divergence.py \
  --metadata-dir data_analysis/outputs/metadata \
  --out-dir data_analysis/outputs/divergence \
  --n-boot 1000
```

### Sanity check (CPU)

Verify all six input files load correctly without running the full pipeline:

```bash
python3 data_analysis/scripts/00_sanity.py \
  --root . \
  --out-dir data_analysis/outputs \
  --limit 100
```

## Outputs

### `outputs/composition/`
- `composition_by_group_records.pdf`, `composition_by_group_tokens.pdf`
- `composition_syn_vs_src_records.pdf`, `composition_syn_vs_src_tokens.pdf`
- `counts.json` — cached per-group counts (delete or pass `--rebuild-cache` to refresh)

### `outputs/metadata/`
- `{moove,Curated_QA,guidelines}__{source,synthetic}.jsonl` — original record
  dicts plus an `analysis_metadata` field

### `outputs/divergence/`
- `{moove,curated,guidelines}_divergences.csv` — JSD, TV, W1 with 95% CIs
- `proportions.csv` — per-category proportions
- `task_format.csv` — MCQ / paired / document rates and token medians
- `figures/{pair}_{field}.pdf`, `figures/summary_heatmap.pdf`,
  `figures/task_format.pdf`
- `report.md` — auto-generated text summary

## Reproducibility

**Determinism**:
- Sampling: deterministic by `sha256(seed, record_id)`. Pass `--seed N` to vary.
- Bootstrap: deterministic by `sha256(pair, field, metric)`. CIs reproduce exactly.
- vLLM: `temperature=0.1` introduces small nondeterminism; absolute JSD values
  may vary by ~0.001 across reruns. Set `temperature=0.0` for full determinism.



**Hardware**: stage 1 expects 2× NVIDIA H100 (96 GB). For single-GPU runs
use `--tp 1 --gpu-util 0.95` and accept slower throughput. For smaller GPUs
(A100 40GB, etc.), substitute a smaller model via `--model` (e.g.
`Qwen/Qwen2.5-7B-Instruct`).

