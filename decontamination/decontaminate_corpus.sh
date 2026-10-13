#!/bin/bash
#SBATCH --job-name=decontam
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=container/inference_4.toml
#SBATCH --output=decontamination/logs/decontam_%j.out
#SBATCH --error=decontamination/logs/decontam_%j.out
#
# Decontaminate training JSONLs against the MeditronFO evaluation set.
# Submit from the repo root:
#
#   sbatch decontamination/decontaminate_corpus.sh <out_dir> <train.jsonl>...
#
# 1. build_eval_benchmarks.py  the benchmark prompts -> $BENCH (built once;
#                              delete $BENCH and ${BENCH}_ngram_cache to rebuild)
# 2. decontamination.py        per JSONL: 8-gram overlap between the first user
#                              turn and each benchmark prompt, confirmed at a
#                              diff ratio of 0.5 -> <out_dir>/reports/<name>/
# 3. apply_reports.py          per JSONL: the original rows minus the reported
#                              ones -> <out_dir>/<name>_decontam.jsonl
#
# Replaces run_decontamination.sh, which points at another user's checkout,
# container and account.

set -euo pipefail
unset SSL_CERT_FILE
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
if [ -f .env ]; then set -o allexport; source .env; set +o allexport; fi
export HF_HOME=${USER_STORAGE:-$SCRATCH}/hf

OUT_DIR=${1:?usage: sbatch decontamination/decontaminate_corpus.sh <out_dir> <train.jsonl>...}
shift
BENCH=${BENCH:-decontamination/data/decontam_prompts/meditronfo_eval}
TOKENIZER=${TOKENIZER:-$STORAGE_ROOT/apertus/huggingface/swiss-ai/Apertus-70B-Instruct-2509}
# decontamination.py also saves an Arrow copy of each filtered set; nothing reads it
WORK=${USER_STORAGE:-$SCRATCH}/decontam/${SLURM_JOB_ID:-local}
mkdir -p "$OUT_DIR" "$WORK"

# BENCH_ONLY="moove_unanimous576 ..." builds only those benchmarks into $BENCH
[ -d "$BENCH" ] || python3 decontamination/build_eval_benchmarks.py --out "$BENCH" \
    ${BENCH_ONLY:+--only $BENCH_ONLY}

for jsonl in "$@"; do
    name=$(basename "$jsonl" .jsonl)
    echo "=== $name"
    # the n-gram cache is keyed on benchmark *name*, not content: keep it with $BENCH
    python3 decontamination/decontamination.py "$jsonl" \
        --output "$WORK/$name" \
        --decontamination_prompts "$BENCH" \
        --train_format sharegpt \
        --tokenizer_name "$TOKENIZER" \
        --report_path "$OUT_DIR/reports/$name" \
        --cache_dir "${BENCH}_ngram_cache" \
        --ngram_length 8 \
        --diff_threshold 0.5 \
        --num_proc 128 \
        --overwrite \
        --show_contaminated
    python3 decontamination/apply_reports.py --input "$jsonl" \
        --reports "$OUT_DIR/reports/$name" \
        --output "$OUT_DIR/${name}_decontam.jsonl"
done
rm -rf "$WORK"
