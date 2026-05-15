#!/bin/bash
#SBATCH --job-name meditron-eval-vllm
#SBATCH --output eval_reports/R-%x.%j.err
#SBATCH --error eval_reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 256
#SBATCH --partition=normal
#SBATCH --time=1:59:59
#SBATCH --environment ../.edf/inference_3.toml
#SBATCH -A a127

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/scripts/slack_helpers.sh"

# PHASE 1: SUBMISSION LOGIC /LOGIN NODE

if [ -z "$SLURM_JOB_ID" ]; then
    RUN_NAME="meditron-eval-vllm"
    MODEL_PATH="$1"
    DEBUG_FLAG=0
    MODEL_PARALLELISM=0
    if [ "${2:-}" = "--debug" ]; then
        DEBUG_FLAG=1
    elif [ "${2:-}" = "--model_parallelism" ]; then
        MODEL_PARALLELISM=1
    fi
    if [ "${3:-}" = "--debug" ]; then
        DEBUG_FLAG=1
    elif [ "${3:-}" = "--model_parallelism" ]; then
        MODEL_PARALLELISM=1
    fi
    SCRIPT_PATH="$0"

    # Load env (PROJECT_ROOT, USER_STORAGE, etc.)
    if [ -f .env ]; then
        set -o allexport
        source .env
        set +o allexport
    fi

    echo "This script is self-submitting..."
    echo "🏷️  Run Name:  $RUN_NAME"
    echo "📁  Model Path: $MODEL_PATH"

    # Avoid exporting a job-scoped TMPDIR before SLURM_JOB_ID exists.
    export TMPDIR="${SLURM_TMPDIR:-/tmp}"

    if [ -z "$MODEL_PATH" ]; then
        echo "Usage: $0 /path/to/model [--debug] [--model_parallelism]"
        exit 1
    fi

    SBATCH_ARGS=("$RUN_NAME" "$MODEL_PATH")
    if [ "$DEBUG_FLAG" -eq 1 ]; then
        SBATCH_ARGS+=("--debug")
    fi
    if [ "$MODEL_PARALLELISM" -eq 1 ]; then
        SBATCH_ARGS+=("--model_parallelism")
    fi
    SUBMISSION_OUTPUT=$(sbatch -J "$RUN_NAME" "$SCRIPT_PATH" "${SBATCH_ARGS[@]}")
    JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')

    echo "🚀 Submitted Job: $JOB_ID"

    LOG_FILE="$PROJECT_ROOT/eval_reports/R-${RUN_NAME}.${JOB_ID}.err"
    echo "Waiting for log file: $LOG_FILE"

    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done

    echo "✅ Log found! Tailing (Ctrl+C to stop watching, job will continue)..."
    echo "-------------------------------------------------------------------"
    tail -n 0 -F "$LOG_FILE" &
    TAIL_PID=$!

    while squeue -j "$JOB_ID" >/dev/null 2>&1; do
        sleep 5
    done

    echo "Job $JOB_ID finished; stopping log tail."
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true

    exit 0
fi



# PHASE 2: WORKER LOGIC / COMPUTE NODE

RUN_NAME="$1"
MODEL_PATH="$2"
DEBUG_FLAG=0
MODEL_PARALLELISM=0
if [ "$MODEL_PATH" = "--debug" ]; then
    DEBUG_FLAG=1
    MODEL_PATH=""
elif [ "$MODEL_PATH" = "--model_parallelism" ]; then
    MODEL_PARALLELISM=1
    MODEL_PATH=""
elif [ "${3:-}" = "--debug" ]; then
    DEBUG_FLAG=1
elif [ "${3:-}" = "--model_parallelism" ]; then
    MODEL_PARALLELISM=1
fi

# Back-compat: allow a single positional arg to mean MODEL_PATH.
if [ -z "$MODEL_PATH" ] && [ -n "$RUN_NAME" ]; then
    MODEL_PATH="$RUN_NAME"
    RUN_NAME=""
fi

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="eval-gen"
fi
if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is empty; pass it as the first argument."
    exit 1
fi
if [ ! -e "$MODEL_PATH" ] && [ -n "$STORAGE_ROOT" ] && [ -e "$STORAGE_ROOT$MODEL_PATH" ]; then
    MODEL_PATH="$STORAGE_ROOT$MODEL_PATH"
    echo "Resolved MODEL_PATH to: $MODEL_PATH"
fi

if [ -d "$MODEL_PATH" ]; then
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "MODEL_PATH is missing config.json: $MODEL_PATH"
        exit 1
    fi
    if [ ! -f "$MODEL_PATH/tokenizer_config.json" ] && [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
        echo "MODEL_PATH is missing tokenizer files: $MODEL_PATH"
        exit 1
    fi
    if ! compgen -G "$MODEL_PATH/model-*.safetensors" > /dev/null \
       && [ ! -f "$MODEL_PATH/model.safetensors" ] \
       && [ ! -f "$MODEL_PATH/pytorch_model.bin" ] \
       && ! compgen -G "$MODEL_PATH/checkpoint-*/model-*.safetensors" > /dev/null \
       && ! compgen -G "$MODEL_PATH/checkpoint-*/pytorch_model*.bin" > /dev/null; then
        echo "MODEL_PATH has no model weights yet: $MODEL_PATH"
        exit 1
    fi
fi

export SLACK_MODEL_PATH="$MODEL_PATH"
export SLACK_MODEL_TAG="$SAFE_MODEL_TAG"

# 1. Project root & env
export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$PROJECT_ROOT"

echo "Project Root detected as: $PROJECT_ROOT"

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# Ensure TMPDIR exists to avoid Slurm temp directory errors.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

# Default to insecure curl for Slack if the node lacks CA bundle; can override by exporting SLACK_INSECURE=0
export SLACK_INSECURE="${SLACK_INSECURE:-1}"

# Slack runtime bookkeeping (worker side)
START_TS="$(date +%s)"
START_HUMAN="$(date -Is)"
SLACK_JOB_ID="${SLURM_JOB_ID:-?}"
FAILED_CMD=""
export SLACK_PHASE="eval"
export SLACK_REPORTS_DIR="$PROJECT_ROOT/eval_reports"
export SLACK_TAG=":robot_face: [Meditron]"
trap 'FAILED_CMD=$BASH_COMMAND' ERR
trap 'rc=$?; slack_notify "$rc" "eval"; exit "$rc"' EXIT
set -eo pipefail

cd "$PROJECT_ROOT/../lm-evaluation-harness"
pip install -e .[ifeval]
#pip install --upgrade transformers
#pip install --upgrade --no-deps "datasets>=2.19.0,<3.0.0"

cd "$PROJECT_ROOT/../lm-evaluation-harness/lm_eval/tasks"

export WORLD_SIZE=$SLURM_NNODES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=6300

export HF_HOME=$USER_STORAGE/hf
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRITON_CACHE_DIR=$USER_STORAGE/triton
export HF_DATASETS_CACHE=$HF_HOME/datasets
unset TRANSFORMERS_CACHE
# Required on some clusters/PyTorch builds: avoid CUDA init in forked subprocesses.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export LM_EVAL_INCLUDE_PATH="/users/$USER/lm-evaluation-harness/lm_eval/tasks"

echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MODEL_PATH=$MODEL_PATH"

echo "START TIME: $(date)"
set -x

SAFE_MODEL_TAG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_.-' '_' | sed 's/^_//;s/_$//')"
RUN_TAG="${RUN_NAME:-eval}"
JOB_TAG="${SLURM_JOB_ID:-nojob}"
OUTPUT_DIR="$PROJECT_ROOT/eval_results/${RUN_TAG}_${SAFE_MODEL_TAG}_${JOB_TAG}"
mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=()
if [ "$DEBUG_FLAG" -eq 1 ]; then
    LIMIT_ARGS=(--limit 100)
fi

TOKENIZER_ARGS=""
if [ -f "$MODEL_PATH/config.json" ] && rg -q '"model_type"\s*:\s*"apertus"' "$MODEL_PATH/config.json"; then
    APERTUS_TOKENIZER_PATH="${APERTUS_TOKENIZER_PATH:-$STORAGE_ROOT/apertus/huggingface/swiss-ai/Apertus-8B-Instruct-2509}"
    if [ -d "$APERTUS_TOKENIZER_PATH" ]; then
        TOKENIZER_ARGS=",tokenizer=$APERTUS_TOKENIZER_PATH"
        echo "Using Apertus tokenizer fallback: $APERTUS_TOKENIZER_PATH"
    else
        echo "Apertus model detected but fallback tokenizer path not found: $APERTUS_TOKENIZER_PATH"
    fi
fi
NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
DP_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=$TP_SIZE,data_parallel_size=$DP_SIZE,gpu_memory_utilization=$GPU_MEM_UTIL,max_model_len=$MAX_MODEL_LEN,max_gen_toks=4096,seed=42$TOKENIZER_ARGS"

unset SSL_CERT_FILE

python3 -m lm_eval \
  --model vllm \
  --model_args "$MODEL_ARGS" \
  --tasks medbench \
  --batch_size auto \
  --verbosity "DEBUG" \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  --include_path "$LM_EVAL_INCLUDE_PATH" \
  --gen_kwargs temperature=0.0 \
  --seed 42 \
  "${LIMIT_ARGS[@]}"
  
echo "END TIME: $(date)"

#pubmedqa,medmcqa,medqa_4options,pubmedqa_g,medmcqa_g,medqa_g,medxpertqa,medxpertqa_g,mmlu_flan_cot_zeroshot, medbench
