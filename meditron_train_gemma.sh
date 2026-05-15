#!/bin/bash
#SBATCH --job-name meditron-default-job
#SBATCH --output train_reports/R-%x.%j.err
#SBATCH --error train_reports/R-%x.%j.err
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 256
#SBATCH --partition=normal
#SBATCH --time 7:59:59
#SBATCH --environment ../.edf/new_axolotl.toml
#SBATCH -A a127

#usage: bash meditron_train.sh <axolotl_config>

ulimit -c 0 # prevents core dumps

# =========================================================
# PHASE 1: SUBMISSION LOGIC (Runs on Login Node)
# =========================================================
if [ -z "$SLURM_JOB_ID" ]; then
    if [ -z "$1" ]; then
        echo "❌ Error: No config file provided."
        exit 1
    fi
    TEMPLATE_CFG="$1"
    
    # Load Environment
    set -o allexport
    if [ -f "$PROJECT_ROOT/.env" ]; then source "$PROJECT_ROOT/.env"; elif [ -f .env ]; then source .env; fi
    set +o allexport

    SRC_CFG="$PROJECT_ROOT/$TEMPLATE_CFG"
    JOB_NAME=$(basename "$TEMPLATE_CFG" .yaml)
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    
    GEN_CONFIG_DIR="$PROJECT_ROOT/axolotl_config/generated"
    mkdir -p "$GEN_CONFIG_DIR"
    DEST_CFG="$GEN_CONFIG_DIR/${JOB_NAME}-${TIMESTAMP}.yaml"
    
    echo "📝 Using template: $SRC_CFG"
    envsubst < "$SRC_CFG" > "$DEST_CFG"

    SCRIPT_PATH="$0"
    # Ensure Slurm doesn't try to create a non-existent TMPDIR before scratch exists.
    SUBMISSION_OUTPUT=$(TMPDIR=/tmp sbatch -J "$JOB_NAME" "$SCRIPT_PATH" "$DEST_CFG")
    JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')
    echo "🚀 Submitted Job: $JOB_ID"

    # Avoid bash here-doc temp files (can fail if shared /tmp has no space/inodes).
    MODEL_PATH="$(python3 -c 'import sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}; print((cfg.get("output_dir", "") or "").strip())' "$DEST_CFG")"
    if [ -z "$MODEL_PATH" ]; then
    echo "❌ Could not read output_dir from $DEST_CFG"
    exit 1
    fi
    echo "📦 Model path (from output_dir): $MODEL_PATH"

    EVAL_SCRIPT="$PROJECT_ROOT/meditron_eval_fix.sh"
    #EVAL_SCRIPT_NO_COT="$PROJECT_ROOT/meditron_eval_no_cot.sh"
    # LOGITS_SCRIPT="$PROJECT_ROOT/meditron_eval_logits.sh"

    EVAL_SUBMIT_OUT="$(sbatch --dependency=afterok:"$JOB_ID" -J "eval_${JOB_NAME}" "$EVAL_SCRIPT" "$MODEL_PATH")"
    EVAL_JOB_ID="$(echo "$EVAL_SUBMIT_OUT" | awk '{print $4}')"
    echo "🧪 Submitted eval job (afterok:$JOB_ID): $EVAL_JOB_ID"

    #EVAL_SUBMIT_OUT_NO_COT="$(sbatch --dependency=afterok:"$JOB_ID" -J "${JOB_NAME}-eval-no-cot" "$EVAL_SCRIPT_NO_COT" "$MODEL_PATH")"
    #EVAL_NO_COT_JOB_ID="$(echo "$EVAL_SUBMIT_OUT_NO_COT" | awk '{print $4}')"
    #echo "🧪 Submitted eval no cot job (afterok:$JOB_ID): $EVAL_NO_COT_JOB_ID"

    # LOGITS_SUBMIT_OUT="$(sbatch --dependency=afterok:"$JOB_ID" -J "${JOB_NAME}-logits" "$LOGITS_SCRIPT" "$MODEL_PATH")"
    # LOGITS_JOB_ID="$(echo "$LOGITS_SUBMIT_OUT" | awk '{print $4}')"
    # echo "📊 Submitted logits eval job (afterok:$JOB_ID): $LOGITS_JOB_ID"

    LOG_FILE="$PROJECT_ROOT/train_reports/R-${JOB_NAME}.${JOB_ID}.err"
    echo "waiting for log $LOG_FILE"
    while [ ! -f "$LOG_FILE" ]; do sleep 1; done
    tail -n 0 -F "$LOG_FILE" &
    TAIL_PID=$!
    while squeue -j "$JOB_ID" >/dev/null 2>&1; do sleep 5; done
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    exit 0
fi

# =========================================================
# PHASE 2: WORKER LOGIC (Runs on Compute Nodes)
# =========================================================

FROZEN_CONFIG_PATH="$1"
if [ ! -f "$FROZEN_CONFIG_PATH" ]; then
    echo "CRITICAL ERROR: Config not found: $FROZEN_CONFIG_PATH"
    exit 1
fi

# --- ADDED: Initialize Vars for Slack ---
export START_TS=$(date +%s)
export START_HUMAN=$(date -Is)
# Extract just the filename without extension for the run name
export RUN_NAME=$(basename "$FROZEN_CONFIG_PATH" .yaml)
# Ensure Node count is available (Slurm usually sets this, but we explicitly export it)
export NODE_COUNT=${SLURM_NNODES:-1}
# ----------------------------------------

export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}

# 1. Load Environment
set -o allexport
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
elif [ -f .env ]; then
    source .env
fi
set +o allexport

# 2. Fix Slack Helper Import
# Using absolute path ensures workers find the script despite spooling
if ! declare -F slack_notify >/dev/null 2>&1; then
    if [ -f "$PROJECT_ROOT/scripts/slack_helpers.sh" ]; then
        source "$PROJECT_ROOT/scripts/slack_helpers.sh"
    fi
fi

# 3. Define Scratch Paths (Do NOT export TMPDIR yet!)
JOB_SCRATCH_BASE=${TMPDIR_BASE:-/iopsstor/scratch/cscs/$USER/axolotl-cache}
mkdir -p $JOB_SCRATCH_BASE
JOB_SCRATCH="$JOB_SCRATCH_BASE/${SLURM_JOB_ID:-nojob}"
LOCAL_TMP="$JOB_SCRATCH/tmp"
LOCAL_HF="$JOB_SCRATCH/hf_cache"
LOCAL_DS="$JOB_SCRATCH/hf_datasets"
LOCAL_TRITON="$JOB_SCRATCH/triton"
LOCAL_WANDB="$JOB_SCRATCH/wandb"
mkdir -p "$LOCAL_TMP" "$LOCAL_HF" "$LOCAL_DS" "$LOCAL_TRITON" "$LOCAL_WANDB"

# 4. PRE-FLIGHT CHECK: Create Directories on ALL Nodes
# Added: -A and --reservation to match the main job parameters
echo "🛠️  Pre-creating scratch directories on all $SLURM_NNODES nodes..."

export TMPDIR=/iopsstor/scratch/cscs/$USER/tmp
mkdir -p "$TMPDIR"
srun --ntasks-per-node=1 \
     --cpus-per-task=1 \
     --nodes=$SLURM_NNODES \
     -A a127 \
     bash -c "mkdir -p $LOCAL_TMP $LOCAL_HF $LOCAL_DS $LOCAL_TRITON $LOCAL_WANDB/wandb && ulimit -n 65535"

if [ $? -ne 0 ]; then
    echo "❌ CRITICAL: Failed to create scratch directories on remote nodes."
    exit 1
fi
echo "✅ Scratch directories ready."

# 5. Export Environment
export TMPDIR="$LOCAL_TMP"
export HF_HOME="$LOCAL_HF"
export TRANSFORMERS_CACHE="$LOCAL_HF/transformers"
export HF_DATASETS_CACHE="$LOCAL_DS"
export TRITON_CACHE_DIR="$LOCAL_TRITON"
export WANDB_DIR="$LOCAL_WANDB"
export AXOLOTL_CONFIG_FILE="$FROZEN_CONFIG_PATH"
export AXOLOTL_DO_NOT_TRACK=1 
export WANDB_MODE="online"

# Quiet noisy logs
export DEEPSPEED_DISABLE_ASYNC_IO=1
export DS_BUILD_SPARSE_ATTN=0
export SLACK_INSECURE="${SLACK_INSECURE:-1}"

export SLACK_PHASE="train"
export SLACK_REPORTS_DIR="$PROJECT_ROOT/train_reports"
export SLACK_TAG=":robot_face: [Meditron]"

trap 'rc=$?; slack_notify "$rc" "train"; exit "$rc"' EXIT

export PYTHONNOUSERSITE=1

# 6. Network Setup
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

export ACCELERATE_DEEPSPEED_ZERO3_INIT=false

LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 0"

# 7. Launch Training
FULL_CMD="$LAUNCHER -m axolotl.cli.train $AXOLOTL_CONFIG_FILE"

echo "🚀 Launching Axolotl on $SLURM_NNODES nodes..."
echo "Command: $FULL_CMD"

LOG_FILE="$PROJECT_ROOT/train_reports/R-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    -A a127 \
    bash -c "$FULL_CMD"
TRAIN_RC=$?

if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
fi

echo "END TIME: $(date)"

if [ "${TRAIN_RC:-1}" -ne 0 ]; then
    echo "❌ Training failed with exit code: $TRAIN_RC"
    exit "$TRAIN_RC"
fi
