#!/bin/bash
#SBATCH --job-name meditron-default-job
#SBATCH --output train_reports/R-%x.%j.err
#SBATCH --error train_reports/R-%x.%j.err
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 256
#SBATCH --partition=normal
#SBATCH --time 8:59:59
# No #SBATCH --environment: the batch step runs on the host and only the
# training srun enters the container. Since 2026-09-21 an srun issued inside a
# container dies loading Slurm's cli_filter_SitePolicies plugin (the image
# lacks libldap_r-2.4.so.2), so a batch step in the container cannot srun.

#usage: bash meditron_train.sh <axolotl_config>

ulimit -c 0 # prevents core dumps

# ---------------------------------------------------------------------------
# Shared argument parsing. Runs in BOTH phases -- the script re-submits itself
# via sbatch, so the worker re-parses the same flags.
#
#   --time HH:MM:SS      walltime override
#   --account NAME       slurm account override
#   --env FILE           container EDF override
#   --nodes N            node-count override (default: the 8 in #SBATCH)
#   --partition NAME     partition override, e.g. debug for a trial run
#   --no-eval            skip chaining the dependent eval job
#   --eval-args "..."    extra args forwarded to the dependent eval job
#   --no-user-site BOOL  export PYTHONNOUSERSITE=1        (default: true)
#   --zero3-init BOOL    ACCELERATE_DEEPSPEED_ZERO3_INIT  (default: false)
#   --requeue            let Slurm requeue the job when it is preempted (the
#                        preemptable partition does so many times a night);
#                        the log is appended to, and a checkpoint the kill
#                        left half-written is removed before resuming. Only
#                        useful with save_steps + auto_resume_from_checkpoints
#   --no-follow          submit and return instead of tailing the log
#
# Folded-in variants (defaults above reproduce meditron_train.sh exactly):
#   gemma : --time 7:59:59 --eval-args "--chat-template none"
#   chuv  : --time 4:29:59 --no-user-site false --zero3-init true
#
# chuv omitted both exports that meditron_train.sh sets. That omission dates
# from the same commit that created the fork (80e6ab0), so it is preserved
# behind flags rather than silently normalised.
# ---------------------------------------------------------------------------
TEMPLATE_CFG=""
WALLTIME=""
ACCOUNT=""
EDF="$PWD/container/new_axolotl.toml"
EVAL_ARGS=""
NO_USER_SITE="true"
ZERO3_INIT="false"
NODES=""
PARTITION=""
NO_EVAL="false"
REQUEUE="false"
NO_FOLLOW="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --time)          WALLTIME="$2"; shift 2 ;;
        --account)       ACCOUNT="$2"; shift 2 ;;
        --env)           EDF="$2"; shift 2 ;;
        --eval-args)     EVAL_ARGS="$2"; shift 2 ;;
        --no-user-site)  NO_USER_SITE="$2"; shift 2 ;;
        --zero3-init)    ZERO3_INIT="$2"; shift 2 ;;
        --nodes)         NODES="$2"; shift 2 ;;
        --partition)     PARTITION="$2"; shift 2 ;;
        --no-eval)       NO_EVAL="true"; shift ;;
        --requeue)       REQUEUE="true"; shift ;;
        --no-follow)     NO_FOLLOW="true"; shift ;;
        -*)              echo "Unknown flag: $1"; exit 1 ;;
        *)               [ -z "$TEMPLATE_CFG" ] && TEMPLATE_CFG="$1"; shift ;;
    esac
done

# =========================================================
# PHASE 1: SUBMISSION LOGIC (Runs on Login Node)
# =========================================================
if [ -z "$SLURM_JOB_ID" ]; then
    if [ -z "$TEMPLATE_CFG" ]; then
        echo "❌ Error: No config file provided."
        exit 1
    fi
    
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
    SBATCH_OPTS=(-J "$JOB_NAME")
    [ -n "$WALLTIME" ] && SBATCH_OPTS+=(--time="$WALLTIME")
    [ -n "$ACCOUNT" ]  && SBATCH_OPTS+=(--account="$ACCOUNT")
    [ -n "$NODES" ]     && SBATCH_OPTS+=(--nodes="$NODES")
    [ -n "$PARTITION" ] && SBATCH_OPTS+=(--partition="$PARTITION")
    [ "$REQUEUE" = "true" ] && SBATCH_OPTS+=(--requeue --open-mode=append)
    # Optional reservation (scripts/reservation.sh), only if it exists now
    # (RESERVATION unset: none).
    source "$(dirname "$SCRIPT_PATH")/scripts/reservation.sh"
    RES_OPT="$(reservation_opts "${ACCOUNT:-}" "${PARTITION:-normal}")"
    [ -n "$RES_OPT" ] && SBATCH_OPTS+=("$RES_OPT") && echo "🎫 $RES_OPT"
    SUBMISSION_OUTPUT=$(TMPDIR=/tmp sbatch "${SBATCH_OPTS[@]}" "$SCRIPT_PATH" "$DEST_CFG" \
        --no-user-site "$NO_USER_SITE" --zero3-init "$ZERO3_INIT" --env "$EDF" \
        $([ "$REQUEUE" = "true" ] && echo --requeue))
    JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')
    echo "🚀 Submitted Job: $JOB_ID"

    # Avoid bash here-doc temp files (can fail if shared /tmp has no space/inodes).
    MODEL_PATH="$(python3 -c 'import sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}; print((cfg.get("output_dir", "") or "").strip())' "$DEST_CFG")"
    if [ -z "$MODEL_PATH" ]; then
    echo "❌ Could not read output_dir from $DEST_CFG"
    exit 1
    fi
    echo "📦 Model path (from output_dir): $MODEL_PATH"

    EVAL_SCRIPT="$PROJECT_ROOT/meditron_eval.sh"
    # no-cot eval is now: "$EVAL_SCRIPT" "$MODEL_PATH" --tasks medbench_no_cot
    # LOGITS_SCRIPT="$PROJECT_ROOT/meditron_eval_logits.sh"

    if [ "$NO_EVAL" = "true" ]; then
        echo "⏭  --no-eval: not chaining an eval job"
        exit 0
    fi

    EVAL_EXTRA=()
    [ -n "$EVAL_ARGS" ] && read -r -a EVAL_EXTRA <<< "$EVAL_ARGS"
    EVAL_OPTS=(--dependency=afterok:"$JOB_ID" -J "eval_${JOB_NAME}")
    # the eval job gets the same reservation as the training job,
    # whatever partition the training job used
    EVAL_RES="$(reservation_opts "${ACCOUNT:-}" normal)"
    [ -n "$EVAL_RES" ] && EVAL_OPTS+=("$EVAL_RES")
    EVAL_SUBMIT_OUT="$(sbatch "${EVAL_OPTS[@]}" "$EVAL_SCRIPT" "$MODEL_PATH" "${EVAL_EXTRA[@]}")"
    EVAL_JOB_ID="$(echo "$EVAL_SUBMIT_OUT" | awk '{print $4}')"
    echo "🧪 Submitted eval job (afterok:$JOB_ID): $EVAL_JOB_ID"

    [ "$NO_FOLLOW" = "true" ] && exit 0

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

FROZEN_CONFIG_PATH="$TEMPLATE_CFG"   # parsed at the top, in both phases
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
JOB_SCRATCH_BASE=${TMPDIR_BASE:-$SCRATCH/axolotl-cache}
mkdir -p $JOB_SCRATCH_BASE
JOB_SCRATCH="$JOB_SCRATCH_BASE/${SLURM_JOB_ID:-nojob}"
LOCAL_TMP="$JOB_SCRATCH/tmp"
LOCAL_HF="$JOB_SCRATCH/hf_cache"
LOCAL_DS="$JOB_SCRATCH/hf_datasets"
LOCAL_TRITON="$JOB_SCRATCH/triton"

# WandB run logs go to WANDB_DIR from .env.
WANDB_LOGS_ROOT="${WANDB_LOGS_ROOT:-$SCRATCH/wandb_logs}"
SHARED_WANDB="${WANDB_DIR:-$WANDB_LOGS_ROOT/$USER}"
# One-off escape hatch, exported at submission: .env's WANDB_DIR would win
# otherwise. For when the shared store refuses writes (over its file
# limit, 2026-09-14); copy the run dir back to $SHARED_WANDB afterwards.
if [ -n "$WANDB_DIR_OVERRIDE" ]; then
    echo "⚠️  WANDB_DIR_OVERRIDE: WandB run files go to $WANDB_DIR_OVERRIDE, not $SHARED_WANDB"
    SHARED_WANDB="$WANDB_DIR_OVERRIDE"
fi

mkdir -p "$LOCAL_TMP" "$LOCAL_HF" "$LOCAL_DS" "$LOCAL_TRITON" "$SHARED_WANDB"

# 4. PRE-FLIGHT CHECK: Create Directories on ALL Nodes
# Added: -A to match the main job parameters
echo "🛠️  Pre-creating scratch directories on all $SLURM_NNODES nodes..."

export TMPDIR=$SCRATCH/tmp
mkdir -p "$TMPDIR"
srun --ntasks-per-node=1 \
     --cpus-per-task=1 \
     --nodes=$SLURM_NNODES \
     ${SLURM_JOB_ACCOUNT:+-A "$SLURM_JOB_ACCOUNT"} \
     bash -c "mkdir -p $LOCAL_TMP $LOCAL_HF $LOCAL_DS $LOCAL_TRITON && ulimit -n 65535"

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
export WANDB_DIR="$SHARED_WANDB"
export AXOLOTL_CONFIG_FILE="$FROZEN_CONFIG_PATH"
export AXOLOTL_DO_NOT_TRACK=1 
export WANDB_MODE="online"

# Fragmentation, not size, is what runs out: OLMo-2-32B on 16 nodes OOMed at
# step 2 (2026-09-21) asking for 270 MiB with ~20 GiB reserved but unused.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Quiet noisy logs
export DEEPSPEED_DISABLE_ASYNC_IO=1
export DS_BUILD_SPARSE_ATTN=0
export SLACK_INSECURE="${SLACK_INSECURE:-1}"

export SLACK_PHASE="train"
export SLACK_REPORTS_DIR="$PROJECT_ROOT/train_reports"
export SLACK_TAG=":robot_face: [Meditron]"

trap 'rc=$?; slack_notify "$rc" "train"; exit "$rc"' EXIT

if [ "$NO_USER_SITE" = "true" ]; then
    export PYTHONNOUSERSITE=1
fi

# A preemption can kill the job while it writes a checkpoint. The Trainer
# writes trainer_state.json after the model and optimizer state, so a
# checkpoint without it is incomplete, and auto_resume would load it.
if [ "$REQUEUE" = "true" ]; then
    echo "🔁 Attempt $((${SLURM_RESTART_COUNT:-0} + 1)) of job $SLURM_JOB_ID"
    # every attempt logs into one WandB run instead of opening a new one
    export WANDB_RUN_ID="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
    export WANDB_RESUME=allow
    OUT_DIR="$(sed -n 's/^output_dir:[[:space:]]*//p' "$FROZEN_CONFIG_PATH" | head -1)"
    for ckpt in "$OUT_DIR"/checkpoint-*; do
        [ -d "$ckpt" ] && [ ! -f "$ckpt/trainer_state.json" ] || continue
        echo "🗑  Removing incomplete checkpoint $ckpt"
        rm -rf "$ckpt"
    done
fi

# 6. Network Setup
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

export ACCELERATE_DEEPSPEED_ZERO3_INIT="$ZERO3_INIT"

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
    --environment="$EDF" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    ${SLURM_JOB_ACCOUNT:+-A "$SLURM_JOB_ACCOUNT"} \
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
