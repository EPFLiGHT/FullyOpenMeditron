#!/bin/bash
#SBATCH --job-name=vllm-task
#SBATCH --output=logs/R-%x.%j.err
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=256
#SBATCH --time=7:59:59
#SBATCH --partition=normal
#SBATCH --environment container/inference_4.toml
#
# Usage: bash new_launch.sh [launcher flags] <script.py> [script args...]
#
# Launcher flags (must precede <script.py>; everything after it is passed
# through to the python script untouched):
#   --env FILE           container EDF, overrides the #SBATCH default
#   --time HH:MM:SS      walltime override
#   --account NAME       slurm account override
#   --partition NAME     slurm partition override (e.g. debug)
#                        RESERVATION set: also --reservation, via
#                        scripts/reservation.sh, if it exists
#   --requirements FILE  pip requirements installed on the worker;
#                        "none" skips installation entirely
#   --requeue            let Slurm requeue the job if it is preempted, which
#                        the preemptable partition does routinely; the
#                        resampling scripts resume from their checkpoints
#   --dependency SPEC    sbatch --dependency, e.g. afterok:12345, so a run
#                        starts only once the job it needs has succeeded
#   --array N            submit N independent 1-node jobs (a Slurm array) that
#                        each run the same script; pass --shard auto to a script
#                        that supports it (resampling/*) so every task takes a
#                        different slice of the rows. Tasks start as nodes free
#                        up rather than waiting for N nodes at once. The
#                        launcher does not tail an array; watch the logs
#                        yourself.
#   --no-user-site BOOL  export PYTHONNOUSERSITE=1 (default: true) so the
#                        container's site-packages win over ~/.local, which
#                        holds a torchaudio built against a different CUDA
#                        than the container torch. meditron_train.sh has
#                        always set this; new_launch.sh did not.
#
# Replaces new_launch_apertus.sh, which was this file with a different EDF
# and its pip installs commented out:
#   bash new_launch.sh --env ../.edf/apertus15.toml --requirements none <script.py>

EDF=""
WALLTIME=""
ACCOUNT=""
PARTITION=""
ARRAY=""
DEPENDENCY=""
REQUEUE=""
NO_USER_SITE="${NO_USER_SITE:-true}"
REQUIREMENTS="${REQUIREMENTS:-requirements-inference.txt}"

while [ $# -gt 0 ]; do
    case "$1" in
        --env)          EDF="$2"; shift 2 ;;
        --time)         WALLTIME="$2"; shift 2 ;;
        --account)      ACCOUNT="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --no-user-site) NO_USER_SITE="$2"; shift 2 ;;
        --requirements) REQUIREMENTS="$2"; shift 2 ;;
        --array)        ARRAY="$2"; shift 2 ;;
        --dependency)   DEPENDENCY="$2"; shift 2 ;;
        --requeue)      REQUEUE="yes"; shift ;;
        --)             shift; break ;;
        *)              break ;;   # first non-flag token is <script.py>
    esac
done

PY_SCRIPT="$1"

if [ -z "$PY_SCRIPT" ]; then
    echo "Error: No python script specified."
    echo "Usage: bash $0 [--env FILE] [--time HH:MM:SS] [--account NAME]"
    echo "              [--partition NAME] [--requirements FILE|none] <script.py> [args...]"
    exit 1
fi

# PHASE 1: LOGIN NODE
if [ -z "$SLURM_JOB_ID" ]; then
    echo "login node"
    mkdir -p logs

    # sbatch CLI options override the #SBATCH directives above, so a variant
    # walltime/image/account is a flag rather than a forked copy of this file.
    SBATCH_OPTS=()
    [ -n "$EDF" ]      && SBATCH_OPTS+=(--environment="$EDF")
    [ -n "$WALLTIME" ] && SBATCH_OPTS+=(--time="$WALLTIME")
    [ -n "$ACCOUNT" ]   && SBATCH_OPTS+=(--account="$ACCOUNT")
    [ -n "$PARTITION" ] && SBATCH_OPTS+=(--partition="$PARTITION")
    [ -n "$DEPENDENCY" ] && SBATCH_OPTS+=(--dependency="$DEPENDENCY")
    [ -n "$REQUEUE" ] && SBATCH_OPTS+=(--requeue)
    if [ -n "$ARRAY" ]; then
        if ! [ "$ARRAY" -ge 1 ] 2>/dev/null; then
            echo "Error: --array takes a job count >= 1"; exit 1
        fi
        SBATCH_OPTS+=(--array="0-$((ARRAY - 1))")
    fi

    # Optional reservation: only when RESERVATION is set and the reservation exists.
    source "$(dirname "$0")/scripts/reservation.sh"
    RES_OPT="$(reservation_opts "${ACCOUNT:-}" "${PARTITION:-normal}")"
    [ -n "$RES_OPT" ] && SBATCH_OPTS+=("$RES_OPT")

    # Slurm 25.05 refuses --environment given both on the command line and in
    # #SBATCH, so --env submits a copy of this file without that header line.
    # The header stays so a direct `sbatch new_launch.sh` keeps its default.
    SUBMIT_SCRIPT="$0"
    if [ -n "$EDF" ]; then
        SUBMIT_SCRIPT="$(mktemp)"
        grep -v '^#SBATCH --environment' "$0" > "$SUBMIT_SCRIPT"
    fi

    # Re-pass launcher flags so PHASE 2 sees the same REQUIREMENTS value.
    SUBMIT_OUT=$(sbatch "${SBATCH_OPTS[@]}" "$SUBMIT_SCRIPT" --requirements "$REQUIREMENTS" \
        --no-user-site "$NO_USER_SITE" "$@")
    [ "$SUBMIT_SCRIPT" != "$0" ] && rm -f "$SUBMIT_SCRIPT"
    JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $4}')
    echo "Submitted job $JOB_ID running $PY_SCRIPT with args: $@"

    if [ -n "$ARRAY" ]; then
        # every array task writes its own logs/R-vllm-task.<taskjobid>.err, so
        # there is no single file to follow
        echo "Array of $ARRAY tasks; watch with: squeue -j $JOB_ID; tail -f logs/R-vllm-task.*.err"
        exit 0
    fi

    LOG_FILE="logs/R-vllm-task.${JOB_ID}.err"
    while [ ! -f "$LOG_FILE" ]; do sleep 1; done
    echo "Log file: $LOG_FILE"
    tail -n 0 -F "$LOG_FILE" &
    TAIL_PID=$!

    while squeue -h -j "$JOB_ID" | grep -q .; do sleep 5; done
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    exit 0
fi

# PHASE 2: COMPUTE NODE
echo "compute node"

export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -o allexport
    source "$PROJECT_ROOT/.env"
    set +o allexport
fi

# Model weights must not live in a job-scoped cache. .env sets
# HF_HOME=$CACHE_ROOT/hf where CACHE_ROOT ends in ${SLURM_JOB_ID:-nojob}, so
# every job would start with an empty hub cache and re-download every model.
# meditron_eval.sh already overrides this the same way.
if [ "$NO_USER_SITE" = "true" ]; then
    # ~/.local/lib/python3.12/site-packages shadows the container's packages and
    # carries a torchaudio compiled against a different CUDA than the container
    # torch, which breaks `import transformers` (loss_rnnt imports torchaudio).
    export PYTHONNOUSERSITE=1
fi

export HF_HOME="${HF_HOME_OVERRIDE:-$USER_STORAGE/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/scripts/slack_helpers.sh"

START_TS="$(date +%s)"
START_HUMAN="$(date -Is)"
RUN_NAME="vllm-task"
SLACK_JOB_ID="${SLURM_JOB_ID:-?}"
FAILED_CMD=""
export SLACK_INSECURE="${SLACK_INSECURE:-1}"

REPORT_FILE="$PROJECT_ROOT/logs/R-${RUN_NAME}.${SLACK_JOB_ID}.err"

# Custom notifier: builds a message with last 10 lines of the log
notify_slack() {
    local rc="$1"
    local end_ts end_human elapsed status text payload last_lines

    end_ts="$(date +%s)"
    end_human="$(date -Is)"
    elapsed="$((end_ts - START_TS))"

    if [ "$rc" -eq 0 ]; then
        status="COMPLETED"
    else
        status="FAILED"
    fi

    text=":robot_face: [vLLM] [$status] ${RUN_NAME} (job ${SLACK_JOB_ID}) on $(hostname)
Script:  ${PY_SCRIPT}
Start:   ${START_HUMAN}
End:     ${end_human}
Elapsed: $(format_duration "$elapsed")
Exit code: ${rc}"

    if [ "$rc" -ne 0 ] && [ -n "${FAILED_CMD}" ]; then
        text="${text}
Failed at: ${FAILED_CMD}"
    fi

    if [ -f "$REPORT_FILE" ]; then
        last_lines="$(tail -n 2 "$REPORT_FILE" 2>/dev/null)"
        if [ -n "$last_lines" ]; then
            text="${text}

*Last lines:*
\`\`\`
${last_lines}
\`\`\`"
        fi
    fi

    echo "== SLACK MESSAGE =="
    printf '%s\n' "$text"

    if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then
        return 0
    fi

    payload="$(_slack_build_payload "$text")"

    local curl_ssl_flag=()
    if [ "${SLACK_INSECURE:-0}" = "1" ]; then
        curl_ssl_flag+=(--insecure)
    fi

    { set +x; } 2>/dev/null
    curl -sS "${curl_ssl_flag[@]}" -X POST -H 'Content-type: application/json' \
        --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null || true
    { set -x; } 2>/dev/null
}

trap 'FAILED_CMD=$BASH_COMMAND' ERR
trap 'rc=$?; notify_slack "$rc"; exit "$rc"' EXIT
set -eo pipefail

if [ "$REQUIREMENTS" = "none" ]; then
    echo "Skipping dependency install (--requirements none)"
elif [ -f "$PROJECT_ROOT/$REQUIREMENTS" ]; then
    echo "Installing $REQUIREMENTS"
    pip install -r "$PROJECT_ROOT/$REQUIREMENTS"
else
    echo "WARNING: requirements file '$REQUIREMENTS' not found; skipping install"
fi

shift

echo "Executing: python $PY_SCRIPT $@"
python "$PY_SCRIPT" "$@"
