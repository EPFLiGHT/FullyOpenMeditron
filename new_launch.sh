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
#SBATCH --environment ../.edf/inference_3.toml
#SBATCH -A a127

PY_SCRIPT="$1"

if [ -z "$PY_SCRIPT" ]; then
    echo "Error: No python script specified."
    echo "Usage: sbatch $0 <script.py> [args...]"
    exit 1
fi

# PHASE 1: LOGIN NODE
if [ -z "$SLURM_JOB_ID" ]; then
    echo "login node"
    mkdir -p logs

    SUBMIT_OUT=$(sbatch "$0" "$@")
    JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $4}')
    echo "Submitted job $JOB_ID running $PY_SCRIPT with args: $@"

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
        last_lines="$(tail -n 20 "$REPORT_FILE" 2>/dev/null)"
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

cd meditron-4

shift

echo "Executing: python $PY_SCRIPT $@"
python "$PY_SCRIPT" "$@"