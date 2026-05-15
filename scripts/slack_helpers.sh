#!/usr/bin/env bash
#scripts/slack_helpers.sh
# Shared Slack helpers for training/eval scripts.
# Expects the caller to define START_TS, START_HUMAN, RUN_NAME, SLURM_JOB_ID/SLACK_JOB_ID, and optionally FAILED_CMD.

if [ -n "${SLACK_HELPERS_LOADED:-}" ]; then
  return 0
fi
SLACK_HELPERS_LOADED=1

: "${SLACK_INSECURE:=1}"

format_duration() {
    local s="$1"
    printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

_slack_find_errors_summary() {
    local report_file="$1"
    local reports_dir find_errors_path output summary

    if [ -n "${FIND_ERRORS_SCRIPT:-}" ] && [ -x "$FIND_ERRORS_SCRIPT" ]; then
        find_errors_path="$FIND_ERRORS_SCRIPT"
    elif [ -n "${PROJECT_ROOT:-}" ] && [ -x "$PROJECT_ROOT/find_errors.sh" ]; then
        find_errors_path="$PROJECT_ROOT/find_errors.sh"
    else
        find_errors_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/find_errors.sh"
        if [ ! -x "$find_errors_path" ]; then
            return 0
        fi
    fi

    reports_dir="$(dirname "$report_file")"
    if [ ! -d "$reports_dir" ]; then
        return 0
    fi

    output="$("$find_errors_path" "$reports_dir" 2>/dev/null || true)"
    summary="$(printf '%s\n' "$output" | awk -v target="$report_file" '
        $0 == "== " target " ==" {found=1; next}
        found && $0 == "" {exit}
        found {print; exit}
    ')"

    if [ -n "$summary" ]; then
        printf '%s' "$summary"
    fi
}

_slack_extract_eval_results() {
    local report="$1"
    local job_id="$2"
    local results_dir="${PROJECT_ROOT}/eval_results"
    local output=""

    # 1. Extract the results table from the SLURM log
    local table
    table="$(awk '
        /^\|[[:space:]]*Tasks[[:space:]]*\|/ {in_table=1; print; next}
        in_table && /^\|/ {print; next}
        in_table {exit}
    ' "$report" 2>/dev/null)"

    if [ -n "$table" ]; then
        output="${table}\n"
    fi

    # 2. Find the specific eval_results folder for this job ID to count invalids
    if [ -n "$job_id" ] && [ -d "$results_dir" ]; then
        local eval_dir
        eval_dir="$(find "$results_dir" -type d -name "*_${job_id}" | sort | head -n 1)"
        
        if [ -n "$eval_dir" ]; then
            local invalid_total=0 line_total=0
            
            # Count invalids (prefer ripgrep if available, fallback to grep)
            if command -v rg >/dev/null 2>&1; then
                invalid_total=$(find "$eval_dir" -type f -name "*.jsonl" -print0 | xargs -0 -r rg -o -i "\\[invalid\\]" 2>/dev/null | wc -l | tr -d ' ')
            else
                invalid_total=$(find "$eval_dir" -type f -name "*.jsonl" -print0 | xargs -0 -r grep -o -i "\\[invalid\\]" 2>/dev/null | wc -l | tr -d ' ')
            fi
            
            # Count total lines
            line_total=$(find "$eval_dir" -type f -name "*.jsonl" -print0 | xargs -0 -r cat 2>/dev/null | wc -l | tr -d ' ')
            
            if [ "$line_total" -gt 0 ]; then
                local percent
                percent=$(awk -v i="$invalid_total" -v t="$line_total" 'BEGIN {printf "%.2f", (i / t) * 100}')
                output="${output}\nInvalids: ${invalid_total}/${line_total} (${percent}%)"
            fi
        fi
    fi

    # Echo back to the caller
    echo -e "$output"
}

_slack_extract_excel_row() {
    local report="$1"
    python3 - "$report" <<'PY'
import re, sys

with open(sys.argv[1], "r", errors="replace") as f:
    text = f.read()

row_re = re.compile(
    r"^\|\s*-?\s*([A-Za-z0-9_]+)\s*\|[^|]*\|[^|]*\|\s*\d+\|\s*\S+\s*\|[^|]*\|\s*([0-9.]+)\s*\|",
    re.M,
)
scores = {}
for name, val in row_re.findall(text):
    scores.setdefault(name, val)

def g(k): return scores.get(k, "")

medmcqa  = g("medmcqa_g")
medqa    = g("medqa_g")
pubmedqa = g("pubmedqa_g")
medxpert = g("medxpertqa_g")
mmlu_pro = g("mmlu_pro")

m = re.search(r"\|\s*prompt_level_strict_acc\s*\|[^|]*\|\s*([0-9.]+)\s*\|", text)
ifeval = m.group(1) if m else ""

def avg(vals):
    nums = [float(v) for v in vals if v]
    return f"{sum(nums)/len(nums):.4f}" if nums else ""

med_avg = avg([medmcqa, medqa, pubmedqa])
gen_avg = avg([mmlu_pro, ifeval])

# Columns: medmcqa | medqa | pubmedqa | avg | weighted_avg | medxpertqa | gain | mmlu_pro | ifeval | avg | gain
row = [medmcqa, medqa, pubmedqa, med_avg, "", medxpert, "", mmlu_pro, ifeval, gen_avg, ""]
print("\t".join(row))
PY
}

_slack_extract_training_loss() {
    # 1. Determine config path
    local config_file="${AXOLOTL_CONFIG_FILE:-${FROZEN_CONFIG_PATH:-}}"
    if [ -z "$config_file" ] || [ ! -f "$config_file" ]; then 
        echo "[Debug Slack] Config file not found: '$config_file'" >&2
        return
    fi
    
    # 2. Parse output_dir safely using native AWK (no Python yaml dependency)
    local model_dir
    model_dir=$(awk '/^output_dir:/ {print $2}' "$config_file" | tr -d '"' | tr -d "'")
    
    if [ -z "$model_dir" ] || [ ! -d "$model_dir" ]; then 
        echo "[Debug Slack] Model directory not found or invalid: '$model_dir'" >&2
        return
    fi
    
    # 3. Find trainer_state.json (Root or latest checkpoint folder)
    local state_file="$model_dir/trainer_state.json"
    if [ ! -f "$state_file" ]; then
        # Use find instead of ls for safer recursive searching
        state_file=$(find "$model_dir" -maxdepth 2 -name "trainer_state.json" | grep "checkpoint-" | sort -V | tail -n 1)
    fi
    
    if [ -z "$state_file" ] || [ ! -f "$state_file" ]; then 
        echo "[Debug Slack] trainer_state.json not found in $model_dir" >&2
        return
    fi

    echo "[Debug Slack] Successfully found state file: $state_file" >&2

    # 4. Generate pure ASCII plot via embedded Python using a Heredoc
    python3 - "$state_file" <<'EOF'
import json, sys

def draw(steps, losses, width=45, height=12):
    if not steps: return "No step data found in logs."
    min_x, max_x = min(steps), max(steps)
    min_y, max_y = min(losses), max(losses)
    if max_x == min_x: max_x += 1e-9
    if max_y == min_y: max_y += 1e-9
    
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for s, l in zip(steps, losses):
        x = int((s - min_x) / (max_x - min_x) * (width - 1))
        y = int((l - min_y) / (max_y - min_y) * (height - 1))
        grid[height - 1 - y][x] = "•"
        
    out = [f"Max: {max_y:.4f}"]
    out.append("   ┌" + "─" * width + "┐")
    for i, row in enumerate(grid):
        if i == 0: lbl = f"{max_y:.2f}"
        elif i == height - 1: lbl = f"{min_y:.2f}"
        elif i == height // 2: lbl = f"{(max_y+min_y)/2:.2f}"
        else: lbl = ""
        out.append(f"{lbl:>5}│{''.join(row)}│")
        
    out.append("   └" + "─" * width + "┘")
    out.append(f"Min: {min_y:.4f} {' '*(width - 25)} Steps: {min_x}➔{max_x}")
    return "\n".join(out)

try:
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
    steps, losses = [], []
    for log in data.get("log_history", []):
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
    print(draw(steps, losses))
except Exception as e:
    # Print the error so it shows up in Slack instead of failing silently
    print(f"Error parsing JSON for plot: {e}")
EOF
}

_slack_build_payload() {
    local text="$1"
    local payload=""

    local PY_BIN=""
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PY_BIN="python"
    fi

    if [ -n "$PY_BIN" ]; then
        payload="$("$PY_BIN" - <<'PY' "$text"
import json, sys
msg = sys.argv[1] if len(sys.argv) > 1 else ""
print(json.dumps({"text": msg}))
PY
)"
    else
        local escaped_msg
        escaped_msg="$(printf '%s' "$text" | sed 's/\"/\\\\\"/g')"
        payload="{\"text\": \"${escaped_msg}\"}"
    fi

    printf '%s' "$payload"
}

slack_notify() {
    # First arg: exit code. Additional args ignored.
    local rc="$1"
    local phase="${2:-${SLACK_PHASE:-}}"
    local end_ts end_human elapsed status text payload job_id report_file reports_dir error_summary send_slack

    send_slack=0
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        send_slack=1
    fi

    end_ts="$(date +%s)"
    end_human="$(date -Is)"
    elapsed="$((end_ts - ${START_TS:-end_ts}))"
    job_id="${SLACK_JOB_ID:-${SLURM_JOB_ID:-?}}"

    if [ "$rc" -eq 0 ]; then
        status="COMPLETED"
    else
        status="FAILED"
    fi

    local phase_label=""
    if [ -n "$phase" ]; then
        phase_label="[$phase] "
    fi

    text="${phase_label}[$status] ${RUN_NAME:-job} (job ${job_id}) on $(hostname)
Config: ${RUN_NAME:-unknown}
Model:  ${SLACK_MODEL_PATH:-${SLACK_MODEL_TAG:-unknown}}
Nodes:  ${NODE_COUNT:-${SLURM_NNODES:-?}}
Start:  ${START_HUMAN:-unknown}
End:    ${end_human}
Elapsed: $(format_duration "$elapsed")
Exit code: ${rc}"

    if [ "$rc" -ne 0 ] && [ -n "${FAILED_CMD:-}" ]; then
        text="${text}
Failed at: ${FAILED_CMD}"
    fi

    # ---- ADDED: Training Loss Plot Injection ----
    if [ "$phase" = "train" ]; then
        local loss_plot
        loss_plot="$(_slack_extract_training_loss)"
        if [ -n "$loss_plot" ]; then
            text="${text}

*Training Loss:*
\`\`\`
${loss_plot}
\`\`\`"
        fi
    fi
    # ---------------------------------------------

    reports_dir="${SLACK_REPORTS_DIR:-${PROJECT_ROOT:-.}/reports}"
    if [ -n "${SLACK_REPORT_FILE:-}" ]; then
        report_file="$SLACK_REPORT_FILE"
    elif [ -n "${RUN_NAME:-}" ]; then
        report_file="${reports_dir}/R-${RUN_NAME}.${job_id}.err"
    elif [ -n "${SLURM_JOB_NAME:-}" ]; then
        report_file="${reports_dir}/R-${SLURM_JOB_NAME}.${job_id}.err"
    else
        report_file=""
    fi

    if [ -n "$report_file" ] && [ -f "$report_file" ]; then
        if [ "$rc" -eq 0 ]; then
            # If successful, extract the evaluation table and stats
            local eval_results
            eval_results="$(_slack_extract_eval_results "$report_file" "$job_id")"
            if [ -n "$eval_results" ]; then
                text="${text}

*Evaluation Results:*
\`\`\`
${eval_results}
\`\`\`"
            fi

            local excel_row
            excel_row="$(_slack_extract_excel_row "$report_file")"
            if [ -n "$excel_row" ]; then
                text="${text}

*Excel row:*
\`\`\`
${excel_row}
\`\`\`"
            fi
        else
            # If failed, extract the error summary
            local error_summary
            error_summary="$(_slack_find_errors_summary "$report_file")"
            if [ -n "$error_summary" ]; then
                text="${text}

*Detected Error:*
\`\`\`
${error_summary}
\`\`\`"
            fi
        fi
    fi

    if [ -n "${SLACK_TAG:-}" ]; then
        text="${SLACK_TAG} ${text}"
    fi

    echo "== SLACK MESSAGE =="
    printf '%s\n' "$text"

    if [ "$send_slack" -ne 1 ]; then
        return 0
    fi

    payload="$(_slack_build_payload "$text")"

    curl_ssl_flag=()
    if [ "${SLACK_INSECURE:-0}" = "1" ]; then
        curl_ssl_flag+=(--insecure)
    fi

    { set +x; } 2>/dev/null
    curl -sS "${curl_ssl_flag[@]}" -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null || true
    { set -x; } 2>/dev/null
}