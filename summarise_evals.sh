#!/usr/bin/env bash
# summarise_evals.sh
set -euo pipefail

# ==========================================
# 1. Argument Parsing & Setup
# ==========================================
SHOW_SAMPLE=0
ONLY_COMPLETED=0
NO_INVALIDS=1
args=()

for arg in "$@"; do
  if [ "$arg" = "--show_sample" ]; then
    SHOW_SAMPLE=1
  elif [ "$arg" = "--completed" ]; then
    ONLY_COMPLETED=1
  elif [ "$arg" = "--invalids" ]; then
    NO_INVALIDS=0
  else
    args+=("$arg")
  fi
done

REPORTS_DIR="${args[0]:-eval_reports}"
EVAL_RESULTS_DIR="${args[1]:-eval_results}"

if [ ! -d "$REPORTS_DIR" ]; then
  echo "Reports directory not found: $REPORTS_DIR" >&2
  exit 1
fi

if [ ! -d "$EVAL_RESULTS_DIR" ]; then
  echo "Eval results directory not found: $EVAL_RESULTS_DIR" >&2
  exit 1
fi

# Determine the best search tool
if command -v rg >/dev/null 2>&1; then
  line_cmd=(rg -n -m1)
  grep_cmd=(rg -n)
  has_rg=1
else
  line_cmd=(grep -n -m1)
  grep_cmd=(grep -n)
  has_rg=0
fi

# ==========================================
# 2. Helper Functions
# ==========================================

count_invalids_recursive() {
  local dir="$1"
  local total=0
  if [ "$has_rg" -eq 1 ]; then
    total=$(
      find "$dir" -type f -name "*.jsonl" -print0 \
        | xargs -0 -r rg -o -i "\\[invalid\\]" 2>/dev/null \
        | wc -l | tr -d ' '
    )
  else
    total=$(
      find "$dir" -type f -name "*.jsonl" -print0 \
        | xargs -0 -r grep -o -i "\\[invalid\\]" 2>/dev/null \
        | wc -l | tr -d ' '
    )
  fi
  echo "$total"
}

count_lines_recursive() {
  local dir="$1"
  find "$dir" -type f -name "*.jsonl" -print0 \
    | xargs -0 -r wc -l 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

eval_resps_stats() {
  local eval_dir="$1"
  local jsonl_path=""
  jsonl_path="$(find "$eval_dir" -type f -name "*.jsonl" | sort | head -n1)"
  
  if [ -z "$jsonl_path" ]; then
    echo -e "n/a\tn/a\tn/a"
    return
  fi

  # Preserved exact Python logic from original script
  python3 - "$jsonl_path" <<'PY' 2>/dev/null || true
import json
import sys

path = sys.argv[1]
first_resps = "n/a"
first_len = "n/a"
total = 0
count = 0

with open(path, "r", encoding="utf-8") as handle:
    for idx, line in enumerate(handle):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        resps = obj.get("resps")
        if isinstance(resps, list) and resps:
            first = resps[0]
            text = None
            if isinstance(first, list) and first:
                if isinstance(first[0], str):
                    text = first[0]
            elif isinstance(first, str):
                text = first
            if text:
                if first_resps == "n/a":
                    first_resps = str(resps)
                    first_len = str(len(text))
                total += len(text)
                count += 1
        if idx == 0 and first_resps == "n/a":
            break

mean_len = f"{total / count:.1f}" if count else "n/a"
print(f"{first_resps}\t{first_len}\t{mean_len}")
PY
}

extract_excel_row() {
  local report="$1"
  local model_name="$2"
  python3 - "$report" "$model_name" <<'PY' 2>/dev/null || true
import re, sys

with open(sys.argv[1], "r", errors="replace") as f:
    text = f.read()
model_name = sys.argv[2] if len(sys.argv) > 2 else ""

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
arc_challenge = g("arc_challenge")


m = re.search(r"\|\s*prompt_level_strict_acc\s*\|[^|]*\|\s*([0-9.]+)\s*\|", text)
ifeval = m.group(1) if m else ""

def pct(v):
    return f"{float(v) * 100:.2f}" if v else ""

def avg(vals):
    nums = [float(v) for v in vals if v]
    return f"{(sum(nums) / len(nums)) * 100:.2f}" if nums else ""

med_avg = avg([medmcqa, medqa, pubmedqa])
gen_avg = avg([mmlu_pro, ifeval, arc_challenge])

# Columns: model | medmcqa | medqa | pubmedqa | avg | weighted_avg | medxpertqa | gain | mmlu_pro | ifeval | arc_challenge | avg | gain
row = [
    model_name,
    pct(medmcqa), pct(medqa), pct(pubmedqa), med_avg, "",
    pct(medxpert), "",
    pct(mmlu_pro), pct(ifeval), pct(arc_challenge), gen_avg, "",
]
print(",".join(row))
PY
}

print_invalids_for_report() {
  local report="$1"
  local base
  local job_id=""
  
  base="$(basename "$report")"
  if [[ "$base" =~ \.([0-9]+)\. ]]; then
    job_id="${BASH_REMATCH[1]}"
  fi
  
  if [ -z "$job_id" ]; then
    echo "Invalids: n/a"
    return
  fi

  mapfile -t matches < <(find "$EVAL_RESULTS_DIR" -type d -name "*_${job_id}" | sort)
  if [ "${#matches[@]}" -eq 0 ]; then
    echo "Invalids: n/a"
    echo "Eval folder: n/a"
    if [ "$SHOW_SAMPLE" -eq 1 ]; then
      echo "First resps: n/a"
    fi
    echo "Resps mean length: n/a"
    echo "Hit max length: n/a"
    return
  fi

  echo "Eval folder: ${matches[*]}"
  echo "Invalids per eval:"

  local grand_invalid=0
  local grand_total=0
  local jsonl_files=()

  # Collect jsonl files from all matched dirs (plain newline-sorted; paths shouldn't contain newlines)
  local dir
  for dir in "${matches[@]}"; do
    while IFS= read -r f; do
      [ -n "$f" ] && jsonl_files+=("$f")
    done < <(find "$dir" -type f -name "*.jsonl" 2>/dev/null | sort)
  done

  if [ "${#jsonl_files[@]}" -eq 0 ]; then
    echo "  (no jsonl files found)"
  else
    local jsonl fname invalids lines pct
    for jsonl in "${jsonl_files[@]}"; do
      fname="$(basename "$jsonl" .jsonl)"
      fname="${fname#samples_}"

      invalids=0
      if [ "$has_rg" -eq 1 ]; then
        invalids=$(rg -o -i '\[invalid\]' "$jsonl" 2>/dev/null | wc -l | tr -d ' ' || true)
      else
        invalids=$(grep -o -i '\[invalid\]' "$jsonl" 2>/dev/null | wc -l | tr -d ' ' || true)
      fi
      invalids=${invalids:-0}

      lines=$(wc -l < "$jsonl" 2>/dev/null | tr -d ' ' || echo 0)
      lines=${lines:-0}

      if [ "$lines" -gt 0 ]; then
        pct=$(awk -v a="$invalids" -v b="$lines" 'BEGIN {printf "%.2f", (a / b) * 100}')
        printf '  %-50s %s/%s (%s%%)\n' "$fname" "$invalids" "$lines" "$pct"
      else
        printf '  %-50s %s/%s\n' "$fname" "$invalids" "$lines"
      fi
      grand_invalid=$((grand_invalid + invalids))
      grand_total=$((grand_total + lines))
    done
  fi

  if [ "$grand_total" -gt 0 ]; then
    local grand_pct
    grand_pct=$(awk -v a="$grand_invalid" -v b="$grand_total" 'BEGIN {printf "%.2f", (a / b) * 100}')
    echo "Invalids total: ${grand_invalid}/${grand_total} (${grand_pct}%)"
  else
    echo "Invalids total: 0/0"
  fi

  IFS=$'\t' read -r first_resps first_len mean_len hit_max < <(eval_resps_stats "${matches[0]}")
  if [ "$SHOW_SAMPLE" -eq 1 ]; then
    echo "First resps: ${first_resps}"
  fi
  echo "Resps mean length: ${mean_len}"
  echo "Hit max length: ${hit_max}"
}

# ==========================================
# 3. Main Execution Loop
# ==========================================

shopt -s nullglob
mapfile -d '' report_files < <(find "$REPORTS_DIR" -maxdepth 1 -type f -print0)
if [ "${#report_files[@]}" -eq 0 ]; then
  echo "No report files found in $REPORTS_DIR" >&2
  exit 0
fi

# Preserved exact sorting logic
sorted_reports=()
while IFS= read -r report; do
  sorted_reports+=("$report")
done < <(
  printf '%s\0' "${report_files[@]}" \
    | xargs -0 stat -c '%Y %n' \
    | sort -n \
    | awk '{$1=""; sub(/^ /,""); print}'
)

for report in "${sorted_reports[@]}"; do
  # Preserved exact exit code parsing
  exit_code="$(awk -F'Exit code: ' '/Exit code:/{code=$2} END{if(code!="") print code}' "$report" | sed -E 's/[^0-9].*$//')"
  status="UNKNOWN"
  
  if [ -n "$exit_code" ]; then
    if [ "$exit_code" = "0" ]; then
      status="COMPLETED"
    else
      status="FAILED (exit code $exit_code)"
    fi
  fi
  
  if [ "$ONLY_COMPLETED" -eq 1 ] && [ "$status" != "COMPLETED" ]; then
    continue
  fi

  echo "== $report =="

  # Preserved exact hierarchical model parsing (MODEL_PATH then pretrained)
  model_line="$("${line_cmd[@]}" "MODEL_PATH=" "$report" || true)"
  model=""
  if [ -n "$model_line" ]; then
    model="${model_line#*MODEL_PATH=}"
  else
    model_line="$("${line_cmd[@]}" "pretrained=" "$report" || true)"
    if [ -n "$model_line" ]; then
      model="$(printf '%s' "$model_line" | sed -E 's/.*pretrained=([^, ]+).*/\1/')"
    fi
  fi

  if [ -n "$model" ]; then
    echo "Model: $model"
  else
    echo "Model: UNKNOWN"
  fi

  echo "Status: $status"

  # Preserved exact results table parsing
  results_table="$(awk '
    /^\|[[:space:]]*Tasks[[:space:]]*\|/ {in_table=1; print; next}
    in_table && /^\|/ {print; next}
    in_table {exit}
  ' "$report")"

  if [ -n "$results_table" ]; then
    echo "Results:"
    printf '%s\n' "$results_table"
    [ "$NO_INVALIDS" -eq 0 ] && print_invalids_for_report "$report"
    excel_row="$(extract_excel_row "$report" "$model")"
    if [ -n "$excel_row" ]; then
      echo "Excel row:"
      printf '%s\n' "$excel_row"
    fi
  else
    error_summary=""
    if "${grep_cmd[@]}" "Traceback" "$report" >/dev/null 2>&1; then
      error_summary="$(awk '
        /Traceback/ {tb=1}
        tb && ($0 ~ /(Error|Exception|KeyError|RuntimeError|ValueError|TypeError|AssertionError|ChildFailedError)/) {last=$0}
        END {if (last!="") print last}
      ' "$report")"
    fi

    if [ -z "$error_summary" ]; then
      # Preserved exact fallback error parsing
      error_summary="$("${line_cmd[@]}" "ERROR:|Error:|FAILED|exit code: [1-9]" "$report" 2>/dev/null || true)"
    fi

    if [ -n "$error_summary" ]; then
      echo "Error: $error_summary"
    else
      echo "Error: UNKNOWN"
    fi
    [ "$NO_INVALIDS" -eq 0 ] && print_invalids_for_report "$report"
  fi

  echo
done