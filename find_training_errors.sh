#!/usr/bin/env bash
set -euo pipefail

REPORTS_DIR="${1:-train_reports}"

if [ ! -d "$REPORTS_DIR" ]; then
  echo "Reports directory not found: $REPORTS_DIR" >&2
  exit 1
fi

shopt -s nullglob
mapfile -d '' report_files < <(find "$REPORTS_DIR" -maxdepth 1 -type f -print0)
if [ "${#report_files[@]}" -eq 0 ]; then
  echo "No report files found in $REPORTS_DIR" >&2
  exit 0
fi

sorted_reports=()
while IFS= read -r report; do
  sorted_reports+=("$report")
done < <(
  printf '%s\0' "${report_files[@]}" \
    | xargs -0 stat -c '%Y %n' \
    | sort -n \
    | awk '{$1=""; sub(/^ /,""); print}'
)

if command -v rg >/dev/null 2>&1; then
  search_cmd=(rg -n -m1 -i)
  nan_pattern="\\bnan\\b|loss is nan|nan loss"
else
  search_cmd=(grep -n -m1 -i -E)
  nan_pattern="\\<nan\\>|loss is nan|nan loss"
fi

print_elapsed() {
  local report="$1"
  local elapsed=""
  if command -v rg >/dev/null 2>&1; then
    elapsed=$(
      {
        rg -i "elapsed" "$report" 2>/dev/null \
          | rg -o -e '[0-9]{2}:[0-9]{2}:[0-9]{2}' 2>/dev/null \
          | tail -n1
      } || true
    )
  else
    elapsed=$(
      {
        grep -i "elapsed" "$report" 2>/dev/null \
          | grep -E -o '[0-9]{2}:[0-9]{2}:[0-9]{2}' 2>/dev/null \
          | tail -n1
      } || true
    )
  fi

  if [ -n "$elapsed" ]; then
    echo "Elapsed: $elapsed"
  else
    echo "Elapsed: n/a"
  fi
}

print_progress() {
  local report="$1"

  # Find the last tqdm-style progress marker: e.g. " 62%|...| 243/391 [2:23:27<1:28:02"
  # tqdm output may be embedded mid-line, so we use grep -o to extract just the bar segment.
  local progress_line=""
  if command -v rg >/dev/null 2>&1; then
    progress_line=$(
      rg -o -e '[0-9]{1,3}%\|[^|]*\|[[:space:]]*[0-9]+/[0-9]+[[:space:]]*\[[^]]+\]' "$report" 2>/dev/null \
        | tail -n1 || true
    )
  else
    progress_line=$(
      grep -E -o '[0-9]{1,3}%\|[^|]*\|[[:space:]]*[0-9]+/[0-9]+[[:space:]]*\[[^]]+\]' "$report" 2>/dev/null \
        | tail -n1 || true
    )
  fi

  # Latest loss + epoch from the metric dicts
  local last_loss=""
  local last_epoch=""
  if command -v rg >/dev/null 2>&1; then
    last_loss=$(rg -o -e "'loss':[[:space:]]*[0-9.]+" "$report" 2>/dev/null | tail -n1 || true)
    last_epoch=$(rg -o -e "'epoch':[[:space:]]*[0-9.]+" "$report" 2>/dev/null | tail -n1 || true)
  else
    last_loss=$(grep -E -o "'loss':[[:space:]]*[0-9.]+" "$report" 2>/dev/null | tail -n1 || true)
    last_epoch=$(grep -E -o "'epoch':[[:space:]]*[0-9.]+" "$report" 2>/dev/null | tail -n1 || true)
  fi

  if [ -n "$progress_line" ]; then
    echo "Progress: $progress_line"
  fi

  if [ -n "$last_loss" ] || [ -n "$last_epoch" ]; then
    echo "Latest:  ${last_loss:-no loss} | ${last_epoch:-no epoch}"
  fi
}

get_elapsed() {
  local report="$1"
  local elapsed=""
  if command -v rg >/dev/null 2>&1; then
    elapsed=$(
      {
        rg -i "elapsed" "$report" 2>/dev/null \
          | rg -o -e '[0-9]{2}:[0-9]{2}:[0-9]{2}' 2>/dev/null \
          | tail -n1
      } || true
    )
  else
    elapsed=$(
      {
        grep -i "elapsed" "$report" 2>/dev/null \
          | grep -E -o '[0-9]{2}:[0-9]{2}:[0-9]{2}' 2>/dev/null \
          | tail -n1
      } || true
    )
  fi
  printf '%s' "$elapsed"
}

for report in "${sorted_reports[@]}"; do
  echo "== $report =="

  status="IDK"
  if "${search_cmd[@]}" "cuda out of memory|cudnn_status_alloc_failed|cublas_status_alloc_failed|out of memory" "$report"; then
    status="OOM"
  elif "${search_cmd[@]}" "$nan_pattern" "$report"; then
    status="NAN"
  elif "${search_cmd[@]}" "too many open files|errno 24" "$report"; then
    status="FD_LIMIT"
  elif "${search_cmd[@]}" "training finished|training completed" "$report"; then
    echo "FINISHED"
    status="FINISHED"
  fi

  elapsed=$(get_elapsed "$report")

  # Show progress only when elapsed is unavailable AND the run didn't finish
  if [ "$status" != "FINISHED" ] && [ -z "$elapsed" ]; then
    print_progress "$report"
  fi

  if [ -n "$elapsed" ]; then
    echo "Elapsed: $elapsed"
  else
    echo "Elapsed: n/a"
  fi

  echo
done