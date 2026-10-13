#!/bin/bash
# Keep resumable Slurm jobs alive past the preemptable partition's churn.
#
# Slurm requeues a preempted --requeue job itself, but only MaxBatchRequeue
# (5 on the cluster) times; after that the job is held (JobHoldMaxRequeue), and a
# crash on resume (e.g. an OOM loading the checkpoint) ends it for good. This
# loop notices both and resubmits the job's exact original sbatch line, which
# resumes from the last checkpoint, then repoints any pending job that
# depended on the old ID (the chained evals) at the new one.
#
# It runs on the login node, detached, so it outlives the editor session:
#   bash scripts/job_watchdog.sh start <jobid> [<jobid> ...]
#   bash scripts/job_watchdog.sh add <jobid> [...]   # track more, while running
#   bash scripts/job_watchdog.sh status | stop
#
# State lives on scratch: jobs.tsv holds "<current id> <resubmits> <name>".
# Jobs that COMPLETE or that you scancel yourself are dropped from it.

WD_DIR="${WD_DIR:-$SCRATCH/job_watchdog}"
STATE="$WD_DIR/jobs.tsv"
LOG="$WD_DIR/watchdog.log"
PIDFILE="$WD_DIR/watchdog.pid"
INTERVAL="${WD_INTERVAL:-300}"
MAX_RESUBMITS="${WD_MAX_RESUBMITS:-6}"

mkdir -p "$WD_DIR"
[ -f "$STATE" ] || : > "$STATE"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

add_jobs() {
    for id in "$@"; do
        grep -q "^$id " "$STATE" && continue
        name="$(sacct -j "$id" -X -n -o JobName%200 | head -1 | xargs)"
        echo "$id 0 $name" >> "$STATE"
        log "tracking $id ($name)"
    done
}

running_pid() {
    [ -f "$PIDFILE" ] || return 1
    read -r host pid < "$PIDFILE"
    [ "$host" = "$(hostname)" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

# Resubmit a dead job from its own submit line; echo the new job id.
resubmit() {
    local id="$1" line dir out new
    line="$(sacct -j "$id" -X -n -o SubmitLine%2000 | head -1 | xargs)"
    dir="$(sacct -j "$id" -X -n -o WorkDir%500 | head -1 | xargs)"
    [[ "$line" == sbatch* ]] || { log "$id: no usable submit line ('$line')"; return 1; }
    out="$(cd "$dir" && eval "$line" 2>&1)"
    # "Submitted batch job N", or with --parsable a bare "N[;cluster]" line
    # (after the --environment banner, which goes to the same stream)
    new="$(grep -oE 'Submitted batch job [0-9]+' <<< "$out" | awk '{print $4}')"
    [ -n "$new" ] || new="$(grep -oE '^[0-9]+(;|$)' <<< "$out" | tail -1 | tr -d ';')"
    [ -n "$new" ] || { log "$id: resubmit failed: $out"; return 1; }
    # Repoint pending dependents (afterok:<id>(failed) etc.) at the new job.
    squeue -u "$USER" -h -t PD -o '%i %E' | while read -r dep deps; do
        [[ "$deps" =~ (^|[^0-9])$id([^0-9]|$) ]] || continue
        fixed="$(sed -E 's/\([^)]*\)//g; s/(^|[^0-9])'"$id"'([^0-9]|$)/\1'"$new"'\2/g' <<< "$deps")"
        if scontrol update JobId="$dep" Dependency="$fixed" 2>>"$LOG"; then
            log "  dependent $dep: $deps -> $fixed"
        fi
    done
    echo "$new"
}

check_once() {
    local tmp="$STATE.tmp" id n name st reason final new
    : > "$tmp"
    while read -r id n name; do
        [ -n "$id" ] || continue
        read -r st reason <<< "$(squeue -h -j "$id" -o '%T %r' 2>/dev/null)"
        # squeue -j still lists a job for MinJobAge after it ends; only live
        # states count as queued.
        [[ "$st" =~ ^(PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUE.*|RESIZING)$ ]] || st=""
        if [ -n "$st" ]; then
            if [ "$st" = PENDING ] && [[ "${reason,,}" =~ held|hold ]] && [ "$reason" != JobHeldUser ]; then
                if scontrol release "$id" 2>/dev/null; then
                    log "$id ($name): released hold ($reason)"
                else
                    log "$id ($name): held ($reason), release refused; resubmitting"
                    scancel "$id"
                    sleep 5
                    st=""; final="HELD"
                fi
            fi
            [ -n "$st" ] && { echo "$id $n $name" >> "$tmp"; continue; }
        fi
        [ -n "$final" ] || final="$(sacct -j "$id" -X -n -o State%40 | tail -1 | xargs)"
        case "$final" in
            COMPLETED)
                log "$id ($name): COMPLETED, done" ;;
            PENDING|RUNNING|REQUEUED|SUSPENDED|COMPLETING|"")
                # squeue missed a live job (slurmctld busy): never resubmit it.
                echo "$id $n $name" >> "$tmp" ;;
            CANCELLED*)
                log "$id ($name): $final, assuming you cancelled it; dropping" ;;
            *)
                if [ "$n" -ge "$MAX_RESUBMITS" ]; then
                    log "$id ($name): $final, but $n resubmits already; giving up"
                elif new="$(resubmit "$id")"; then
                    log "$id ($name): $final -> resubmitted as $new ($((n + 1))/$MAX_RESUBMITS)"
                    echo "$new $((n + 1)) $name" >> "$tmp"
                else
                    echo "$id $n $name" >> "$tmp"   # retry next round
                fi ;;
        esac
        final=""
    done < "$STATE"
    mv "$tmp" "$STATE"
}

case "$1" in
    start)
        shift; add_jobs "$@"
        if pid="$(running_pid)"; then echo "already running (pid $pid)"; exit 0; fi
        setsid nohup bash "$0" _loop < /dev/null >> "$LOG" 2>&1 &
        echo "$(hostname) $!" > "$PIDFILE"
        echo "watchdog started on $(hostname), pid $!; log: $LOG" ;;
    add)
        shift; add_jobs "$@" ;;
    stop)
        pid="$(running_pid)" && kill "$pid" && echo "stopped $pid" || echo "not running here" ;;
    status)
        # Every pass rewrites jobs.tsv, so its age shows the loop is alive from
        # any login node, not just the one it runs on.
        age=$(( $(date +%s) - $(stat -c %Y "$STATE") ))
        echo "loop on $(cat "$PIDFILE" 2>/dev/null); last pass ${age}s ago (every ${INTERVAL}s)"
        [ "$age" -gt $((2 * INTERVAL + 60)) ] && echo "STALE: watchdog looks dead; rerun start"
        cat "$STATE"; tail -n 20 "$LOG" ;;
    _loop)
        log "loop up on $(hostname), every ${INTERVAL}s"
        while true; do check_once; sleep "$INTERVAL"; done ;;
    *)
        sed -n '2,20p' "$0" ;;
esac
