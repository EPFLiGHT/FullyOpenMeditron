#!/bin/bash
# Replace a dead training job (REQUEUE_HOLD after a failed relaunch, which only
# an admin can release, or FAILED) by a fresh one on the same config, and move
# every pending job that depends on it to the new job. The config's output_dir
# is unchanged, so the new job resumes from the newest complete checkpoint.
#
# usage: bash axolotl_config/teacher_ablation/resubmit.sh <old job id> <config name> [meditron_train.sh flags]
#   e.g. ... resubmit.sh 3464364 apertus_8b_meditronFO_gptoss_v1 --nodes 2
set -euo pipefail
cd "$(dirname "$0")/../.."
OLD=$1 NAME=$2; shift 2
export WANDB_DIR_OVERRIDE=$SCRATCH/wandb

out="$(bash meditron_train.sh "axolotl_config/teacher_ablation/$NAME.yaml" --partition preemptable \
    --time 23:59:00 --requeue --no-follow --no-eval "$@" 2>&1)"
NEW="$(echo "$out" | awk '/Submitted Job:/ {print $NF}')"
[ -n "$NEW" ] || { echo "$out"; echo "no job id"; exit 1; }
echo "$NAME: $OLD -> $NEW"

# dependents: pending jobs whose Dependency names $OLD
for j in $(squeue -u "$USER" -h -t PD -o %i); do
    dep="$(scontrol show job "$j" | grep -oE 'Dependency=[^ ]+' | cut -d= -f2-)"
    [ "$dep" = "(null)" ] && continue
    echo "$dep" | grep -qE "(^|[:,])$OLD(\(|:|,|$)" || continue
    new_dep="$(echo "$dep" | sed -E 's/\([^)]*\)//g' | sed -E "s/(^|[:,])$OLD(:|,|$)/\1$NEW\2/g")"
    scontrol update JobId="$j" Dependency="$new_dep"
    echo "  $j $(squeue -j "$j" -h -o %j): $new_dep"
done

# a held job an admin later releases would train into the same output_dir
if squeue -j "$OLD" -h 2>/dev/null | grep -q .; then scancel "$OLD"; echo "cancelled $OLD"; fi
