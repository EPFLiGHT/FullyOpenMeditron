#!/bin/bash
# Training and the medbench / HealthBench evaluations of every model in the paper. README.md maps each
# model to its config, mode, job and results; AutoMOOVE runs are in launch_automoove.sh.
#
# Training goes to the preemptable partition with --requeue (each run checkpoints every 25 steps and
# resumes); meditron_train.sh chains medbench after it. HealthBench full and hard: answers + Qwen3-30B
# grade, then a Gemma-4-31B regrade of the same answers.
#
# usage (from the repo root): bash axolotl_config/teacher_ablation/launch.sh <mode>
#   teacher      Apertus-8B (2 nodes) and Apertus-70B (16 nodes, the *_16n configs) on both teachers
#   replicates   the Apertus-70B *_normal copies (8 nodes, normal partition): second run per teacher
#   components   Apertus-8B on the gpt-oss corpus minus one training set, medbench
#   components70 the same on Apertus-70B (8 nodes, normal), with HealthBench full
#   bases        OLMo-2-32B-SFT, Gemma-3-27B-IT, EuroLLM-22B-Instruct and EuroLLM-9B-Instruct
#   iclr         medbench and HealthBench of the untrained bases, MedGemma-27B and the 8B ablations
#   hb M L       HealthBench of model M under label L (HB_SUBSETS picks subsets)
#   timeline     HealthBench for Meditron3-70B and Aloe-Beta-70B
#   openweights  HealthBench full for the other open-weight models of Figure 1 and gpt-oss-120b
#   references   medbench for Apertus-8B/70B-Instruct and HealthBench for Apertus-8B/70B-Instruct
#
# EVAL_ONLY=1 skips training and submits a mode's evaluations against the
# existing output_dirs, e.g. to redo evaluations lost to a failure.
# SKIP_TRAINED=1 trains only the configs that have no final weights yet and
# runs every evaluation of the mode, the finished models' included -- for a
# mode half of which already ran.
# Replace a dead training job with resubmit.sh; scripts/job_watchdog.sh can do
# it unattended.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"

XAV=$SCRATCH/meditronfo/models
HF=swiss-ai
CFG=axolotl_config/teacher_ablation
mkdir -p logs

# The shared store where .env puts WandB was at its file limit
# on 2026-09-21; a run that hits it dies at wandb.init, on every requeue.
# Copy the run dirs to the shared WandB directory afterwards.
export WANDB_DIR_OVERRIDE=$SCRATCH/wandb

# sub <sbatch args...>: submit an evaluation job, print its id. The
# reservation was gone from 2026-09-19 and normal-partition jobs pended on
# QOSGrpNodeLimit, so everything runs preemptable and requeues.
sub() { sbatch --parsable -p preemptable --requeue "$@" | tail -1 | cut -d';' -f1; }
# dep <job ids...>: "afterok:a:b", or nothing when there is nothing to wait for
dep() {
    local ids=() i
    for i in "$@"; do [ -n "$i" ] && ids+=("$i"); done
    [ ${#ids[@]} -eq 0 ] || (IFS=:; echo "afterok:${ids[*]}")
}

HB_MODEL_ARGS=()
declare -A TRAIN
PREEMPT=(--partition preemptable --time 23:59:00 --requeue)
train() {  # train <config name> [meditron_train.sh flags]
    local name=$1; shift
    if [ "$EVAL_ONLY" = 1 ]; then
        # medbench, which meditron_train.sh would have chained (with its --eval-args)
        local eval_args=()
        while [ $# -gt 0 ]; do
            [ "$1" = --eval-args ] && { read -r -a eval_args <<< "$2"; break; }
            shift
        done
        echo "medbench $name: $(sub -J "eval_$name" meditron_eval.sh "$XAV/$name" "${eval_args[@]}")"
        TRAIN[$name]=""
        return
    fi
    # SKIP_TRAINED=1: a config whose output_dir already holds final weights
    # (not just a checkpoint-N) is left alone and its evaluations run with no
    # dependency. That makes a mode re-runnable after part of it finished,
    # which is how the bases group was completed on 22 September. Off by
    # default: without it, re-running a mode retrains it, as it always did.
    if [ "${SKIP_TRAINED:-0}" = 1 ] &&
       { [ -f "$XAV/$name/model.safetensors.index.json" ] || [ -f "$XAV/$name/model.safetensors" ]; }; then
        echo "train $name: already trained, skipping ($XAV/$name)"
        TRAIN[$name]=""
        return
    fi
    local out
    out="$(bash meditron_train.sh "$CFG/$name.yaml" "${PREEMPT[@]}" --no-follow "$@")"
    echo "$out"
    TRAIN[$name]="$(echo "$out" | awk '/Submitted Job:/ {print $NF}')"
    [ -n "${TRAIN[$name]}" ] || { echo "no job id for $name"; exit 1; }
}

healthbench() {  # healthbench <model path> <label> <dependency or "">
    # HB_MODEL_ARGS (array) goes to the answering job only, e.g. the reasoning
    # flags of gpt-oss; see the openweights mode.
    local model=$1 label=$2 d=$3 j
    for s in ${HB_SUBSETS:-full hard}; do
        local input=data/healthbench/$s.jsonl t=02:00:00
        local subset=(--subset "$s")
        if [ "$s" = full ]; then
            input=data/healthbench/oss_eval_full.jsonl; t=05:00:00; subset=()
        fi
        local ans="healthbench/results/${s}_qwen30b/results_${label}_${s}.json"
        # HB_GRADE_ONLY=1: the answers exist, redo only the Gemma-4 regrade
        if [ "${HB_GRADE_ONLY:-0}" = 1 ]; then j=""; else
        j=$(sub -J "hb-$label-$s" --time=$t ${d:+--dependency="$d"} \
            --output="logs/R-hb-$label-$s.%j.err" --error="logs/R-hb-$label-$s.%j.err" \
            new_launch.sh --requirements none healthbench/healthbench.py \
            --model "$model" --grader Qwen/Qwen3-30B-A3B-Instruct-2507 "${subset[@]}" \
            --input-path "$input" --tp 4 "${HB_MODEL_ARGS[@]}" \
            --output-dir "healthbench/results/${s}_qwen30b" --output "$ans")
        echo "hb $label $s: $j"
        fi
        j=$(sub -J "hbg-$label-$s" --time=06:00:00 ${j:+--dependency="afterok:$j"} \
            --output="logs/R-hbg-$label-$s.%j.err" --error="logs/R-hbg-$label-$s.%j.err" \
            new_launch.sh --requirements none healthbench/healthbench.py \
            --model "$label" --grader google/gemma-4-31B-it --grader-max-model-len 32768 \
            "${subset[@]}" --input-path "$input" --tp 4 --completions-from "$ans" \
            --output-dir "healthbench/results/${s}_gemma4_31b" \
            --output "healthbench/results/${s}_gemma4_31b/results_${label}_${s}.json")
        echo "hbg $label $s: $j"
    done
}
evals() {  # evals <config name> <label>: HealthBench of a trained model
    healthbench "$XAV/$1" "$2" "$(dep "${TRAIN[$1]}")"
}

case "$MODE" in
teacher)
    train apertus_8b_meditronFO_gptoss_v1 --nodes 2
    train apertus_8b_meditronFO_qwen38_v3 --nodes 2
    train apertus_70b_meditronFO_gptoss_v1_16n --nodes 16
    train apertus_70b_meditronFO_qwen38_v3_16n --nodes 16

    evals apertus_8b_meditronFO_gptoss_v1 meditronFO_gptoss_8b
    evals apertus_8b_meditronFO_qwen38_v3 meditronFO_qwen38_8b
    evals apertus_70b_meditronFO_gptoss_v1_16n meditronFO_gptoss_70b_16n
    evals apertus_70b_meditronFO_qwen38_v3_16n meditronFO_qwen38_70b_16n
    ;;
replicates)
    PREEMPT=(--partition normal --time 11:59:00)
    train apertus_70b_meditronFO_gptoss_v1_normal
    train apertus_70b_meditronFO_qwen38_v3_normal
    evals apertus_70b_meditronFO_gptoss_v1_normal meditronFO_gptoss_70b_normal
    evals apertus_70b_meditronFO_qwen38_v3_normal meditronFO_qwen38_70b_normal
    ;;
components)
    for part in curated_qa synthetic_qa guidelines_qa synthetic_moove; do
        train "apertus_8b_meditronFO_gptoss_v1_no_$part" --nodes 2
    done
    ;;
components70)
    # The four corpus ablations on Apertus-70B (8 nodes each, normal partition, no mid-run
    # checkpoints), with medbench (chained by meditron_train.sh) and HealthBench full. Their
    # AutoMOOVE runs: launch_automoove.sh ablations70b, after the training jobs.
    PREEMPT=(--partition normal --time 11:59:00)
    for part in curated_qa synthetic_qa guidelines_qa synthetic_moove; do
        train "apertus_70b_meditronFO_gptoss_v1_no_$part" --nodes 8
    done
    export HB_SUBSETS="${HB_SUBSETS:-full}"
    for part in curated_qa synthetic_qa guidelines_qa synthetic_moove; do
        evals "apertus_70b_meditronFO_gptoss_v1_no_$part" "meditronFO_gptoss_70b_no_$part"
    done
    ;;
bases)
    # 16 nodes, not 8: both OOMed at step 1 on 8 (see the configs)
    train olmo_2_32b_meditronFO_gptoss_v1 --nodes 16
    train gemma_3_27b_meditronFO_gptoss_v1 --nodes 16 --eval-args "--chat-template none"
    train euro_llm_22b_meditronFO_gptoss_v1 --nodes 8
    train euro_llm_9b_meditronFO_gptoss_v1 --nodes 2

    evals olmo_2_32b_meditronFO_gptoss_v1 meditronFO_gptoss_olmo32b
    evals gemma_3_27b_meditronFO_gptoss_v1 meditronFO_gptoss_gemma3_27b
    evals euro_llm_22b_meditronFO_gptoss_v1 meditronFO_gptoss_euro22b
    evals euro_llm_9b_meditronFO_gptoss_v1 meditronFO_gptoss_euro9b
    ;;
references)
    for m in Apertus-8B-Instruct-2509 Apertus-70B-Instruct-2509; do
        echo "medbench $m: $(sub -J "eval_$m" meditron_eval.sh "$HF/$m")"
    done
    healthbench "$HF/Apertus-8B-Instruct-2509" apertus8b_instruct ""
    healthbench "$HF/Apertus-70B-Instruct-2509" apertus70b_instruct ""
    ;;
iclr)
    # ICLR_ONLY="<parts>" resubmits only some parts, e.g. after a failure:
    # medbench:<model basename>, hb:<label> (with HB_SUBSETS).
    want() { [ -z "${ICLR_ONLY:-}" ] || [[ " $ICLR_ONLY " == *" $1 "* ]]; }
    # The hub repo of OLMo-2-32B-SFT has no tokenizer.json, and the medbench
    # container cannot build one from the slow tokenizer; this directory links
    # the snapshot and adds the (identical) tokenizer.json of its finetune.
    OLMO=$SCRATCH/models/OLMo-2-0325-32B-SFT
    #
    # 1. medbench for the five untrained bases.
    #    The Gemma family runs without a chat template, like its finetune.
    for m in "$OLMO" utter-project/EuroLLM-22B-Instruct-2512 utter-project/EuroLLM-9B-Instruct; do
        want "medbench:${m##*/}" || continue
        echo "medbench $m: $(sub -J "eval_${m##*/}" meditron_eval.sh "$m")"
    done
    for m in google/gemma-3-27b-it google/medgemma-27b-text-it; do
        want "medbench:${m##*/}" || continue
        echo "medbench $m: $(sub -J "eval_${m##*/}" meditron_eval.sh "$m" --chat-template none)"
    done

    # 2. HealthBench for the same bases and the four corpus ablations.
    hbw() { want "hb:$2" && healthbench "$1" "$2" "" || true; }
    hbw allenai/OLMo-2-0325-32B-SFT base_olmo32b_sft
    hbw utter-project/EuroLLM-22B-Instruct-2512 base_euro22b_instruct
    hbw utter-project/EuroLLM-9B-Instruct base_euro9b_instruct
    hbw google/gemma-3-27b-it base_gemma3_27b_it
    hbw google/medgemma-27b-text-it base_medgemma27b
    for part in curated_qa synthetic_qa guidelines_qa synthetic_moove; do
        hbw "$XAV/apertus_8b_meditronFO_gptoss_v1_no_$part" "meditronFO_gptoss_8b_no_$part"
    done
    # 3. AutoMOOVE: axolotl_config/teacher_ablation/launch_automoove.sh
    ;;
hb)
    # hb <model> <label>: one model's HealthBench, e.g. to replace a failed
    # subset: HB_SUBSETS=hard launch.sh hb <model> <label>
    healthbench "$2" "$3" ""
    ;;
timeline)
    # The open-weight reference points of the paper's HealthBench-over-time
    # figure, regraded on the paper's new scale (Gemma-4-31B regrade column).
    healthbench OpenMeditron/Meditron3-70B ref_meditron3_70b ""
    healthbench HPAI-BSC/Llama3.1-Aloe-Beta-70B ref_aloe_beta_70b ""
    ;;
openweights)
    # HealthBench full for the extra open-weight points of Figure 1d
    # (tech_reports/iclr_clean_reproduction.tex): small medical models with
    # their general bases, the Llama-3.1 bases of Aloe and Meditron-3, and
    # gpt-oss-120b (our teacher; OpenAI reports its HealthBench). Download the
    # weights on the login node first: jobs run with HF_HUB_OFFLINE.
    # OPENW_ONLY="label ..." submits only those labels.
    export HB_SUBSETS="${HB_SUBSETS:-full}"
    ow() { [ -z "${OPENW_ONLY:-}" ] || [[ " $OPENW_ONLY " == *" $2 "* ]] || return 0; healthbench "$1" "$2" ""; }
    ow google/medgemma-4b-it ref_medgemma4b
    ow google/gemma-3-4b-it base_gemma3_4b_it
    ow HPAI-BSC/Llama3.1-Aloe-Beta-8B ref_aloe_beta_8b
    ow <org>/Meditron3-8B ref_meditron3_8b
    ow meta-llama/Llama-3.1-8B-Instruct base_llama31_8b_instruct
    ow meta-llama/Llama-3.1-70B-Instruct base_llama31_70b_instruct
    # gpt-oss-120b: low reasoning effort, 4096 tokens for reasoning and answer
    HB_MODEL_ARGS=(--model-llm-kwargs '{"max_model_len":32768}'
        --chat-template-kwargs '{"reasoning_effort":"low"}'
        --reasoning-format harmony --max-tokens 4096)
    ow openai/gpt-oss-120b ref_gptoss_120b
    HB_MODEL_ARGS=()
    ;;
*)
    sed -n '2,36p' "$0"; exit 1 ;;
esac
