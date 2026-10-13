#!/bin/bash
# The paper's AutoMOOVE runs on the unanimous-576 test set (516 questions): both answer models with no
# system prompt, temperature 0.7, seed 42, both answer orders, results in $U.
#   default      6 MeditronFO-vs-base pairs x {winner: Gemma-4, Qwen3.8, Muse-Glimmer; moove_pref: Gemma-4;
#                Likert (rubric moove): Gemma-4, Qwen3.8} and 2 MeditronFO-vs-MedGemma pairs x
#                {winner: Gemma-4, Qwen3.8, Muse-Glimmer, Nemotron}
#   ablations8b  the four Apertus-8B corpus ablations against Apertus-8B-Instruct (winner + moove_pref, Gemma-4)
#   ablations70b the four Apertus-70B corpus ablations against Apertus-70B-Instruct (the same), after the
#                default mode's Apertus-70B chain (AFTER=<job id>) and their training (launch.sh components70)
#
# The answer cache is a single-writer shelve: jobs that share a model run one after another (afterany,
# so one failure does not strand the rest of its chain). The first job of a chain generates both models'
# answers; the Gemma-3-vs-MedGemma pairs wait for the Apertus-70B chain, which writes the MedGemma answers,
# so both MedGemma pairs are judged against the same answers. Muse-Glimmer and Nemotron judge from the
# cache in the vLLM 0.30 container.
#
# usage (from the repo root): bash axolotl_config/teacher_ablation/launch_automoove.sh [all|ablations8b|ablations70b]
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

XAV=$SCRATCH/meditronfo/models
HF=swiss-ai
U=auto_moove/results/unanimous576
export AUTO_MOOVE_CACHE_ROOT=$SCRATCH/meditronfo/auto_moove_cache
J030=$PWD/container/vllm_030.toml

sub() { sbatch --parsable -p preemptable --requeue "$@" | tail -1 | cut -d';' -f1; }
LAST=""
job() {  # job <name> <model1> <model2> <judge key> <rubric key> [auto_moove.py args...], after $LAST
    local name=$1 m1=$2 m2=$3 jk=$4 rk=$5; shift 5
    local judge rubric script=new_launch.sh envopt=()
    case $jk in
        g4)   judge=google/gemma-4-31B-it ;;
        q38)  judge=Qwen/Qwen3.8-27B ;;
        muse) judge=meta-models/Muse-Glimmer-30B ;;
        nemo) judge=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 ;;
    esac
    case $rk in
        win)  rubric=(--rubric winner --scoring logprob) ;;
        lik)  rubric=(--rubric moove) ;;
        pref) rubric=(--rubric moove_pref) ;;
    esac
    # new_launch.sh ignores --env inside a batch job, so the vLLM 0.30 container
    # goes to sbatch, with a copy of new_launch.sh without its #SBATCH --environment
    if [ "$jk" = muse ] || [ "$jk" = nemo ]; then
        envopt=(--environment="$J030"); script=$(mktemp)
        grep -v '^#SBATCH --environment' new_launch.sh > "$script"
    fi
    LAST=$(sub -J "$name" --time=04:00:00 ${LAST:+--dependency=afterany:$LAST} "${envopt[@]}" \
        --output="logs/R-$name.%j.err" --error="logs/R-$name.%j.err" \
        "$script" --requirements none auto_moove/auto_moove.py \
        --input data/moove/moove_eval_unanimous_576.jsonl --output "$U" \
        --temperature 0.7 --max-new-tokens 8192 --max-model-len 16384 \
        --judge-max-model-len 32768 --bidirectional --judge-utilization 0.5 \
        --model1 "$m1" --model2 "$m2" --judge "$judge" "${rubric[@]}" "$@")
    [ "$script" = new_launch.sh ] || rm -f "$script"
    echo "$name: $LAST"
}
base_pair() {  # base_pair <tag> <finetune> <base> [auto_moove.py args...]
    local t=$1 m1=$2 m2=$3; shift 3
    job "n-$t-win"      "$m1" "$m2" g4   win  "$@"   # generates both models' answers
    job "n-$t-pref"     "$m1" "$m2" g4   pref "$@"
    job "n-$t-win-q38"  "$m1" "$m2" q38  win  "$@"
    job "n-$t-win-muse" "$m1" "$m2" muse win  "$@"
    # Likert scores: read by tech_reports/figures/iclr_clean_ci.py
    job "n-$t-lik"      "$m1" "$m2" g4   lik  "$@"
    job "n-$t-lik-q38"  "$m1" "$m2" q38  lik  "$@"
}
mg_pair() {  # mg_pair <tag> <finetune>: the 4 judges against MedGemma-27B
    local t=$1 m1=$2 MG=google/medgemma-27b-text-it
    job "n-$t-mg-win"      "$m1" "$MG" g4   win
    job "n-$t-mg-win-q38"  "$m1" "$MG" q38  win
    job "n-$t-mg-win-muse" "$m1" "$MG" muse win
    job "n-$t-mg-win-nemo" "$m1" "$MG" nemo win
}
# OLMo-2-32B-SFT and EuroLLM-9B-Instruct have a 4096-token context; their longest question is 687 tokens
CTX4K=(--max-new-tokens 3072 --max-model-len 4096)

case "${1:-all}" in
ablations8b)
    # one chain: all four share the Apertus-8B-Instruct answers
    LAST=""
    for part in curated_qa synthetic_qa guidelines_qa synthetic_moove; do
        m=$XAV/apertus_8b_meditronFO_gptoss_v1_no_$part
        job "n-8b_no_$part-win"  "$m" "$HF/Apertus-8B-Instruct-2509" g4 win
        job "n-8b_no_$part-pref" "$m" "$HF/Apertus-8B-Instruct-2509" g4 pref
    done
    ;;
ablations70b)
    # one chain on the Apertus-70B-Instruct answers of the default mode's chain A
    LAST=${AFTER:-}
    for part in curated_qa synthetic_qa guidelines_qa synthetic_moove; do
        m=$XAV/apertus_70b_meditronFO_gptoss_v1_no_$part
        job "n-70b_no_$part-win"  "$m" "$HF/Apertus-70B-Instruct-2509" g4 win
        job "n-70b_no_$part-pref" "$m" "$HF/Apertus-70B-Instruct-2509" g4 pref
    done
    ;;
all)
    M70=$XAV/apertus_70b_meditronFO_gptoss_v1_16n
    MGEM=$XAV/gemma_3_27b_meditronFO_gptoss_v1
    # chain A: Apertus-70B, then its MedGemma pair (writes the MedGemma answers)
    LAST=""
    base_pair 70b-inst "$M70" "$HF/Apertus-70B-Instruct-2509"
    mg_pair 70b "$M70"
    A_LAST=$LAST
    # chain B: Gemma-3-27B, then its MedGemma pair after chain A as well
    LAST=""
    base_pair gemma-it "$MGEM" google/gemma-3-27b-it
    LAST="$LAST:$A_LAST"
    mg_pair gemma "$MGEM"
    # chains C-F share no model
    LAST=""
    base_pair 8b "$XAV/apertus_8b_meditronFO_gptoss_v1" "$HF/Apertus-8B-Instruct-2509"
    LAST=""
    base_pair olmo "$XAV/olmo_2_32b_meditronFO_gptoss_v1" allenai/OLMo-2-0325-32B-SFT "${CTX4K[@]}"
    LAST=""
    base_pair euro22 "$XAV/euro_llm_22b_meditronFO_gptoss_v1" utter-project/EuroLLM-22B-Instruct-2512
    LAST=""
    base_pair euro9 "$XAV/euro_llm_9b_meditronFO_gptoss_v1" utter-project/EuroLLM-9B-Instruct "${CTX4K[@]}"
    echo "Apertus-70B chain ends with job $A_LAST (pass AFTER=$A_LAST to the ablations70b mode)"
    ;;
*)
    sed -n '2,17p' "$0"; exit 1 ;;
esac
