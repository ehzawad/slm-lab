#!/bin/bash
# Sequential: (1) v4 tools-dip fix chain with per-stage evals; (2) OPD sweep A/C/D from
# the SAME v3_grpo start as the completed variant B, so all four points are comparable.
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/nimbus-agent
. ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false TRL_EXPERIMENTAL_SILENCE=1
set -e

echo "########## PART 1: v4 tools-dip fix ##########"
for stage in cpt sft reasoning toolsmix dpo grpo; do
  echo "===== V4 STAGE: $stage ====="
  python -u train_nimbus_v4.py "$stage"
done
for stage in toolsmix dpo grpo; do
  python -u eval_nimbus.py "v4_$stage" "adapters_v4/$stage"
done

echo "########## PART 2: OPD sweep (from adapters_v3/grpo, like variant B) ##########"
run_opd () {  # name lmbda beta steps lr
  echo "===== OPD VARIANT: $1 (lmbda=$2 beta=$3 steps=$4 lr=$5) ====="
  OPD_NAME="$1" OPD_LMBDA="$2" OPD_BETA="$3" OPD_STEPS="$4" OPD_LR="$5" \
    python -u train_gkd.py grpo
  python -u eval_nimbus.py "v3_$1" "adapters_v3/$1"
}
run_opd opd_A 0.3 0.5 60 5e-6
run_opd opd_C 0.5 0.3 60 5e-6
run_opd opd_D 0.5 0.5 120 1e-5

echo "########## V4 + OPD SWEEP COMPLETE ##########"
