#!/bin/bash
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/nimbus-agent
. ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false TRL_EXPERIMENTAL_SILENCE=1
set -e
python -u eval_nimbus.py v3_sft adapters_v3/sft
for stage in reasoning tools mcp dpo grpo; do
  echo "########## V3 STAGE: $stage ##########"
  python -u train_nimbus_v3.py "$stage"
  python -u eval_nimbus.py "v3_$stage" "adapters_v3/$stage"
done
echo "########## GKD/OPD ##########"
python -u train_gkd.py grpo
python -u eval_nimbus.py v3_opd adapters_v3/opd
echo "########## V3 + OPD COMPLETE ##########"
