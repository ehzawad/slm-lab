#!/bin/bash
# NimbusWorks: train each stage then measure ALL four metrics at that checkpoint.
# Produces scores.json = the stage-by-stage capability curve. Resumable.
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/nimbus-agent
. ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
set -e
for stage in cpt sft reasoning tools mcp dpo grpo; do
  echo "########## STAGE: $stage ##########"
  python -u train_nimbus.py "$stage"
  python -u eval_nimbus.py "$stage" "adapters/$stage"
done
echo "########## ALL STAGES + EVALS COMPLETE ##########"
python -c "import json; print(json.dumps(json.load(open('scores.json')), indent=2))"
