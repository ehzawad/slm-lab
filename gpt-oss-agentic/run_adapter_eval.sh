#!/bin/bash
set -x
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/gpt-oss-agentic
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID DISABLE_ADDMM_CUDA_LT=1
PY=/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/.venv-unsloth/bin/python
$PY eval_incident_adapter.py --adapter none --label "gpt-oss-20b base (transformers)" --temp 0.0
echo "=====BASE_DONE====="
$PY eval_incident_adapter.py --adapter adapters_gptoss/sft --label "gpt-oss-20b SFT (transformers)" --temp 0.0
echo "=====SFT_DONE====="
