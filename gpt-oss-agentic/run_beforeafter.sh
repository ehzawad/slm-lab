#!/usr/bin/env bash
set -uo pipefail
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/gpt-oss-agentic
PY=../.venv-vllm/bin/python
ADAPTER=/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/gpt-oss-agentic/adapters_gptoss/adapterA
TRAIN_PID=2951466
LOG=beforeafter_run.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== waiter start $(date -u +%FT%TZ) ===" | tee "$LOG"

# 1) wait for training process to exit (frees GPU 1) AND adapter to be saved
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  echo "[wait] train pid $TRAIN_PID still alive $(date -u +%TZ)" | tee -a "$LOG"
  sleep 30
done
echo "[wait] train pid gone $(date -u +%FT%TZ)" | tee -a "$LOG"

# wait for adapter files to land (save happens near end)
for i in $(seq 1 60); do
  if [ -f "$ADAPTER/adapter_config.json" ] && [ -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "[wait] adapter present" | tee -a "$LOG"; break
  fi
  echo "[wait] adapter not yet present ($i)" | tee -a "$LOG"; sleep 10
done

# ensure GPU 1 is actually free of compute procs before booting vLLM
for i in $(seq 1 60); do
  BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader | grep 4754ca84 | awk -F, '{gsub(/[^0-9]/,"",$2); if($2+0>2000) print}' | wc -l)
  if [ "$BUSY" -eq 0 ]; then echo "[wait] GPU1 free" | tee -a "$LOG"; break; fi
  echo "[wait] GPU1 still busy ($i)" | tee -a "$LOG"; sleep 15
done

echo "=== BASE eval (same-path) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=1 $PY eval_reliability_same_path.py \
  --label "gpt-oss-20b base (adapterA same-path)" --port 18490 >>"$LOG" 2>&1
echo "BASE_EXIT=$?" | tee -a "$LOG"

sleep 20  # let GPU fully release between servers

echo "=== ADAPTER A eval (same-path) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=1 $PY eval_reliability_same_path.py \
  --label "gpt-oss-20b + Adapter A (adapterA same-path)" --adapter "$ADAPTER" --port 18491 >>"$LOG" 2>&1
echo "ADAPTER_EXIT=$?" | tee -a "$LOG"

echo "=== DONE $(date -u +%FT%TZ) ===" | tee -a "$LOG"
