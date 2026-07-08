cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv-vllm/bin/activate
pip install -q "triton==3.5.0" 2>&1 | tail -2
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
vllm serve models/qwythos_src --served-model-name qwythos --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 18420
