#!/bin/bash
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
set -e
echo "=== 1. download full repo ==="
hf download empero-ai/Qwythos-9B-Claude-Mythos-5-1M --local-dir models/qwythos_src
echo "config.json present:"; ls -la models/qwythos_src/config.json
echo "safetensors:"; ls -lh models/qwythos_src/*.safetensors
echo "=== 2. convert HF -> GGUF f16 ==="
python llama.cpp/convert_hf_to_gguf.py models/qwythos_src \
  --outfile models/qwythos/qwythos-f16.gguf --outtype f16
echo "=== 3. quantize -> Q4_K_M ==="
./llama.cpp/build/bin/llama-quantize \
  models/qwythos/qwythos-f16.gguf models/qwythos/qwythos-Q4_K_M.gguf Q4_K_M
echo "QWYTHOS_BUILD_DONE"; ls -lh models/qwythos/
