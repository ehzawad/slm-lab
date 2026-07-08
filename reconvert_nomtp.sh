cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
set -e
echo "=== reconvert with --no-mtp ==="
python llama.cpp/convert_hf_to_gguf.py models/qwythos_src --no-mtp \
  --outfile models/qwythos/qwythos-nomtp-f16.gguf --outtype f16 2>&1 | tail -2
echo "=== quantize ==="
./llama.cpp/build/bin/llama-quantize models/qwythos/qwythos-nomtp-f16.gguf \
  models/qwythos/qwythos-nomtp-Q4_K_M.gguf Q4_K_M 2>&1 | tail -1
echo "NOMTP_BUILD_DONE"
