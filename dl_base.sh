cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3-4B --local-dir models/base/Qwen3-4B >/dev/null 2>&1 && echo BASE_DONE || echo BASE_FAIL
