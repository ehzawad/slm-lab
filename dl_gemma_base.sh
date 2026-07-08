cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
hf download ggml-org/gemma-4-12B-it-GGUF --include "*Q4_K_M*" --local-dir models/gemma4base >/dev/null 2>&1 && echo GEMMABASE_DONE || echo GEMMABASE_FAIL
ls -lh models/gemma4base/ 2>/dev/null | grep -iv mmproj | grep gguf
