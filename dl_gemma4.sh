cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
hf download yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF \
  --include "*Q4_K_M*" --exclude "MTP/*" --local-dir models/gemma4agentic >/dev/null 2>&1 \
  && echo GEMMA4_DONE || echo GEMMA4_FAIL
ls -lh models/gemma4agentic/ 2>/dev/null
