cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
set -e
dl(){ hf download "$1" "$2" --local-dir "models/$3" >/dev/null 2>&1 && echo "OK  $2" || echo "FAIL $2"; }
dl unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf       q4b
dl unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf     q4b
dl unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf q4b
dl unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf       q9b
dl unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf     q9b
dl unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf q9b
echo "ALL_DONE"
