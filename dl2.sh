cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
dl(){ hf download "$1" "$2" --local-dir "models/$3" >/dev/null 2>&1 && echo "OK  $2" || echo "FAIL $2"; }
dl unsloth/gemma-3n-E2B-it-GGUF gemma-3n-E2B-it-Q4_K_M.gguf gemma_e2b
dl unsloth/gemma-3n-E2B-it-GGUF gemma-3n-E2B-it-Q8_0.gguf   gemma_e2b
dl unsloth/gemma-3n-E4B-it-GGUF gemma-3n-E4B-it-Q4_K_M.gguf gemma_e4b
dl unsloth/gemma-3n-E4B-it-GGUF gemma-3n-E4B-it-Q8_0.gguf   gemma_e4b
dl unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf deepseek7b
dl unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf   deepseek7b
dl unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf gptoss20b
echo "ALL_DONE2"
