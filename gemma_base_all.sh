cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
hf download ggml-org/gemma-4-12B-it-GGUF gemma-4-12B-it-Q4_K_M.gguf --local-dir models/gemma4base
cd agentic-harness
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
python -u agent_bench.py 2>&1 | tee gemma_base_run.log | grep -E '=== gemma-4-12B-it BASE|PASS|FAIL|=>'
echo GEMMA_BASE_RUN_DONE
