cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/agentic-harness
. ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
until [ -f ../models/gemma4base/gemma-4-12B-it-Q4_K_M.gguf ] && grep -q GEMMABASE_DONE ../dl_gemma_base.log 2>/dev/null; do sleep 15; done
python3 - <<'PY'
src=open('agent_bench.py').read()
if 'gemma4base' not in src:
    src=src.replace('    ("gemma4-12B-agentic-v2 Q4_K_M", f"{REPO}/models/gemma4agentic/gemma4-v2-Q4_K_M.gguf"),\n]',
      '    ("gemma4-12B-agentic-v2 Q4_K_M", f"{REPO}/models/gemma4agentic/gemma4-v2-Q4_K_M.gguf"),\n'
      '    ("gemma-4-12B-it BASE Q4_K_M", f"{REPO}/models/gemma4base/gemma-4-12B-it-Q4_K_M.gguf"),\n]')
    open('agent_bench.py','w').write(src); print("added base gemma")
PY
python -u agent_bench.py 2>&1 | tee gemma_base_run.log | grep -E '=== gemma-4-12B-it BASE|PASS|FAIL|=>'
echo GEMMA_BASE_RUN_DONE
