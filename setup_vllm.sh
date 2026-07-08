cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench
python3 -m venv .venv-vllm
. .venv-vllm/bin/activate
python -m pip install -q --upgrade pip
pip install -q vllm 2>&1 | tail -5
python -c "import vllm; print('VLLM', vllm.__version__)" 2>&1 | tail -2
echo VLLM_SETUP_DONE
