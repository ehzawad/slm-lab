#!/bin/bash
cd /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp
export CUDACXX=/usr/local/cuda-13.2/bin/nvcc PATH=/usr/local/cuda-13.2/bin:$PATH
set -e
git checkout -q origin/master
echo "=== on $(git log -1 --format='%h %cd' --date=short) ==="
cmake --build build --config Release -j "$(nproc)" 2>&1 | tail -3
echo "LLAMA_REBUILD_DONE"
