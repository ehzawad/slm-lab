# slm-lab

A lab for experiments with small / on-device language models (sub-10B, GGUF, agentic).

## Experiments

- [**sub10b-gguf-bench/**](sub10b-gguf-bench/) — head-to-head GGUF benchmark of sub-10B models
  (Qwen3.5-4B/9B, Gemma-3n-E2B/E4B, DeepSeek-R1-Distill-7B, gpt-oss-20b) on an RTX A5000:
  speed, VRAM, reasoning, and tool-call validity across quant levels. See its
  [RESULTS.md](sub10b-gguf-bench/RESULTS.md).

## Roadmap

- **Full agentic-harness evaluation** — the current tool-call probe is single-turn. The next
  experiment measures *full agentic capability* (which subsumes tool-calling, reasoning, MCP
  tool use, and RL-trained behaviors): multi-step tool loops, MCP server integration, planning,
  and error recovery — benchmarked with BFCL-v3 / τ-bench / GAIA-style tasks.

## Notes

- Models default to GPU 0 = **RTX A5000** (`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`).
- Large weights, venvs, and the llama.cpp clone are gitignored — see each experiment's README to reproduce.
