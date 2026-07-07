# slm-lab

A lab for experiments with small / on-device language models (sub-10B, GGUF, agentic).

## Experiments

- [**sub10b-gguf-bench/**](sub10b-gguf-bench/) — head-to-head GGUF benchmark of sub-10B models
  (Qwen3.5-4B/9B, Gemma-3n-E2B/E4B, DeepSeek-R1-Distill-7B, gpt-oss-20b) on an RTX A5000:
  speed, VRAM, reasoning, and single-turn tool-call validity across quant levels. See its
  [RESULTS.md](sub10b-gguf-bench/RESULTS.md).
- [**agentic-harness/**](agentic-harness/) — the **full agentic** eval: real multi-turn tool
  loops with executed tools + result feedback, a minimal **MCP server** (JSON-RPC stdio, stateful
  DB), state/memory, parallel calls, and tool-error recovery — scored **end-to-end**. Overturns
  the single-turn conclusion (gpt-oss-20b 6/8→8/8; Qwen "failures" were a llama.cpp parser gap).
  See [AGENTIC_RESULTS.md](agentic-harness/AGENTIC_RESULTS.md).
- [**pipeline/**](pipeline/) — an OpenAI-style **7-stage post-training flow** run end-to-end on the
  A5000 (CPT → SFT → reasoning → tool-calling → MCP → DPO → GRPO), one accumulating QLoRA adapter,
  real dataset per stage. Proof-of-plumbing (all 7 green), reconciled with a Codex council — incl.
  overriding a version-specific "DPO reference" bug after verifying trl 1.7 handles it correctly.
  See [PIPELINE.md](pipeline/PIPELINE.md).

## Key takeaway

For a full agentic harness on 24GB: **gpt-oss-20b** is the most robust + efficient out-of-the-box
in llama.cpp; **Qwen3.5-9B** matches it on capability (and leads official BFCL/TAU2) but needs the
correct tool-call parser (`qwen3_coder` / a shim) or ~1/3 of its multi-turn tool calls silently drop.

## Roadmap

- Scale the agentic suite to BFCL-v3/v4, τ-bench/τ³, GAIA (harder, statistically powered).
- Add a quant × agentic-reliability ladder (Q8/Q6/Q4/imatrix) with schema-validity + executable-call rate.
- Serve Qwen3.5 via vLLM with the native tool parser to remove the llama.cpp confound.

## Notes

- Models default to GPU 0 = **RTX A5000** (`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`).
- Large weights, venvs, and the llama.cpp clone are gitignored — see each experiment's README to reproduce.
