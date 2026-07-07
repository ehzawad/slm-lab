# slm-lab

A lab for experiments with small / on-device language models (sub-10B, GGUF, agentic, fine-tuning).
Everything runs on a single NVIDIA RTX A5000 (24 GB), GPU 0 only.

## Experiments

- [**sub10b-gguf-bench/**](sub10b-gguf-bench/) - head-to-head GGUF benchmark of sub-10B models
  (Qwen3.5-4B/9B, Gemma-3n-E2B/E4B, DeepSeek-R1-Distill-7B, gpt-oss-20b) on an RTX A5000:
  speed, VRAM, reasoning, and single-turn tool-call validity across quant levels. See its
  [RESULTS.md](sub10b-gguf-bench/RESULTS.md).
- [**agentic-harness/**](agentic-harness/) - the full agentic eval: real multi-turn tool
  loops with executed tools + result feedback, a minimal MCP server (JSON-RPC stdio, stateful
  DB), state/memory, parallel calls, and tool-error recovery, scored end-to-end. Overturns
  the single-turn conclusion (gpt-oss-20b 6/8 to 8/8; Qwen "failures" were a llama.cpp parser gap).
  See [AGENTIC_RESULTS.md](agentic-harness/AGENTIC_RESULTS.md).
- [**pipeline/**](pipeline/) - an OpenAI-style 8-stage post-training flow run end-to-end on the
  A5000 (CPT, SFT, reasoning, tool-calling, MCP, DPO, GRPO, agentic-GRPO), one accumulating QLoRA
  adapter, real dataset per stage. Proof-of-plumbing (all 8 green in one command), reconciled with a
  Codex council - including overriding a version-specific "DPO reference" bug after verifying trl 1.7
  handles it correctly. See [PIPELINE.md](pipeline/PIPELINE.md).
- [**sql-agent/**](sql-agent/) - a quality-focused, execution-verified natural-language-to-SQL agent.
  Qwen3-4B QLoRA on gretel synthetic data, scored by execution accuracy (run the SQL, compare result
  sets). Web-searched SOTA + council-reconciled before RL. Result: base 47.3% to SFT 58.0% execution
  accuracy (real +10.7); GRPO held; CoT-SFT hurt at 4B. See [SQL_RESULTS.md](sql-agent/SQL_RESULTS.md).

## Key takeaways

- **Agentic serving (24 GB):** gpt-oss-20b is the most robust and efficient out-of-the-box in
  llama.cpp; Qwen3.5-9B matches it on capability (and leads official BFCL/TAU2) but needs the correct
  tool-call parser (`qwen3_coder` / a shim) or ~1/3 of its multi-turn tool calls silently drop.
- **Fine-tuning quality (NL-to-SQL):** direct-SQL SFT is the real win (+10.7 pts, execution-verified);
  a conservative execution-reward GRPO held accuracy safely; CoT-SFT hurt at 4B. Not every SOTA trick
  transfers to a small model, and the reward/metric must be hardened first (a probe found a 12-15%
  degenerate reward hack, fixed to 0%).
- **Process:** measure honestly (hardened, non-gameable metrics), reconcile the risky steps with a
  Codex council, and verify claims in the exact environment rather than trusting version-specific advice.

## Roadmap

- Cross-distribution NL-to-SQL eval on Spider/BIRD dev to make the accuracy numbers production-credible.
- Stronger GRPO (variance-selected prompts, more steps) and a larger base (Qwen3.5-9B / gpt-oss-20b).
- Scale the agentic suite to BFCL-v3/v4, tau-bench/tau2, GAIA (harder, statistically powered).
- Serve Qwen3.5 via vLLM with the native tool parser to remove the llama.cpp confound.

## Notes

- Models default to GPU 0 = RTX A5000 (`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`).
- Environment is version-locked in `pipeline/requirements-lock.txt`.
- Large weights, venvs, trained adapters, and the llama.cpp clone are gitignored - see each
  experiment's README to reproduce.
