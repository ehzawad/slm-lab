# Results — Sub-10B GGUF Head-to-Head (RTX A5000)

All numbers from a single **NVIDIA RTX A5000 (24 GB)** via `llama.cpp` (CUDA), GPU pinned
with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Metrics: `llama-bench` throughput (pp = prompt
processing @512, tg = token generation @128), loaded VRAM, a 10-item known-answer
reasoning probe, and an 8-item OpenAI-tools tool-call probe (`--jinja`).

## Qwen3.5 quantization study

| Config | Size | pp t/s | tg t/s | VRAM (MiB) | Reason | Tools |
|--------|-----:|-------:|-------:|-----------:|:------:|:-----:|
| 4B Q8_0      | 4.5 GB | 4877 | 112.3 | 5050 | 10/10 | 8/8 |
| 4B Q4_K_M    | 2.7 GB | 4489 | 139.7 | 3388 | 10/10 | 8/8 |
| 4B UD-Q4KXL  | 2.9 GB | 4464 | 138.5 | 3552 | 10/10 | 8/8 |
| 9B Q8_0      | 9.5 GB | 3388 |  69.4 | 8854 | 10/10 | 8/8 |
| 9B Q4_K_M    | 5.7 GB | 3160 |  96.1 | 5670 | 10/10 | 8/8 |
| 9B UD-Q4KXL  | 6.0 GB | 3172 |  93.5 | 5942 | 10/10 | 8/8 |

## Cross-family (Q8_0 + Q4_K_M)

| Model | Size | pp t/s | tg t/s | VRAM (MiB) | Reason | Tools |
|-------|-----:|-------:|-------:|-----------:|:------:|:-----:|
| Gemma-3n-E2B Q8_0   | 4.8 GB | 4944 | 130.7 | 3080 | 10/10 | N/A |
| Gemma-3n-E2B Q4_K_M | 3.0 GB | 4756 | 160.8 | 2000 |  9/10 | N/A |
| Gemma-3n-E4B Q8_0   | 7.4 GB | 3471 |  86.5 | 5240 | 10/10 | N/A |
| Gemma-3n-E4B Q4_K_M | 4.5 GB | 3266 | 111.4 | 3256 | 10/10 | N/A |
| DeepSeek-R1-7B Q8_0 | 8.1 GB | 4308 |  82.9 | 8006 | 10/10 | N/A |
| DeepSeek-R1-7B Q4_K_M | 4.7 GB | 4053 | 115.7 | 5010 | 10/10 | N/A |
| gpt-oss-20b Q4_K_M  | 11.6 GB | 2658 | 177.1 | 11260 | 10/10 | 6/8 |

## Findings

1. **Quantization was quality-free on these probes.** Qwen3.5 held 10/10 reasoning and
   8/8 tool-calls at every quant, while Q4_K_M ran ~25% faster (tg) and used ~40% less
   VRAM than Q8_0. **imatrix (UD-Q4KXL) showed no measurable benefit** over plain Q4_K_M
   here — expected, since the probes never stressed the low-frequency weights imatrix protects.
2. **Agentic tool-calling is where models separate.** Qwen3.5 = 8/8 across all quants;
   gpt-oss-20b = 6/8. Gemma-3n and DeepSeek-R1 returned **0/8 → reported N/A, not a
   capability failure**: neither exposes structured tool-calls through llama.cpp's OpenAI
   `--jinja` path (DeepSeek-R1 is reasoning-only by design; Gemma-3n needs manual
   prompt-format handling to emit tool calls).
3. **gpt-oss-20b (MoE, ~3.6B active/token) was the fastest generator** (177 t/s) despite
   the largest disk/VRAM footprint — sparse activation beats dense param count for latency.
4. **Smallest viable footprint:** Gemma-3n-E2B Q4_K_M at **2.0 GB** VRAM, 160 t/s.

## Honest limitations

- **The reasoning probe saturated** (nearly all 10/10) — GSM8K-level items are too easy to
  separate these models. Real discrimination needs AIME/GPQA-level problems.
- **Tool-call probe is single-turn only.** It measures "emit one valid tool call," not
  **full agentic-harness capability** (multi-step planning, MCP tool use, error recovery,
  long-horizon tool loops). That is the metric that actually matters for agents and is the
  subject of the follow-up evaluation (see repo root TODO).
- N=8 tool prompts / N=10 reasoning items — small samples; treat as directional.
