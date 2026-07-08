# SLM Lab - Journey, Findings, and What to Learn Next

This repo grew from a single question ("what is an LLM?") into a working laboratory for building,
benchmarking, and post-training small/on-device agentic language models on one box (RTX A5000
24GB = GPU 0, RTX A6000 48GB = GPU 1). This document is the capstone: the arc we took, what each
experiment found, the cross-cutting lessons, and a prioritized roadmap for what to learn next.

Per-experiment detail lives in the sub-project docs linked below; this is the map, not the territory.

## The experiments (in order), with the one-line finding

1. **GGUF benchmark** - [sub10b-gguf-bench/RESULTS.md](sub10b-gguf-bench/RESULTS.md).
   Throughput/quality of sub-10B GGUF quants (Qwen3.5-4B/9B, gpt-oss-20b) in llama.cpp.
   Finding: Q4_K_M is the practical sweet spot; gpt-oss-20b is fastest despite size (MoE, ~3.6B active).

2. **Multi-turn agentic harness (MCP)** - [agentic-harness/AGENTIC_RESULTS.md](agentic-harness/AGENTIC_RESULTS.md).
   8 tasks (sequential mutation, multi-hop, cross-turn memory, parallel compare, error recovery,
   abstention) over a real MCP server + local tools, end-to-end scored.
   Finding: capable models SATURATE at 8/8 - a benchmark that stops discriminating is telling you
   to build a harder one.

3. **8-stage post-training pipeline** - [pipeline/PIPELINE.md](pipeline/PIPELINE.md).
   CPT -> SFT -> reasoning -> tools -> MCP -> DPO -> GRPO -> agentic-GRPO, end to end.
   Finding: the plumbing works; each stage needs its own guardrail (below).

4. **NL-to-SQL quality study** - [sql-agent/SQL_RESULTS.md](sql-agent/SQL_RESULTS.md).
   A verifiable domain with an execution reward.
   Finding: REWARD HACKING is real - a naive comparator let `SELECT 1` score 12-15%. Fixed with a
   type-aware comparator + an anti-degenerate guard (constant/no-FROM answers score 0).

5. **NimbusWorks all-7-stages scenario** - [nimbus-agent/NIMBUS_RESULTS.md](nimbus-agent/NIMBUS_RESULTS.md).
   A fictional-company world where every training stage earns its place.
   Finding: naive multi-stage training causes CATASTROPHIC FORMAT COLLAPSE (23.7); REPLAY MIXING
   prior-stage data restores it (81.6). Interference is measurable and fixable.

6. **Masking + on-policy distillation** - [nimbus-agent/MASKING_OPD_RESULTS.md](nimbus-agent/MASKING_OPD_RESULTS.md).
   Finding: assistant-only loss (`{% generation %}` markers, which Qwen3 does not ship - must patch)
   is a modest, real win; OPD is a safe wash when the teacher prompts are behavior-only and
   pre-filtered (prevents domain erasure).

7. **Community model tests + base-vs-finetune** - [agentic-harness/COMMUNITY_MODEL_FINDINGS.md](agentic-harness/COMMUNITY_MODEL_FINDINGS.md).
   gemma4-12B-agentic-v2 and Qwythos-9B, each measured against its BASE.
   Finding: neither fine-tune beats its base on general agentic capability on a fair harness;
   Qwythos makes a small real reasoning/tool-cleanliness gain (far short of its advertised +30
   gsm8k), and gemma4-agentic trades agentic reliability for a domain (telecom) we did not test.

8. **gpt-oss-20b deep-dive + harder agentic env + training stack** - gpt-oss-agentic/ (in progress
   on the `gpt-oss-agentic` branch). Building an unsaturated executable incident-response
   simulator, baselining gpt-oss-20b on it, then CPT->SFT->agentic-GRPO via Unsloth.

## Model verdicts (measured, not marketing)

| Model | Agentic (8-task) | Reasoning (GSM8K n=40) | Notes |
|-------|:---:|:---:|-------|
| gpt-oss-20b            | 8/8 | - | Reference local agent: 0 parser fallbacks, fewest tool calls, 177 t/s |
| Qwen3.5-9B             | 8/8 | 90.0% | Solid base; needed the XML shim 5x for tool parsing |
| Qwythos-9B (--no-mtp)  | 8/8 | 95.0% | Cleaner tools (0 fallbacks), +5 reasoning; needs temp 0.6 |
| Qwen3.5-4B            | 8/8 | - | Punches above its size |
| gemma-4-12B-it (base)  | 8/8 | 42.5% | Clean base |
| gemma4-12B-agentic-v2  | 7/8 | 95.0% | Reasons well, flails on tools out-of-domain |

Pick: **gpt-oss-20b** for a general local agent (VRAM to spare); **Qwythos-9B** for a compact 9B
with the cleanest tool-calling; **gemma4-agentic** only for its terminal/telecom niche, tested on
your own tasks.

## Cross-cutting lessons (the transferable part)

Technical:
- **Benchmarks saturate. Design for headroom.** An 8/8 ceiling cannot show improvement OR
  regression. Add a discriminating axis (we added a GSM8K probe) and, better, harder tasks.
- **Every verifiable reward invites hacking.** Type-aware comparison + anti-degenerate guards, or
  the model games the metric (SELECT 1).
- **Multi-stage training interferes.** Replay-mix prior-stage data or later stages erase earlier
  formats/skills.
- **The most common "tool error" is a serving-parser mismatch, not the model.** The model emits a
  valid call; the server fails to structure it. Fix by matching the parser (gpt-oss+harmony native,
  Qwen via vLLM `--tool-call-parser`, or GBNF grammar) - a shim is a workaround, not a fix.
- **Base-vs-finetune needs the base run.** A fine-tune's score is meaningless without its base, and
  can win one axis (reasoning) while losing another (tool reliability). Report both.
- **Architecture/quant gotchas bite at convert time.** Qwen3.5 hybrid Gated-DeltaNet + an MTP head
  produced an unloadable GGUF until `--no-mtp`; gpt-oss MoE+MXFP4 cannot be trained natively on
  Ampere (Unsloth's linearized path only).
- **LoRA that works (Thinking Machines):** all-linear incl MoE, alpha=32, ~10x the full-FT LR,
  small batches; RL matches full-FT even at low rank.

Operational (hard-won):
- `pkill -f <pattern>` where the pattern matches your own shell command SIGTERMs your own wrapper
  (exit 144). Manage servers by subprocess handle/SIGINT, or use non-self-matching patterns.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` - llama.cpp otherwise reverses device order (lists A6000 as 0).
- Background long GPU jobs with nohup + a log + an until-grep poll; a single foreground call caps
  around 10 minutes. Never double-download the same HF file (lock deadlock).
- Isolate risky installs (vLLM, Unsloth) in their own venv so they cannot break the training stack.

## What to learn / do next (prioritized)

1. **Harder, domain-matched, statistically-powered evals.** Stand up BFCL v3/v4 and tau2-bench
   (retail AND telecom) instead of home-grown 8-task suites; report schema-validity and executable-
   call rate, multiple seeds, and confidence intervals. This is the single highest-ROI next step -
   almost every conclusion above is currently N=small, single-run.
2. **Finish the gpt-oss-20b training loop end to end.** Get Unsloth QLoRA + agentic-GRPO working on
   the A6000, and prove a before/after delta on the unsaturated incident-response env. This closes
   the loop from "can we serve it" to "can we improve it."
3. **Grammar-constrained tool calling (GBNF/XGrammar/Outlines).** A model-agnostic hard guarantee
   that every tool call is schema-valid - removes the parser-mismatch class of failures entirely.
4. **On-policy distillation / GKD at slightly larger scale.** OPD was a wash at toy scale; test
   whether a strong teacher (or Thinking Machines' Tinker API) moves the needle with more data.
5. **Agentic RL with real multi-turn rollouts.** Our GRPO used verifiable single-step proxies;
   true multi-turn tool-loop rollouts with execution reward are the frontier for agentic capacity.
6. **Serving maturity.** A working vLLM setup (the triton 3.6/0.24 warmup bug blocked us),
   speculative decoding (the MTP head we stripped is exactly for this), and latency/throughput SLOs.
7. **Security for tool-using agents.** Prompt-injection defenses, tool sandboxing, permission/
   validator-first execution - untested here and mandatory before any real deployment.
8. **Data scale and quality.** Everything used small, synthetic datasets. Real gains need larger,
   deduplicated, quality-filtered corpora and more seeds for statistical power.

## Repo layout

- `sub10b-gguf-bench/` GGUF throughput/quality. `agentic-harness/` MCP multi-turn agent eval +
  community-model findings. `pipeline/` the 8-stage post-training pipeline. `sql-agent/` NL-to-SQL
  with hardened verifiable reward. `nimbus-agent/` NimbusWorks 7-stage + masking/OPD.
  `gpt-oss-agentic/` (branch) gpt-oss deep-dive + harder env + training. `llama.cpp/` the serving/
  convert/quantize toolchain. `models/` local GGUF weights (gitignored).
