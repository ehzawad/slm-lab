# 7-Stage Post-Training Pipeline — Proof of Flow (RTX A5000)

An OpenAI-style **staged post-training flow** run end-to-end on a single 24 GB RTX A5000,
as a *proof of plumbing* (tiny data, ~15–40 steps/stage — accuracy is irrelevant, the point
is that every stage lights up green and chains into the next). Base = **Qwen3-4B** (text,
standard transformers), chosen for reliability; **gpt-oss-20b is the intended swap for the
big-GPU / AWS run** (see caveat below).

Method: QLoRA 4-bit base (frozen) + **one LoRA adapter (all-linear, r=16) accumulated across
all 7 stages** — each stage resumes the previous stage's adapter. LoRA-on-all-linear + high LR
follows Thinking Machines' *"LoRA Without Regret."*

## Result — all 7 stages green

| # | Stage | Dataset | Trainer | Final loss |
|---|-------|---------|---------|-----------:|
| 1 | Continual pretrain (CPT-style) | `Salesforce/wikitext` (wt-2-raw) | SFT (raw LM) | 2.82 |
| 2 | Instruction SFT | `HuggingFaceH4/Multilingual-Thinking` | SFT | 1.07 |
| 3 | Reasoning (CoT) | `open-r1/OpenR1-Math-220k` | SFT | 0.51 |
| 4 | Tool-calling | `Salesforce/xlam-function-calling-60k` | SFT | 0.79 |
| 5 | MCP tool-use | synthesized (order-DB `<tool_call>` traces) | SFT | 0.32 |
| 6 | Preference | `HuggingFaceH4/ultrafeedback_binarized` | **DPO** | 0.73 |
| 7 | Verifiable RL | `openai/gsm8k` | **GRPO/RLVR** | −0.02 |

Runtime ~1–2 min/stage on the A5000, ~12 GB VRAM peak, 33 M trainable params (0.81%).

## Environment (frozen in `requirements-lock.txt`)

torch 2.12.1+cu130 · transformers 5.13.0 · trl 1.7.1 · peft 0.19.1 · accelerate 1.14.0 ·
datasets 5.0.0 · bitsandbytes 0.49.2 · CUDA driver 13.2 · verified `pip check` clean.

## Reconciliation with a Codex council (3/3 roles) — what we kept, what we overrode

The council reviewed the env stack, the single-adapter design, and recipe correctness.

- **Alleged bug — "DPO reference = naked base, not the accumulated policy."** The council
  (inspecting **trl 0.17**) said `ref_model=None` + PEFT disables the adapter so DPO regularizes
  toward base, and prescribed a two-adapter (`ref_adapter_name`) fix. **We re-verified in our exact
  trl 1.7 venv and OVERRODE this:** trl 1.7 has no `ref_adapter_name`; instead it *auto-creates a
  frozen `"ref"` adapter and copies the current policy's weights into it* (`dpo_trainer.py` L631–636),
  so the DPO reference IS the correct pre-DPO (post-MCP) policy. The bug was version-specific and is
  already fixed upstream. Their own fix would have errored here. **Lesson: verify in the exact env.**

Version-independent guidance we **accepted** (applied or documented):

- **Single-adapter accumulation maximizes interference** — DPO/GRPO can overwrite earlier
  tool/MCP/reasoning skills, and there's no clean ablation. Fine for proof-of-flow; for quality
  runs use **separate named adapters per objective family + explicit phase checkpoints**.
- **GRPO here is *math* RLVR, not "agentic" RL** — honest relabel. Real agentic-RL plugs a tool/
  environment loop; the natural next step is to use our `agentic-harness/` as the GRPO environment.
- **`num_generations=2` gives sparse/zero advantage** — green but weak learning signal.
- **CPT on wikitext is "CPT-style LM adaptation," not true domain CPT** — use real domain text.
- **`save_pretrained` carried a stray `ref/` adapter** — fixed via `selected_adapters=["default"]`.
- **gpt-oss-20b is NOT a one-line base swap.** Native MXFP4 MoE *training* is unsupported (inference-
  only, Hopper+); on Ampere it needs a **separate Unsloth env** (NF4 "mimicry", ~14 GB, torch≥2.8 +
  triton≥3.4). Plan a dedicated environment for the AWS run — do not reuse this Qwen stack unchanged.

## Scaling to AWS / bigger GPU

1. Swap base → gpt-oss-20b in a **separate Unsloth environment** (harmony format + MXFP4 handling).
2. Use **separate named adapters + phase checkpoints** instead of one growing adapter.
3. Replace toy slices with full datasets; raise steps, `num_generations`, and add eval gates.
4. Make stage 7 truly agentic by using the multi-turn tool harness as the RL environment.
