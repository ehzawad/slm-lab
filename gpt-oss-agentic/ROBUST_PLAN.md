# Robust gpt-oss-20b Agentic Build - Plan (pre-council)

Ambitious multi-track build on gpt-oss-20b, fixing the confounds the council caught in the first
run. Hardware: A5000 (GPU0, 24GB) + A6000 (GPU1, 48GB), both free; 90GB RAM. Everything on the
`gpt-oss-agentic` branch. NO emojis.

## What broke last time (must not repeat)
1. **Eval confound.** The before/after mixed serving paths: baseline 11/24 on grammar-constrained
   GGUF (llama.cpp --jinja), but the controlled base-vs-SFT arm floored BOTH to 0/24 on
   unconstrained transformers. A harness that floors the base cannot measure a delta.
2. **SFT overfit/degeneration.** LoRA all-linear+MoE r16 a32 lr 2e-4 on 520 trajectories drove
   teacher-forced loss to 0.0055; the adapter then regurgitated the tool-schema/developer block and
   emitted ~0 valid tool calls. Likely: no assistant-only loss masking + too-high LR + no early stop.
3. **GRPO no-op.** reward=0.0 all 12 steps; completions hit the 256-token cap and never terminated,
   so nothing could earn reward. A broken rollout, not an RL result.

## Design principles (non-negotiable)
- **Foundation first: a VALID same-path eval.** Serve base and trained on the SAME grammar-capable
  path and prove a SANITY GATE before any training conclusion: the base must score its known
  ~11/24 on that path (NOT floored). If the eval path floors the base, fix the path, do not train.
- **Every stage standalone-valuable and gated.** Env + baseline already banked; each new stage must
  produce a defensible artifact even if a later one fails.
- **Claim only what the telemetry supports** (reward variance > 0 before any RL claim; held-out
  generalization, not teacher-forced loss, as the SFT success metric).

## Track 1 - Valid eval/serving path (THE foundation; decide + sanity-gate)
Candidates, in preference order:
- (a) **vLLM + LoRA + guided decoding.** Best: same path for base (no adapter) vs adapter, grammar
  for guaranteed valid tool calls, and multi-LoRA hot-swap for the Granite-style adapters. RISK: we
  hit a vLLM 0.24.0 triton warmup crash. Mitigation: try a different vLLM version / the documented
  fix; this is worth real effort because it unlocks Tracks 3-5 too.
- (b) **Merge adapter -> GGUF -> llama.cpp --jinja + GBNF grammar.** Fallback. RISK: merging a QLoRA
  adapter into gpt-oss MoE+MXFP4 and re-converting is untested.
- (c) **transformers + PEFT + Outlines/lm-format-enforcer grammar.** Last resort; slow, but grammar
  prevents the flooring we saw (the flooring was from UNCONSTRAINED decoding).
Sanity gate: base gpt-oss-20b must reproduce ~11/24 on the chosen path.

## Track 2 - Robust training fix
- **SFT:** assistant-only loss masking (mask system/developer/tool-schema/user; train only on
  assistant + tool-call tokens) - the fix for schema regurgitation. LR 1e-5 to 5e-5 (not 2e-4),
  EARLY STOP on a held-out generation metric (valid-tool-call rate on held-out scenarios), correct
  harmony rendering IDENTICAL train==eval, held-out scenario split to detect overfit.
- **GRPO (only after SFT is solid):** terminating rollouts (max_tokens >= 1024, correct harmony stop
  tokens), reward SHAPING (partial credit for correct diagnostic steps, not just final solved), a
  verifiable proxy the policy can actually satisfy, or a true multi-turn tool-loop. Gate: only claim
  RL results if reward std > 0.

## Track 3 - Two Granite-style task adapters (over the frozen base)
- **Adapter A "tool-call reliability":** data with HARD NEGATIVES (malformed JSON args, wrong tool,
  hallucinated tool name, no-tool-needed), structured harmony tool-call targets.
- **Adapter B "incident-diagnosis":** structured root-cause + CALIBRATED CONFIDENCE + abstention
  ("insufficient evidence"), grounded on logs/metrics.
- Standard LoRA first (vLLM multi-LoRA hot-swap, `--enable-lora --max-loras 2`); aLoRA deferred
  (unproven on gpt-oss MoE+MXFP4 per the Granite research).

## Track 4 - Inference performance
- Bench llama.cpp vs vLLM on A6000: tokens/s + TTFT. Speculative decoding via the MTP head if
  supported. Reasoning-effort (low/med/high) latency/quality tradeoff. Both-GPU config (tensor
  parallel across A5000+A6000, or one-model-per-GPU concurrency).

## Track 5 - Harness reliability
- Grammar-constrained tool calling -> invalid-call rate to 0, measured on the 24-scenario hard env
  with the VALID path. BFCL-style schema-validity as a second axis.

## Open decisions for the council
1. Which eval path (Track 1) is the right foundation given the vLLM triton risk on A6000 - push
   vLLM, or go straight to merge->GGUF, or transformers+grammar?
2. Is assistant-only masking + LR reduction + early-stop the correct AND sufficient fix for the
   degeneration, or is there a deeper harmony train/eval mismatch to hunt?
3. Defer GRPO entirely until SFT+eval are proven, or include a fixed GRPO this round?
4. Scope: 2 adapters + training fix + inference + reliability in one dynamic workflow - too much?
   What is the critical path, what to gate/cut, and how to sequence for standalone value?
5. aLoRA - correct to defer, or worth prototyping now?
