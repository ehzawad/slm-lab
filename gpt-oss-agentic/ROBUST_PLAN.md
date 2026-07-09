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

## COUNCIL VERDICT - final de-risked scope (this is what the workflow builds)
Unanimous across 4 roles:
- **Eval foundation FIRST and it gates everything.** No training result is interpretable until the
  exact eval path reproduces a NON-FLOORED base baseline. Smoke-test this before any training.
- **Eval path ranking:** (1) recent-stable vLLM + LoRA + guided decoding (foundation, if it passes
  the A6000 smoke test - aligns eval with production serving + multi-LoRA); (2) transformers + PEFT
  + Outlines/lm-format-enforcer grammar (correct fallback, unmerged adapter, avoids MXFP4 merge);
  (3) merge -> GGUF + GBNF is OUT for training conclusions (MXFP4 merge/reconvert untested).
- **CUT GRPO this round** - defer until SFT + same-path eval show a measurable held-out gain, else
  another no-op. Then single-turn RLVR only; multi-turn later.
- **CUT MTP speculative decoding** (false premise: gpt-oss has no draft head). **DEFER aLoRA.**
- **Prove ONE adapter: Adapter A (tool-call reliability)** - directly validates harness + mask +
  guided decoding. Adapter B (incident-diagnosis) queued for a later workflow.
- **SFT: instrument the mask** (decode a batch; train_on_responses_only is a helper not proof) and
  judge success by FREE-RUNNING held-out generation (valid-tool-call rate), not teacher-forced loss.
- **Highest-risk assumption to smoke-test FIRST:** recent-stable vLLM runs gpt-oss-20b + LoRA +
  guided JSON on Ampere in the harmony path without crashing and reproduces a non-floored base.

### Final workflow shape (gated, standalone-valuable phases)
1. FOUNDATION (gate): stand up the eval path (vLLM primary, transformers+grammar fallback), prove a
   NON-FLOORED base baseline on the incident env via the SAME path adapters will use. Correct
   sampling (temp 1.0/top_p 1.0/top_k 0), constrain only JSON args, reasoning_effort medium. GATE.
2. DATA + SFT (Adapter A, tool-call reliability): harmony commentary-channel rendering + hard
   negatives + held-out split; assistant-only mask VERIFIED by decoding; 1 epoch; held-out
   valid-tool-call rate as the success metric.
3. VALID before/after: base vs Adapter A on the SAME path - solved/24 + executable-call rate +
   invalid-call rate (novel, unpublished for gpt-oss). Base must stay non-floored.
4. INFERENCE bench (parallel on A5000): llama.cpp vs vLLM tok/s + TTFT, prefix caching,
   one-replica-per-GPU, correct sampling. No MTP.
5. SYNTHESIZE + commit + push. Deferred/queued: GRPO, Adapter B, aLoRA (documented).

## Open decisions for the council (ANSWERED ABOVE)
1. Which eval path (Track 1) is the right foundation given the vLLM triton risk on A6000 - push
   vLLM, or go straight to merge->GGUF, or transformers+grammar?
2. Is assistant-only masking + LR reduction + early-stop the correct AND sufficient fix for the
   degeneration, or is there a deeper harmony train/eval mismatch to hunt?
3. Defer GRPO entirely until SFT+eval are proven, or include a fixed GRPO this round?
4. Scope: 2 adapters + training fix + inference + reliability in one dynamic workflow - too much?
   What is the critical path, what to gate/cut, and how to sequence for standalone value?
5. aLoRA - correct to defer, or worth prototyping now?
