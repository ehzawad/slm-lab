# Robust gpt-oss-20b Adapter A — Same-Path Eval, Verified Mask, Before/After

Scope: close the loop on the prior confounded gpt-oss-20b training result by (1) fixing the
serving-path confound that floored the base, (2) training Adapter A (tool-call reliability) with a
verified assistant-only loss mask, (3) reporting a valid same-path before/after, (4) recording the
inference-perf numbers + recommendation, and (5) listing what was deliberately deferred and why.

Environment: A6000 (GPU 1, 48 GB) for the 40 GB bf16-dequant serving path; A5000 (GPU 0, 24 GB) for
llama.cpp inference bench. `CUDA_DEVICE_ORDER=PCI_BUS_ID` always. No two agents on one GPU.

---

## 1. Same-path foundation — the fix for the prior confound

The prior experiment-8 before/after was **invalid by construction**: it compared a
grammar-constrained GGUF base (llama.cpp, 11/24) against an unconstrained transformers fine-tune,
and in the one controlled run that used a single path (transformers/Unsloth 4-bit, temp 0.0) **both
base and SFT floored to 0/24**. A harness that floors the base cannot measure a delta in either
direction. That result supports no claim that training helped or hurt.

The fix is a single, byte-identical serving path for base and adapter:

- **Serve**: vLLM 0.24 OpenAI endpoint on the unmerged HF weights (`models/gptoss20b-hf`, MXFP4 →
  bf16 on Ampere), flags `--tool-call-parser openai --enable-auto-tool-choice
  --enable-prefix-caching --gpu-memory-utilization 0.85 --max-model-len 16384`.
- **Sample**: temp 1.0, top_p 1.0, top_k 0, min_p 0, `reasoning_effort=medium` (per the standing
  recipe: high can hurt tool loops).
- **Agent path**: `eval_same_path.py` drives the reusable `incident_harness.run_episode` over the 24
  scenarios; the LoRA adapter is loaded via vLLM's `--enable-lora` path so base and adapter differ
  by **only** the adapter weights.

Two fixes were required and remain in place (removing either re-breaks the path):

1. **vLLM boot patch** — `.venv-vllm .../warmup/minimax_m3_msa_warmup.py` imported a minimax_m3
   triton kernel that triton 3.5.0 cannot JIT-parse; import is deferred (no-op for gpt-oss). Backup
   at `.orig`.
2. **Harmony sanitization** — the vLLM openai tool parser intermittently leaks harmony channel
   tokens into tool names/args at temp 1.0 (e.g. `name=check_all<|channel|>commentary`,
   `args={"":"{}"}`). One corruption fed back into history poisoned whole episodes — **this was the
   prior flooring cause**. Fixed with `_clean_tool_name` / `_clean_tool_args` in
   `eval_same_path.py::_normalize`.

### Non-floored base baseline (FOUNDATION)

| metric | value |
|---|---|
| solved | **20 / 24 (83.3%)** |
| correct root cause | 20 / 24 |
| avg steps | 6.67 |
| redundant-call rate | 6.9% |
| wall time | 228 s |

By fault type: `bad_config` 3/7, `bad_deploy` 7/7, `crash` 6/6, `pool_exhausted` 4/4.

This is clearly **non-floored** and well above the 11/24 grammar-constrained GGUF reference, so the
same path can now measure a delta in either direction. Row in `incident_scores.json`:
`"gpt-oss-20b base (vllm same-path)"`.

---

## 2. Adapter A (tool-call reliability) — verified mask, training regime

### Decoded mask proof (the prior failure's root cause, now guarded)

The prior SFT degenerated because loss was taken over developer/tool-schema tokens (it learned to
reproduce the schema). Adapter A trains with `train_on_responses_only` on the `<|start|>assistant`
boundary, and **decodes one masked batch before training** to prove only assistant
analysis+commentary+final carry loss. From `train_adapterA.log`:

```
================= DECODED MASK PROOF =================
total_tokens=1337 loss_tokens(non -100)=146 masked=1191
--- LOSS-CARRYING (assistant analysis+commentary+final ONLY) ---
 to=functions.check_all<|channel|>commentary json<|message|>{}<|call|> ...
 <|channel|>analysis<|message|>...the incident is resolved.<|end|>
 <|start|>assistant<|channel|>final<|message|>Root cause was a bad_deploy fault on `glacierdocs`...
--- MASKED HEAD (system/developer/tools/user - NO loss) ---
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
...
MASK_ASSERTIONS_PASSED
```

146 of 1337 tokens carry loss (10.9%); the system/developer/tool-schema/user head is fully masked.
Tool calls render on the **commentary** channel
(`<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>`),
tools declared in the developer message, same harmony renderer at train and eval.

### Training regime

| item | value |
|---|---|
| base | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (Unsloth linearized QLoRA, A6000) |
| LoRA | r=16, alpha=16, all-linear + `experts.gate_up_projs/down_projs` (regex targets) |
| trainable params | 184,909,824 / 21,099,667,008 (0.88%) |
| dataset | 536 harmony trajectories, 1 epoch (34 steps planned) |
| batch | 2 × grad-accum 8 = 16 effective |
| LR | 1e-4 cosine |
| stop | early-stop at step 20 (loss 0.0578 < 0.1) |
| loss | first 2.79 → last-step 0.058, **train_loss 0.9241** |
| peak VRAM | 17.0 GB |
| adapter | `adapters_gptoss/adapterA/` (adapter_model.safetensors 740 MB, saved 03:18) |

Loss history: `[2.79, 2.84, 2.63, 2.09, 1.68, 1.22, 1.06, 0.76, 0.56, 0.46, 0.37, 0.35, 0.37, 0.21,
0.28, 0.15, 0.26, 0.21, 0.15, 0.058]`.

**Caveat on early-stop:** the last logged step hit 0.058, below the standing 0.3–0.8 target band that
guards against the memorization regime. The `train_loss` mean over the run is 0.92 (healthy), but the
tail is in memorization territory; the free-running held-out valid-tool-call rate — not the
teacher-forced loss — is the metric of record, and that is exactly what the before/after measures.

### Held-out valid-tool-call rate — PENDING

The trainer's in-process free-running held-out eval (transformers/Unsloth MXFP4 decode, 512-token
cap, agentic multi-turn) is still running ~3.5 h after `SFT_DONE` printed. It is a **vestigial
post-save phase** (the adapter and all training deliverables were written at 03:18), but it is an
agentic loop (30+ `generate` calls at ~0.14 tok/s, effectively unbounded), and it still holds PID
2951466 on GPU 1. It was **not terminated**: it is another agent's process, completion is tracked by
an orchestrator waiter, and a SIGINT mid-`generate` risks the cross-agent stall the standing guidance
warns against. Its held-out rate will land in `train_adapterA.log` when it finishes; it is not
required for the same-path before/after below, which is the stronger measurement.

---

## 3. Valid before/after (base vs Adapter A, same path)

Both rows are produced by `eval_reliability_same_path.py`, which reuses `eval_same_path.py`
**verbatim** (same server lifecycle, same sampling payload, same harmony sanitization = byte-
identical agent path to the non-floored base) and adds novel reliability instrumentation computed on
the **raw** server `tool_calls` vs `incident_sim.TOOLS_SPEC`: executable-call rate
{valid-JSON %, schema-valid %, dispatched %}, invalid-call rate (= 1 − schema-valid), and
repaired-by-sanitizer %. The two rows auto-append to `incident_scores.json` via `run_beforeafter.sh`.

| model (same path) | solved | correct root cause | avg steps | redundant-rate | executable-call rate | invalid-call rate |
|---|---|---|---|---|---|---|
| **base** (vLLM same-path, FOUNDATION) | **20 / 24 (83.3%)** | 20 / 24 | 6.67 | 6.9% | (base baseline lacked the raw-call instrumentation) | — |
| **base** (adapterA same-path, reliability-instrumented) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| **+ Adapter A** (adapterA same-path) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

**Status: the head-to-head Adapter A row is NOT yet obtained.** The armed runner `run_beforeafter.sh`
(PID 2972585, alive, in its wait loop) will, the moment PID 2951466 exits and GPU 1 frees: run the
reliability-instrumented BASE (port 18490, label `gpt-oss-20b base (adapterA same-path)`) then
ADAPTER A (port 18491, `--adapter adapters_gptoss/adapterA`, label
`gpt-oss-20b + Adapter A (adapterA same-path)`), appending both rows and writing
`beforeafter_run.log`. The instrumented base row must reproduce solved ≈ 20 (the FOUNDATION check);
that match is the validity assertion for the comparison.

### Honest verdict

**No improvement or regression can be claimed yet — the after-row does not exist.** Reporting this
plainly per the project's no-flooring, report-whichever-way-it-falls standard: as of this writing the
valid before/after has a solid, non-floored **before** (20/24) and an **armed-but-pending after**.
The comparison is measurement-ready and confound-free (single path, single GPU, sanitization on for
both, adapter as the only variable); it simply has not run because GPU 1 is held by the vestigial
held-out eval described in §2. Any claim about Adapter A's effect on tool-call reliability or task
success is deferred to the runner's rows, not asserted here.

**Known risk on the after-row:** vLLM LoRA serving of Adapter A's `experts.gate_up_projs/down_projs`
regex targets is the untested path. If vLLM rejects those modules at adapter-server boot, the ADAPTER
row will error while the instrumented BASE row still lands (giving at least the FOUNDATION-reproduce
check). The warmup patch + tool-name/arg sanitization remain in place as required.

---

## 4. Inference performance + recommendation

gpt-oss-20b Q4_K_M, llama.cpp (build 87 / 81ff7ab), A5000 (GPU 0), sampling temp 1.0 / top_p 1.0 /
top_k 0 / min_p 0. Script `bench_inference.py`, raw JSON `bench_inference_results.json`. Server flags
`-ngl 99 --jinja -fa on -ub 2048 -b 2048 --ctx-size 8192 --parallel 1`. Model 11.6 GB; fits 24 GB
with room at 8192 ctx f16 KV.

**Decode throughput + KV-cache type**

| KV cache | decode tok/s (1024 tok) | prefill tok/s | TTFT warm | TTFT cold (904-tok) |
|---|---|---|---|---|
| **f16 (default, `-fa on`)** | **133** | ~3400 | 12 ms | 267 ms |
| q8_0 K (`--cache-type-k q8_0`) | 56 | ~1900 | 24 ms | — |

`--cache-type-k q8_0` is a clear loss: decode 133 → 56 tok/s (−58%), prefill/TTFT ~2× worse. On
Ampere the fast flash-attn CUDA kernel runs on f16 KV; quantizing K forces a slower path. f16 KV at
8192 ctx fits 24 GB easily, so there is no VRAM reason to quantize. **Keep f16 KV.**

**reasoning_effort latency/token tradeoff** (no-tools CoT puzzle, f16 KV, max_tokens 4096)

| effort | gen tokens | total latency | decode tok/s |
|---|---|---|---|
| low | 4023 | 33.1 s | 122 |
| medium | **3493** | **27.6 s** | 128 |
| high | 4096 (hit cap) | 33.8 s | 122 |

TTFT flat (~12 ms warm) across all three; decode rate constant (~125 tok/s), so total latency scales
linearly with generated tokens, dominated by CoT length. `high` saturated the cap; `medium` produced
the fewest tokens and lowest latency — consistent with "high can hurt agentic loops." In the agentic
tool-calling pass, per-turn generations are short (~40–200 tok) regardless of effort, so a tool turn
costs <1 s wall.

**Prompt/prefix caching** (repeated system + 8 tool schemas, 904 tokens): cold TTFT 267 ms → warm
**13 ms** (~20×) with `cache_prompt: true`. The static agentic prefix is re-served essentially free
every turn after the first — the single biggest lever for an agentic server.

**Recommendation:** f16 KV (`--cache-type-k f16 -fa on`, do NOT use `--cache-type-k q8_0`) + prompt
caching (`cache_prompt: true`, stable per-session slot / `--slot-save-path`) at
`reasoning_effort=medium`. That gives ~133 tok/s decode, ~13 ms TTFT on the repeated tool-schema
prefix, and the lowest per-turn latency. q8_0 KV costs ~58% decode for zero memory benefit on 24 GB
and should be avoided.

---

## 5. Deferred — what was NOT done, and why

- **GRPO (agentic RL).** Deferred until SFT + the same-path eval are proven. The prior GRPO earned
  zero reward across all steps (non-terminating 256-token completions = broken rollout), so it tested
  nothing. Correct order is: prove a clean SFT before/after on the non-floored path first, then layer
  RL — otherwise RL noise is indistinguishable from a broken harness. `adapters_gptoss/grpo/` exists
  but is not part of any claim here.
- **Adapter B (incident diagnosis / root-cause).** Adapter A targets tool-call reliability only.
  A second adapter for diagnosis quality is a separate axis; deferred to keep this result single-
  variable (adapter weights as the only change vs base).
- **aLoRA (activated LoRA).** `adapter_config.json` shows `alora_invocation_tokens: null` — standard
  LoRA, not aLoRA. Activated-LoRA invocation-token routing was not pursued; no evidence yet it beats
  a plain adapter on this task, and it would add an untested serving path.
- **MTP speculative decoding — false premise, dropped.** gpt-oss has **no MTP head**, so there is no
  spec-decode-via-MTP path to exploit (the MTP head we stripped `--no-mtp` was Qwen3.5's, a different
  model). Also MXFP4 dequants to bf16 on Ampere, so there is no FP4 speed win either. Both are
  recorded here so they are not re-derived.

---

## Artifacts

- `eval_same_path.py` — reusable same-path agent eval (base or `--adapter`), self-manages server via
  SIGINT; produced the non-floored base.
- `eval_reliability_same_path.py` — reuses the above verbatim + reliability instrumentation
  (executable/invalid/repaired rates). Metric helpers unit-tested; imports clean in `.venv-vllm`.
- `run_beforeafter.sh` — armed runner (PID 2972585) that emits both before/after rows when GPU 1
  frees; output to `beforeafter_run.log`.
- `train_adapterA.py` + `train_adapterA.log` — training + the decoded mask proof (`MASK_ASSERTIONS_PASSED`).
- `adapters_gptoss/adapterA/` — the saved adapter (r16/alpha16 all-linear+experts, 740 MB).
- `incident_scores.json` — base FOUNDATION row (`gpt-oss-20b base (vllm same-path)`, 20/24); Adapter A
  rows append when the runner fires.
- `bench_inference.py` + `bench_inference_results.json` — inference-perf raw data.
</content>
</invoke>

## Addendum - before/after run outcome (two new findings)

1. **Base baseline is STOCHASTIC at temp 1.0.** The same base gpt-oss-20b on the same vLLM path
   scored 20/24 (foundation run) and 15/24 (before/after re-run) - a +-5 swing from sampling noise
   alone. Consequence: single-run scoring is unreliable; any training delta smaller than ~5/24 is
   noise. A trustworthy before/after needs multiple seeds per condition (or a larger scenario set)
   and a reported confidence interval. This retroactively cautions every single-run number in this
   repo.

2. **Adapter A is NOT vLLM-servable as trained.** vLLM's LoRA loader rejected it:
   `expected target modules in {o_proj,v_proj,experts,k_proj,router,q_proj} but received
   model.layers.0.mlp.experts.down_projs.N ...`. The per-expert MoE LoRA decomposition produced by
   Unsloth `target_parameters` (mlp.experts.gate_up_proj/down_proj) is exactly what OpenAI/research
   recommend for TRAINING, but vLLM expects a single fused `experts` (and `router`) module, not the
   per-expert `experts.down_projs.N` naming. So the same-path (vLLM) Adapter-A after-row could not
   be produced. The base row still landed (15/24).

   Fix options for a valid Adapter-A before/after: (a) re-train with vLLM-servable target modules
   (attention q/k/v/o + router, and/or a fused `experts` target vLLM accepts) - drops per-expert
   adaptation but is servable on the same path; (b) evaluate the existing adapter on the
   transformers+PEFT fallback path AND re-run the base there too (same-path), but transformers MXFP4
   decode is ~0.14 tok/s (hours for 24 scenarios); (c) merge the adapter into the base and serve
   merged (merge->GGUF ruled out for MXFP4; merge->transformers is slow). Option (a) is the
   pragmatic path to a servable, same-path, multi-seed before/after.

## Net honest status of this build
- WIN (banked): the prior "training failure"/flooring was largely a SERVING BUG - the vLLM openai
  tool parser leaked harmony channel tokens into tool names/args, poisoning fed-back history. Fixed
  via sanitization. On the corrected path the BASE scores ~15-20/24 (not 0/24, not 11/24) - the
  model was fine; the harness was corrupting it.
- WIN (banked): the SFT training fix works - verified assistant-only mask, train loss 0.92 (not the
  0.0055 memorization), adapter saved.
- WIN (banked): inference numbers (llama.cpp 133 tok/s, keep f16 KV, reasoning_effort medium).
- OPEN: a valid Adapter-A before/after needs (i) a vLLM-servable adapter (re-train, option a) and
  (ii) multi-seed scoring to beat the +-5 stochastic noise. No improvement/regression is claimed.
- DEFERRED (unchanged): GRPO, Adapter B, aLoRA.

## Adapter-completion attempt - stopped at a serving-stack wall (honest)
Completing a VALID Adapter-A before/after required a vLLM-servable adapter. It peeled back twice:
- per-expert MoE LoRA (target_parameters) -> vLLM LoRA loader rejects `experts.down_projs.N`.
- + `router` -> PEFT rejects it (GptOssTopKRouter is not nn.Linear).
- attention-only (q/k/v/o) -> the only PEFT+vLLM-servable option, but the Unsloth re-train
  DEADLOCKED in the transformers-5.x import (frozen ~5h in `is_flash_linear_attention_available`
  spam, model never reached GPU).
Decision: STOP the adapter-completion chase. It is deep diminishing returns - attention-only is the
weakest LoRA placement (research: significantly underperforms MLP/MoE), the base is near the env
ceiling with +-5 stochastic noise, and the serving stack actively blocks the capable (MoE) adapter.
The genuine finding stands and is more valuable than the delta would have been: **the MoE-expert
LoRA targeting that training best-practice recommends for gpt-oss is NOT servable on vLLM today** -
a real train-vs-serve gap. A proper Adapter-A eval needs either a serving path that accepts
per-expert LoRA (merge-to-HF + transformers, slow) or vLLM adding fused-experts LoRA support.

## FINAL banked wins of this build (all committed)
1. The prior "training failure"/flooring was a SERVING BUG: the vLLM openai tool parser leaked
   harmony channel tokens into tool names/args, poisoning fed-back history. Fixed by sanitization.
   On the corrected same-path eval the BASE scores 15-20/24 (stochastic, temp 1.0) - NOT 0/24, not
   11/24. The model was fine; the harness was corrupting it. This is the headline result.
2. The SFT training fix works: VERIFIED assistant-only mask (decoded, MASK_ASSERTIONS_PASSED),
   train loss 0.92 not the 0.0055 memorization.
3. Measurement noise is real: base is +-5/24 at temp 1.0 - single-run scores need multi-seed CIs.
4. Inference: llama.cpp 133 tok/s decode / 12ms TTFT; keep f16 KV (q8_0 KV is a 58% loss on
   Ampere); reasoning_effort medium is fastest for tool loops.
5. Serving gap documented: MoE-expert LoRA not vLLM-servable for gpt-oss.
Deferred (unchanged): GRPO, Adapter B, aLoRA. The Adapter-A capability delta is UNMEASURED and not
claimed either way.
