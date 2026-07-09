# gpt-oss-20b on a harder agentic env: deep-dive baseline, training stack, and honest before/after

NimbusWorks incident-response environment. This report covers (1) why the environment is
unsaturated and anti-brute-force, (2) the gpt-oss-20b deep-dive baseline (its agentic
capacity on a real, unsaturated problem), (3) the training stack that actually ran
(gpt-oss-20b via Unsloth QLoRA; gate was GO), and (4) an honest before/after with caveats.

## 1. The environment: unsaturated and anti-brute-force by construction

`incident_sim.py` builds a deterministic 10-service fleet (NimbusWorks) with a dependency
graph. Each scenario injects exactly ONE root-cause fault; other alerting services are
cascading symptoms whose logs explicitly say `NBX-3301 upstream ... symptom, not root
cause`. The agent has a 15-tool-call budget and must (a) find the true root, (b) apply the
CORRECT fix for the fault type, and (c) leave every service healthy.

Four fault classes, two of them two-step:

| fault_type      | correct remediation                               | steps |
|-----------------|---------------------------------------------------|-------|
| bad_deploy      | `rollback(root)`                                  | 1     |
| crash           | `restart(root)`                                   | 1     |
| bad_config      | `set_config(root, key, value)` THEN `restart(root)` | 2   |
| pool_exhausted  | `set_config(root, pool_max>=512)` THEN `restart(root)` | 2 |

The environment is anti-brute-force: a bare "restart everything" agent is proven (in
`incident_harness.py`) to FAIL every bad_config / bad_deploy / pool_exhausted scenario,
because a restart does not fix a config, a deploy, or an exhausted pool. The expert
(gold-trajectory) agent solves 24/24. So there is a hard ceiling that is reachable and a
cheap heuristic that provably cannot reach it — the definition of an unsaturated,
skill-discriminating env.

## 2. gpt-oss-20b deep-dive baseline (GGUF, llama.cpp `--jinja`, grammar-constrained)

Served via `eval_incident_gguf.py`: `llama-server` on GPU-1/A6000 with `--jinja` (tool-call
grammar), OpenAI `/v1/chat/completions` wired into the same `incident_harness`, all 24
scenarios. This is the legitimate "capacity" measure of gpt-oss agentic behavior because
the tool-call grammar keeps output well-formed.

| model         | temp | solved       | root-cause | avg steps | redundant-call rate | elapsed |
|---------------|------|--------------|------------|-----------|---------------------|---------|
| gpt-oss-20b   | 1.0  | 11/24 (45.8%)| 11/24      | 8.88      | 35.2% (75/213)      | 366s    |
| Qwen3.5-9B    | 0.7  | 1/24 (4.2%)  | 1/24       | 2.38      | 1.8% (1/57)         | 104s    |
| Qwythos-9B    | 0.6  | 10/24 (41.7%)| 10/24      | 12.08     | 16.9% (49/290)      | 335s    |

Verdict: appropriately hard, NOT saturated. The strongest baseline (gpt-oss-20b) solves
under half. Failure structure is informative, not random:

- The two capable models (gpt-oss-20b, Qwythos-9B) solve the SINGLE-STEP classes
  (bad_deploy -> rollback, crash -> restart) but score ~0 on the TWO-STEP classes
  (bad_config and pool_exhausted, which need set_config THEN restart). gpt-oss got 0/7
  bad_config and 0/4 pool_exhausted. This is exactly the axis the env was built around.
- gpt-oss's 35.2% redundant-call rate reflects blind restart/retry loops on faults it
  cannot reason through (10-11 redundant calls on failed config/pool episodes to the budget).
- Qwen3.5-9B barely engages tool-calling (20/24 episodes end at step 1 with no tool call):
  a genuine weak agentic prior, a valid floor with maximum headroom.

The marginal learning signal therefore lives in bad_config + pool_exhausted (the
set_config->restart sequencing).

## 3. The training stack that ran (gpt-oss-20b via Unsloth; gate = GO)

Feasibility gate was GO. gpt-oss-20b is MoE + MXFP4; native MXFP4 training is unsupported,
so the only Ampere QLoRA path is Unsloth's linearized `unsloth/gpt-oss-20b` loaded in
4-bit. `DISABLE_ADDMM_CUDA_LT=1` is required on this stack (sm_86 / bitsandbytes / cu130);
without it, 4-bit matmul raises `CUBLAS_STATUS_NOT_INITIALIZED`. Script: `train_gptoss.py`.

STAGE 1 — SFT (real, converged): all 520 harmony multi-turn tool trajectories
(`sft_trajectories.json`), rendered with the tool schema, seq 2048, 1 epoch = 65 optimizer
steps (bs 2 x accum 4). LoRA all-linear incl. MoE experts, r=16 alpha=32 lr 2e-4.
Trainable 184,909,824 (0.88%). Peak 17.01 GB. Loss 3.72 -> 0.0055 (final training_loss
0.4005), clean monotone convergence. Adapter: `adapters_gptoss/sft`.

STAGE 2 — GRPO (ran end-to-end, SHORTENED, learning NO-OP, honest): single-turn verifiable
execution-reward proxy (multi-turn rollout is not native to trl GRPOTrainer). The reward
replays the model's JSON remediation plan on the real IncidentSim (solved=1.0). The reward
fn was validated offline (gold plans -> 1.0; brute-force restart-all -> 0.0 on config/
deploy/pool, 1.0 on crash). But reward was uniformly 0.0 across all 12 steps -> zero
advantage -> no useful gradient. Root cause: the single-turn JSON-plan proxy is
out-of-distribution for the SFT policy, which is a specialized multi-turn harmony
tool-caller; asked for a one-shot plan it degenerates and never emits a parseable plan. So
the GRPO adapter is effectively identical to the SFT adapter. Shortened because HF-generate
rollout on the 20B MoE ran ~5-10 min/step (full config projected ~5.5 h) and, once trimmed,
produced no signal.

## 4. Before -> After (honest, controlled)

The trained artifact is a LoRA adapter on the linearized `unsloth/gpt-oss-20b`, which cannot
be served through llama.cpp the way the GGUF baseline was. So the adapter must be evaluated
via local transformers/Unsloth `generate`. `eval_incident_adapter.py` renders the harmony
prompt (stock `unsloth/gpt-oss-20b` tokenizer + tool schema), generates one turn, stops at
`<|call|>` / `<|return|>`, and parses the harmony continuation back into OpenAI tool_calls —
the SAME `incident_harness`, the SAME 24 scenarios. To isolate the adapter's effect we run
the base weights through this IDENTICAL path (controlled before) and the SFT adapter through
it (after). Greedy decoding, 256 new-token cap.

### Controlled results (identical transformers/Unsloth 4-bit path, greedy)

| model (path)                          | solved      | root-cause | avg steps | free-running generation quality (greedy probe) |
|---------------------------------------|-------------|------------|-----------|-----------------------------------------------|
| gpt-oss-20b base (transformers)       | 0/24 (0.0%) | 0/24       | 0.0       | emits VALID harmony tool calls, but verbose analysis overruns the 256-tok window before the call on most prompts |
| gpt-oss-20b SFT/TRAINED (transformers)| 0/24 (0.0%) | 0/24       | 0.0       | DEGENERATE: regurgitates the developer/tool-schema block, ~0 valid tool calls |

Delta (SFT - base) on solved: 0 (both floored to 0 by the unconstrained decoding harness).
The informative difference is qualitative, not in the tied solved count: see below.

Reference point (different backend, grammar-CONSTRAINED): gpt-oss-20b GGUF baseline = 11/24.
Full base run 2045s, full SFT run 4512s (both greedy, 256-tok cap, all 24 scenarios, GPU-1/A6000).

### What actually happened, and what the delta supports

- The controlled transformers path does NOT reproduce the GGUF baseline's capacity, for
  either model. Without llama.cpp's tool-call grammar, base gpt-oss-20b's verbose
  "Reasoning: medium" harmony analysis overruns the generation window before it emits a
  tool call on these multi-service prompts, so most episodes end at step 0. This is a
  property of the unconstrained decoding harness, not of the model's latent capacity
  (the grammar-constrained GGUF proves the capacity is 11/24).
- The two paths TIE at 0/24 solved on this floored harness, but the SFT adapter is
  qualitatively WORSE in generation quality on the same footing. Direct greedy generation
  probes show the SFT policy DEGENERATES: from `<|start|>assistant`
  it regurgitates the developer/tool-schema block (e.g. `to=functions<|message|># Tools ...
  namespace functions { ... }`) instead of emitting an analysis + a well-formed
  `to=functions.NAME<|channel|>commentary json<|message|>{...}<|call|>` tool call. The base
  model, by contrast, emits valid harmony tool calls under identical greedy decoding
  (e.g. `to=functions.get_status<|channel|>commentary <|constrain|>json<|message|>
  {"service":"courierbot"}<|call|>`). So the adapter damaged free-running generation.
- This is consistent with the training telemetry: teacher-forced loss collapsed to 0.0055
  on only 520 examples with a high LR (2e-4) and heavy all-linear+MoE LoRA — a regime that
  memorizes token transitions (low teacher-forced loss) while degrading autoregressive
  free-running behavior (exposure bias / overfit). The GRPO stage, which was a documented
  0-reward no-op, did not repair this.

### Honest verdict

Training did NOT improve gpt-oss-20b's agentic capacity on the hard env, and on an
identical evaluation footing the SFT adapter measurably degraded free-running tool-calling.
The positive results in this project are: (a) a well-shaped, unsaturated, anti-brute-force
environment; (b) a credible capacity baseline for gpt-oss-20b (11/24, grammar-constrained);
and (c) a training stack that runs end-to-end on the Ampere/Unsloth path (SFT converges;
GRPO wired and verified to execute with a real, offline-validated execution reward). The
missing piece for a genuine improvement is a rollout/eval regime aligned with the policy.

### Caveats (what the delta does and does NOT support)

- Backend confound: the BEFORE capacity number (11/24) is grammar-constrained GGUF; the
  controlled before/after pair is grammar-UNconstrained transformers. The clean, apples-to-
  apples controlled pair (base vs SFT through the same transformers path) is what isolates
  the adapter's effect; the GGUF number is a separate, stronger-harness reference.
- The controlled transformers numbers are floored by the unconstrained decoding harness
  (base near-0), so they under-report base capacity; they still validly show the adapter
  did not help and in fact hurt free-running generation (base emits valid tool calls,
  SFT regurgitates schema).
- Dataset sizes are small: SFT 520 trajectories, GRPO a 64-scenario subset, 12 steps
  (shortened). The GRPO stage is a documented learning no-op (uniform 0 reward, proxy
  format OOD for the policy), so no GRPO-driven improvement is claimed.
- The SFT LoRA is heavy (all-linear incl. MoE, r=16, lr 2e-4) on a tiny corpus; the
  overfit/degeneration is the most likely cause and is directly observed in greedy probes.

### Recommendation for a real improvement next

Either (a) convert the SFT-merged model to GGUF and serve via llama.cpp `--jinja` so the
tool-call grammar constrains the (degenerate) free-running output the same way it does the
base — this would give a true grammar-constrained before/after on identical footing; and/or
(b) reduce the SFT regime (lower LR ~5e-5, fewer epochs / early stop, lighter LoRA) to stop
the free-running degeneration; and (c) implement a genuine multi-turn rollout reward (custom
loop, not vanilla GRPOTrainer) so RL uses the trained harmony tool-calling instead of the
OOD single-turn JSON-plan proxy.

## Files

- `incident_sim.py` — environment (10-service fleet, faults, expert trajectories, TOOLS_SPEC).
- `incident_harness.py` — backend-agnostic multi-turn agent loop + expert/brute-force refs.
- `eval_incident_gguf.py` — GGUF baseline via llama-server `--jinja` (grammar-constrained).
- `eval_incident_adapter.py` — adapter/base eval via transformers/Unsloth generate (controlled).
- `train_gptoss.py` — SFT + GRPO stages (Unsloth QLoRA, all-linear+MoE LoRA).
- `incident_scores.json` — all eval records (GGUF baselines + controlled transformers pair).
- `adapters_gptoss/sft`, `adapters_gptoss/grpo` — trained LoRA adapters (git-ignored, ~733 MB each).
