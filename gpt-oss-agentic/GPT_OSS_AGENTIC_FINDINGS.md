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

STAGE 1 — SFT (ran and fit the rendered strings): all 520 harmony multi-turn tool trajectories
(`sft_trajectories.json`), rendered with the tool schema, seq 2048, 1 epoch = 65 optimizer
steps (bs 2 x accum 4). LoRA all-linear incl. MoE experts, r=16 alpha=32 lr 2e-4.
Trainable 184,909,824 (0.88%). Peak 17.01 GB. Loss 3.72 -> 0.0055 (final training_loss
0.4005), clean monotone convergence. Adapter: `adapters_gptoss/sft`.

Important caveat: the loss collapse proves the adapter fit the teacher-forced rendered
sequence distribution; it does not prove useful free-running tool use. The rendered
examples include the full harmony developer/tool-schema preamble. If SFT labels were not
assistant-only masked, the run may have directly supervised schema/developer tokens, which
would make later schema regurgitation a data/masking bug rather than pure overfit.

STAGE 2 — GRPO (ran end-to-end, but did not test RL): single-turn verifiable
execution-reward proxy (multi-turn rollout is not native to trl GRPOTrainer). The reward
replays the model's JSON remediation plan on the real IncidentSim (solved=1.0). The reward
fn was validated offline (gold plans -> 1.0; brute-force restart-all -> 0.0 on config/
deploy/pool, 1.0 on crash). But the online rollout produced no usable samples: reward was
uniformly 0.0 across all 12 steps, `completions/clipped_ratio=1`,
`mean_terminated_length=0`, `rewards/reward_exec=0`, and `frac_reward_zero_std=1`. Every
256-token completion hit the cap and never terminated.

This is a reward-wiring / rollout-length / format failure, not evidence about RL. With no
terminated or rewarded completions and no reward variance, no conclusion about whether GRPO
can improve the agent is supportable. More GRPO steps would not fix this by itself; the
minimum fix is rollouts that can terminate and earn reward: larger completion length,
correct stop tokens, a faithful multi-turn tool rollout, and/or shaped partial rewards that
fire before a complete solved plan.

## 4. Before -> After (confounded; controlled arm floored)

The trained artifact is a LoRA adapter on the linearized `unsloth/gpt-oss-20b`, which was
not served through llama.cpp the way the GGUF baseline was. The attempted controlled
before/after therefore used local transformers/Unsloth `generate`.
`eval_incident_adapter.py` renders the harmony prompt (stock `unsloth/gpt-oss-20b`
tokenizer + tool schema), generates one turn, stops at `<|call|>` / `<|return|>`, and
parses the harmony continuation back into OpenAI tool_calls — the SAME
`incident_harness`, the SAME 24 scenarios. Greedy decoding, 256 new-token cap.

This does NOT reproduce the grammar-constrained serving path that produced the
11/24 gpt-oss-20b capacity result. It is only a same-backend sanity check of the
base and adapter under the transformers/Unsloth path.

### Controlled results (identical transformers/Unsloth 4-bit path, greedy)

| model (path)                          | solved      | root-cause | avg steps | free-running generation quality (greedy probe) |
|---------------------------------------|-------------|------------|-----------|-----------------------------------------------|
| gpt-oss-20b base (transformers)       | 0/24 (0.0%) | 0/24       | 0.0       | emits VALID harmony tool calls, but verbose analysis overruns the 256-tok window before the call on most prompts |
| gpt-oss-20b SFT/TRAINED (transformers)| 0/24 (0.0%) | 0/24       | 0.0       | observed probes suggest degeneration: regurgitates the developer/tool-schema block, ~0 valid tool calls |

Delta (SFT - base) on solved: 0, but this delta is NOT an informative task-capacity
measurement because both runs are floored by the unconstrained decoding harness.

Reference point (different backend, grammar-CONSTRAINED): gpt-oss-20b GGUF baseline = 11/24.
Full base run 2045s, full SFT run 4512s (both greedy, 256-tok cap, all 24 scenarios, GPU-1/A6000).

### What actually happened, and what the result supports

- The controlled transformers path does NOT reproduce the GGUF baseline's capacity, for
  either model. Without llama.cpp's tool-call grammar, base gpt-oss-20b's verbose
  "Reasoning: medium" harmony analysis overruns the generation window before it emits a
  tool call on these multi-service prompts, so most episodes end at step 0. This is a
  property of the unconstrained decoding harness, not of the model's latent capacity
  (the grammar-constrained GGUF proves the capacity is 11/24).
- The base-vs-SFT task result is therefore inconclusive. A harness that floors the base
  model from 11/24 to 0/24 cannot detect existing capability, so it cannot validly measure
  whether training improved or degraded that capability.
- Separate from task success, direct greedy generation probes suggest the SFT adapter may
  have damaged free-running harmony/tool-call behavior on this transformers path: from
  `<|start|>assistant`
  it regurgitates the developer/tool-schema block (e.g. `to=functions<|message|># Tools ...
  namespace functions { ... }`) instead of emitting an analysis + a well-formed
  `to=functions.NAME<|channel|>commentary json<|message|>{...}<|call|>` tool call. The base
  model, by contrast, emits valid harmony tool calls under identical greedy decoding
  (e.g. `to=functions.get_status<|channel|>commentary <|constrain|>json<|message|>
  {"service":"courierbot"}<|call|>`). This is a warning sign about the adapter or
  rendering/masking path, not a valid capacity delta on the incident benchmark.
- This is consistent with the training telemetry: teacher-forced loss collapsed to 0.0055
  on only 520 examples with a high LR (2e-4) and heavy all-linear+MoE LoRA — a regime that
  memorizes token transitions (low teacher-forced loss) while degrading autoregressive
  free-running behavior (exposure bias / overfit). But this remains a plausible diagnosis,
  not proven: a harmony rendering, label-masking, tokenizer/template, stop-token, or parser
  bug could produce similar symptoms. The GRPO stage did not test RL because every rollout
  clipped at the token cap, never terminated, and received zero reward.

### Honest verdict

The controlled before/after is inconclusive. The only defensible task-level claim is:
the unconstrained transformers/Unsloth eval harness floored both base and SFT to 0/24, so
it cannot measure a training delta. It is not defensible to conclude from these task
scores that training did not improve gpt-oss-20b, or that SFT degraded gpt-oss-20b's
agentic capacity.

What can be said: the GGUF/llama.cpp `--jinja` baseline shows real base capacity
(11/24); the attempted transformers control failed to recover any of that capacity
(0/24); qualitative probes raise concern that the SFT adapter or its rendering/eval path
produces malformed/free-running tool calls; and GRPO was a failed 0-reward rollout. Those
facts motivate remediation, but they do not establish a before/after capability delta.

The positive results in this project are: (a) a well-shaped, unsaturated, anti-brute-force
environment; (b) a credible capacity baseline for gpt-oss-20b (11/24, grammar-constrained);
and (c) a training stack that can run on the Ampere/Unsloth path. The training result is
negative/confounded: SFT free-running collapse is observed but not diagnosed, and GRPO had
no usable online reward signal. The missing piece for a genuine before/after is serving
base and SFT through the same valid tool-constrained path after the SFT data/masking path
is audited.

### Caveats (what the delta does and does NOT support)

- Backend confound: the BEFORE capacity number (11/24) is grammar-constrained GGUF; the
  attempted before/after pair is grammar-UNconstrained transformers. Because the
  transformers path floors the base model to 0/24, it is not a valid apples-to-apples
  capacity comparison.
- The controlled transformers numbers under-report base capacity and do not prove that
  the adapter helped, failed to help, or hurt task capacity. They only show that this
  particular unconstrained transformers harness could not detect success from either
  base or SFT.
- The SFT degeneration evidence is qualitative and thin: base emits valid harmony tool
  calls in probes while SFT often regurgitates schema. That is enough to flag a likely
  adapter/eval-path problem, not enough to diagnose overfit as the sole cause.
- Dataset sizes are small: SFT 520 trajectories, GRPO a 64-scenario subset, 12 steps
  (shortened). The GRPO stage is a documented rollout/reward failure (uniform 0 reward,
  all completions clipped at 256 tokens, zero terminated completions), so no RL conclusion
  is claimed.
- The SFT LoRA is heavy (all-linear incl. MoE, r=16, lr 2e-4) on a tiny corpus; the
  overfit/degeneration hypothesis is plausible, but it is confounded with train/eval
  template mismatch and possible missing assistant-only loss masking.

### How to distinguish overfit from a data/rendering bug

Run the cheap audit before interpreting this as memorization:

1. Render several train and eval prompts to raw text and diff the assistant generation
   boundary. Role headers, tool namespace shape, channel names, `<|call|>`, `<|return|>`,
   and `add_generation_prompt` placement must match.
2. Inspect an actual SFT batch's `labels`. If the intent is assistant-only SFT, system,
   developer, tool schema, user text, and tool-result observations should be `-100`. If
   schema/developer tokens are supervised, schema regurgitation is a data/masking bug.
3. Break loss down by segment: schema/developer, user, tool observations, assistant
   tool-call headers/JSON, final answers. Near-zero schema loss plus schema generation is a
   red flag for masking/rendering contamination.
4. Run forced-prefix probes. End the prompt with
   `<|start|>assistant to=functions.get_status<|channel|>commentary json<|message|>` and
   check whether the SFT adapter completes valid JSON plus `<|call|>`. If forced prefixes
   work but normal prompts start with schema, suspect the generation prompt/template
   boundary. If forced prefixes also fail, overfit/corruption is more likely.
5. Train a tiny diagnostic adapter on 20-50 examples with confirmed assistant-only masking,
   lower LR, and the exact eval renderer. If schema regurgitation disappears, the original
   run was likely a stack/data issue. If it persists only when driven to a near-zero loss
   floor, overfit/exposure bias becomes the stronger explanation.

### Recommendation for a real improvement next

Single cheapest next step: audit and fix SFT rendering + label masking, then rerun a short
low-LR SFT probe and measure valid tool-call rate before doing the full 24-scenario eval.
This is cheaper and more diagnostic than immediately converting a possibly-buggy adapter.

After the masking/template audit passes, merge the SFT LoRA into the base, convert the
merged checkpoint to GGUF, and serve both base and SFT via the same llama.cpp `--jinja`
grammar-constrained path with the same prompt, tools, scenario set, context/token budget,
temperature, parser, and scoring. That is the apples-to-apples before/after that can answer
whether SFT changed incident-response capacity.

Separately, replace the single-turn JSON-plan GRPO proxy or make it terminate and reward:
larger `max_completion_length`, correct stop tokens, a real multi-turn tool loop if
possible, and shaped partial rewards. The current GRPO run supports no RL conclusion.

## Files

- `incident_sim.py` — environment (10-service fleet, faults, expert trajectories, TOOLS_SPEC).
- `incident_harness.py` — backend-agnostic multi-turn agent loop + expert/brute-force refs.
- `eval_incident_gguf.py` — GGUF baseline via llama-server `--jinja` (grammar-constrained).
- `eval_incident_adapter.py` — adapter/base eval via transformers/Unsloth generate (controlled).
- `train_gptoss.py` — SFT + GRPO stages (Unsloth QLoRA, all-linear+MoE LoRA).
- `incident_scores.json` — all eval records (GGUF baselines + controlled transformers pair).
- `adapters_gptoss/sft`, `adapters_gptoss/grpo` — trained LoRA adapters (git-ignored, ~733 MB each).
