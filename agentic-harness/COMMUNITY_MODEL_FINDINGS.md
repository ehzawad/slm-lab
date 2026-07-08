# Community-Model & Tool-Reliability Findings

Findings from testing user-requested community models on the agentic harness (single 8-task
multi-turn MCP suite, end-to-end scored) and investigating why tool calls fail. Successes,
failures, breaking points, and nuances - recorded as we hit them. Hardware: A5000 (GPU 0) and
A6000 (GPU 1, used when free).

## Results table (agentic harness, Q4_K_M unless noted)

| Model | Runtime | Tasks | Iters | Tool calls | Parser fallbacks | tg t/s | Verdict |
|-------|---------|:-----:|------:|-----------:|:----------------:|-------:|---------|
| gpt-oss-20b            | llama.cpp | 8/8 | 17 |  8 | 0 | 177 | best: accurate + efficient |
| Qwen3.5-9B (old build) | llama.cpp b(Jul-6) | 8/8 | 22 | 14 | 5 | 96 | needs XML shim |
| Qwen3.5-9B (new build) | llama.cpp b9923 | 8/8 | 14 | 14 | 5 | 96 | STILL needs shim (see below) |
| gemma-4-12B-it BASE    | llama.cpp | 8/8 | 22 | 13 | 0 | 72 | base is clean 8/8 |
| gemma4-12B-agentic-v2  | llama.cpp | 7/8 | 29 | 24 | 0 | 72 | capable but flails |
| Qwythos-9B nomtp       | llama.cpp (--no-mtp) | 8/8 | 20 | 12 | 0 | 96 | 8/8 at temp 0.6, no shim needed |
| Qwythos-9B-Mythos      | vLLM      | -   | -  |  - | - | -  | loads, then Triton warmup crash |

## Successes

- **gpt-oss-20b remains the reference local agent:** 8/8, fewest tool calls (8), zero parser
  workarounds (its harmony format parses natively in llama.cpp), fastest generation (177 t/s).
- **gemma4-12B-agentic-v2 (yuxinlu1)** loads and serves fine; Gemma 4's native tool protocol
  parses cleanly through `--jinja` (0 fallbacks). It is genuinely capable (7/8).
- **llama.cpp update fixed the Qwythos convert** at the file level (426 -> 427 tensors written),
  but the default GGUF is not loadable by the runtime because of the extra MTP block.

## Failures & breaking points (with root cause)

1. **Qwythos default GGUF is blocked by the extra MTP head.** The Hugging Face config has
   `mtp_num_hidden_layers: 1`. The default converter emits this as an extra block, and llama.cpp
   then rejects the file at runtime with `missing tensor 'blk.32.attn_norm.weight'`. The working
   base GGUF was converted without that head. Re-converting Qwythos with `--no-mtp` should produce
   the same 32-layer main-model layout as the base GGUF. This is a defensible benchmarking path
   because the MTP head is an auxiliary speculative/draft head, not the normal next-token head;
   however, it should be reported as "Qwythos converted with `--no-mtp`" rather than as a
   byte-for-byte-equivalent serving path.

2. **gemma4 flails.** 7/8 with 24 tool calls (3x gpt-oss) and 7 tool-errors; `error_recover`
   failed by burning the loop budget on retries without emitting a final answer - matching the
   model card's own honest caveat ("still over-tries/retries"). Fairness note: our order-DB
   tasks are tau2-retail-shaped, outside its tuned telecom/terminal home domain, so this is an
   out-of-domain result, not a refutation of its telecom claim.

3. **vLLM is not worth more pinning for this spot check.** vLLM 0.24.0 loads Qwythos and resolves
   the architecture, but crashes during global Triton kernel warmup in unrelated JIT parsing
   (`AttributeError: 'NoneType' object has no attribute 'start'`, observed around a minimax_m3
   top-k kernel). The crash persisted after downgrading Triton 3.6.0 -> 3.5.0 and using
   `--enforce-eager`. Since llama.cpp `--no-mtp` exercises the main model path, and since the
   benchmark itself is ceilinged for Qwythos, further vLLM version archaeology is low ROI.

4. **The `pkill -f <pattern>` self-kill (recurring operational trap).** Commands containing
   `pkill -f 'llama-server'` (or any pattern that is also a substring of the shell's own command
   line) SIGTERM their own wrapper -> exit 144, before doing useful work. Fix: manage servers via
   subprocess handles / SIGINT, or use patterns that cannot match the invoking command.

## Completed base-vs-fine-tune comparison (2 pairs, agentic + reasoning)

Both pairs are now fully run. Because the agentic harness saturates at 8/8 for capable models, we
added a GSM8K reasoning probe (n=40) as the discriminating axis. Qwythos and its base Qwen were run
at temp 0.6 / top_p 0.95 (Qwythos degenerates at temp 0; base run at 0.6 too for a fair match);
the Gemma pair reasoning probe was also run at temp 0.6.

| Pair | Model | Role | Agentic | Tool calls | XML fallbacks | Reasoning acc% |
|------|-------|------|:-------:|:----------:|:-------------:|:--------------:|
| Qwen  | Qwen3.5-9B Q4_K_M      | base     | 8/8 | 14 | 5 | 90.0 |
| Qwen  | Qwythos-9B nomtp       | finetune | 8/8 | 12 | 0 | 95.0 |
| Qwen  | **delta (ft - base)**  |          | **0** | **-2** | **-5** | **+5.0** |
| Gemma | gemma-4-12B-it BASE Q4_K_M     | base     | 8/8 | 13 | 0 | 42.5 |
| Gemma | gemma4-12B-agentic-v2 Q4_K_M   | finetune | 7/8 | 24 | 0 | 95.0 |
| Gemma | **delta (ft - base)**          |          | **-1** | **+11** | **0** | **+52.5** |

Reasoning probe: n=40 GSM8K, temp 0.6, top_p 0.95. Agentic = tasks passed on the 8-task MCP suite.

### Qwythos vs Qwen3.5-9B: agentic ceilinged, reasoning modestly positive

Qwythos is a fine-tune of Qwen3.5-9B. Both score 8/8 on the agentic suite, so pass/fail on this
saturated harness supports NEITHER an improvement NOR a regression - it is parity on an
already-solved suite. What the agentic axis DOES show is a reliability/efficiency edge: Qwythos used
fewer tool calls (12 vs 14) and needed zero XML-parser fallbacks (vs 5 for base), i.e. its native
tool-calling parsed cleanly through llama.cpp `--jinja` where the base still tripped the shim.

On the discriminating axis (reasoning) Qwythos scored 95.0% vs base 90.0%, a **+5.0 pt** delta.
This SUPPORTS a real but modest reasoning gain from the fine-tune. It does NOT support the card's
claimed +30 GSM8K: at n=40 the base already sits near ceiling (90%), leaving little visible
headroom, and both models converged on the same hard items (both missed q38). The delta is
directionally consistent with the card but far smaller in this setup; a larger, harder,
unsaturated probe would be needed to test the +30 claim.

### gemma4-agentic-v2 vs gemma-4-12B-it: agentic regression, large out-of-domain reasoning gain

On the agentic suite the fine-tune REGRESSED: 7/8 vs base 8/8, with ~1.8x the tool calls (24 vs 13)
and 7 tool-errors, failing `error_recover` by burning the loop budget on retries without emitting a
final answer (matches the card's own "over-tries/retries" caveat). Our order-DB tasks are
tau2-retail-shaped, OUT OF DOMAIN for a model tuned on telecom/terminal loops, so this regression is
a domain-transfer result and does NOT refute the card's telecom claim (which is untested here).

On reasoning the fine-tune scored 95.0% vs base 42.5%, a **+52.5 pt** delta (+21/40 correct). This
is a large, factual, positive reasoning result: whatever the agentic-v2 tuning did, it dramatically
improved general GSM8K reasoning over this particular base checkpoint. Two honest caveats: (1) the
base gemma-4-12B-it scored surprisingly low (42.5%) for a 12B instruct model, so part of the delta
may reflect a weak base checkpoint rather than pure fine-tune strength; (2) strong reasoning did NOT
transfer into agentic reliability on our retail tasks - the specialization helped reasoning but hurt
tool-loop efficiency here. Net: on reasoning the finetune is far stronger; on retail-shaped agentic
loops it is weaker. A fine-tune can win one axis and lose another - reported as found.

### Minimal changes to make the comparison more informative, ranked by ROI

1. **Harder, unsaturated task set.** Highest ROI for Qwythos. Add longer-horizon tasks, larger
   tool inventories, adversarial tool outputs, stricter state mutation, nested/multi-entity
   updates, and tighter loop budgets. Until the base is below ceiling, pass/fail cannot show an
   improvement.
2. **Efficiency metrics on the existing harness.** Cheapest immediate gain. Keep reporting
   tool-call count, iterations, invalid calls, parser fallbacks, retries, latency, and tokens.
   This can distinguish solved runs, but it should be labeled as efficiency/reliability, not
   capability.
3. **Domain-matched task set.** Highest ROI for gemma4-agentic-v2 specifically. To evaluate the
   advertised fine-tune, add telecom/terminal-loop tasks. The current order-DB suite is a domain
   transfer test.
4. **More trials / more tasks for statistical power.** Needed eventually, because N=8 single-run
   deterministic scoring is thin evidence. But extra trials do not fix a ceilinged or mismatched
   task distribution by themselves.

If no harder or domain-matched tasks are added, the base-vs-fine-tune exercise remains mostly a
smoke test: useful for catching regressions, serving failures, and efficiency differences, but not
for claiming fine-tune capability gains.

## The tool-error investigation (the important nuance)

Three distinct things look like "tool trouble"; only one is a real, fixable defect:

- **(a) Intentional error-injection - not a bug.** Two tasks return a tool error on purpose to
  score recovery, so every model shows ~2 "tool-errors" by design.
- **(b) Serving-parser mismatch - the real defect.** The model emits a correct tool call, but the
  server does not parse it into a structured `tool_calls` object, so it arrives as raw text and
  the agent loop stalls. Qwen3.5 hits this: it emits `<tool_call><function=..><parameter=..>` XML
  that llama.cpp's `--jinja` path does not always structure on multi-turn.
- **(c) Model flailing - genuine capability, not config.** Redundant/wrong calls (gemma4). No
  parser fixes this.

### Nuance that corrected an over-claim

A minimal 1-tool, 2-turn probe on the updated llama.cpp (b9923) returned structured `tool_calls`
on both turns - suggesting the parser gap was fixed. **But the full 7-tool harness still needed
the XML shim 5 times** (cancel_unshipped x1, multihop x2, state_memory x1, error_recover x1). So:
the upstream update *helps the simple case but does not close the gap for real multi-tool agentic
flows.* Lesson: verify parser behavior on the actual workload, not a toy probe - the easy case is
misleadingly clean.

## How to make tool calls reliable (the answer)

In order of robustness:

1. **Match the model's tool format to a real parser.** gpt-oss + harmony parses natively in
   llama.cpp (0 fallbacks). For Qwen-family, use **vLLM with `--enable-auto-tool-choice
   --tool-call-parser hermes`** (the dedicated parser) rather than relying on llama.cpp `--jinja`.
2. **Grammar-constrained decoding (GBNF / JSON-schema).** Force every tool call to valid schema
   JSON - a model-agnostic hard guarantee that eliminates causes (b) entirely.
3. **temperature 0** for deterministic tool selection (already used here).
4. **A parser shim** (what this harness does) is a last resort that recovers unparsed XML calls -
   effective but a workaround, not a fix.

Config that "makes all tool calls work": current llama.cpp + `--jinja` for gpt-oss; vLLM +
`--tool-call-parser` (or GBNF) for the Qwen family. Model choice still matters for cause (c):
gpt-oss-20b is the most reliable + efficient agent measured here.

## Toolchain notes

- llama.cpp updated Jul-6 (f36e5c3) -> Jul-8 (b9923, 81ff7ab) - re-convert + full CUDA rebuild.
- vLLM installed in an isolated `.venv-vllm` to protect the training stack (torch 2.12 / trl 1.7).
- Qwythos safetensors (~18 GB) cached at `models/qwythos_src`; default f16 GGUF builds but will
  not load because of the MTP block. The current llama.cpp path is re-conversion with `--no-mtp`,
  then Q4_K_M quantization.
