# Multi-Turn Agentic Harness — Results (RTX A5000)

This is the **v2** eval that the single-turn tool-call probe could not do: tools are
actually **executed**, results are **fed back**, the agent **loops** until done, and
scoring is **end-to-end** (final answer + final MCP database state). Tool sources:
a real minimal **MCP server** (JSON-RPC stdio, stateful order DB) + local tools
(calculator, kv memory, mock search). 8 tasks cover sequential mutation, multi-hop
chains, cross-turn state/memory, parallel compare, tool-error recovery, and abstention.

## Results

| Model | Tasks | Iters | Tool calls | XML-fallback¹ | Err-recovery² |
|-------|:-----:|------:|-----------:|:-------------:|:-------------:|
| Qwen3.5-4B Q4_K_M  | 8/8 | 21 | 13 | 1 | 2/2 |
| Qwen3.5-9B Q4_K_M  | 8/8 | 22 | 14 | 5 | 2/2 |
| Qwen3.5-9B Q8_0    | 8/8 | 21 | 13 | 4 | 2/2 |
| gpt-oss-20b Q4_K_M | 8/8 | 17 |  8 | 0 | 2/2 |

¹ Times llama.cpp's `--jinja` parser failed to structure Qwen's native
`<tool_call><function=…><parameter=…>` output, requiring a fallback parser to recover
the call. ² Tasks with an injected tool error (shipped-order rejection, missing-order)
that the agent still completed.

## Findings

1. **Capability is a tie at this difficulty — all 8/8.** Every model executes
   multi-turn tool loops, multi-hop reasoning, cross-turn memory, parallel calls, and
   recovers from tool errors. The single-turn probe (Qwen 8/8, gpt-oss 6/8, Gemma/DeepSeek
   0/8) was misleading in **both** directions.

2. **The differentiator is the serving stack, not capability.** Out-of-the-box in
   llama.cpp, **Qwen3.5-9B's tool calls silently drop ~1/3 of the time on multi-turn**
   (5 XML-fallback recoveries out of 14 calls) because llama.cpp does not parse Qwen3.5's
   native tool-call XML after the first turn. gpt-oss-20b needed the fallback **zero**
   times — its harmony format parses cleanly. Fix for Qwen: serve via vLLM with
   `--tool-call-parser qwen3_coder`, or apply a parser shim (as done here).

3. **gpt-oss-20b is the most efficient agent** — 17 iters / 8 calls vs Qwen's ~21 / 14,
   and it answered `calc_direct` and `state_memory` with **zero** tool calls (better
   judgment about when *not* to call a tool). Consistent with its agentic-RL training.

4. **Quant effect (9B Q8 vs Q4):** both 8/8; Q8 needed marginally fewer fallbacks (4 vs 5)
   and calls (13 vs 14) — weakly consistent with "Q8 slightly cleaner for agents," but
   within noise at N=8.

## Community-model spot check: gemma4-12B-agentic-v2 (user-requested)

`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF` - a popular community
fine-tune of Gemma-4-12B specialized for coding/terminal agentic loops (claims ~3.5x over
base on tau2-bench telecom; distilled CoT rebuilt with Opus 4.8). Run at Q4_K_M on the A6000,
same harness, same 8 tasks:

| Model | Tasks | Iters | Tool calls | Tool-errors | XML-fallback | tg t/s |
|-------|:-----:|------:|-----------:|------------:|:------------:|-------:|
| gpt-oss-20b Q4_K_M          | 8/8 | 17 |  8 | 2 | 0 | 177 |
| Qwen3.5-9B Q4_K_M           | 8/8 | 22 | 14 | 1 | 5 | 96  |
| gemma4-12B-agentic-v2 Q4_K_M | 7/8 | 29 | 24 | 7 | 0 | 72  |

Findings: (1) Gemma 4's native tool protocol parses cleanly through llama.cpp `--jinja` -
zero parser workarounds, like gpt-oss and unlike Qwen3.5. (2) It is capable but inefficient:
7/8 with 3x the tool calls of gpt-oss and 7 tool-errors; `state_memory` took 8 iterations of
flailing before succeeding, and `error_recover` failed by burning the loop budget on retries
without ever emitting a final answer - matching the model card's own honest caveat that v2
"still flails a little sometimes (over-trying, retrying)". (3) Fairness note: these order-DB
tasks resemble tau2 retail (where the card says the base model wins) more than the
telecom/terminal loops it was tuned for, so this reads as an out-of-home-domain result, not a
refutation of the card's telecom claim.

## Corroboration with official numbers

Qwen3.5's own model card reports strong multi-turn agentic scores (BFCL-V4: 9B **66.1**,
4B 50.3; TAU2-Bench: 9B **79.1**, 4B **79.9** — both beating Qwen3-Next-80B), confirming
the capability our harness measured. gpt-oss-20b's card documents agentic-RL training,
harmony format, and Structured Outputs, consistent with its clean, efficient runs here.

## Honest limitations

- **N=8 tasks, deterministic (temp 0), single seed** — directional, not statistically
  powered. Task difficulty saturates (all 8/8); harder/longer-horizon tasks (10–30 steps,
  bigger tool sets, adversarial tool outputs) would re-separate the models.
- **MCP server is a minimal subset** (initialize / tools/list / tools/call over stdio) —
  enough to exercise real discovery + invocation, not a full MCP conformance test.
- The **XML-fallback** measures *capability despite* the parser gap; production users on
  llama.cpp must apply the shim or a proper parser, or accept dropped Qwen tool calls.

## Reconciled recommendation (full agentic harness, 24GB A5000)

- **gpt-oss-20b Q4_K_M** — most robust + efficient *out-of-the-box* in llama.cpp; the
  strongest default if MoE/~21B-total is acceptable (fits at ~11GB).
- **Qwen3.5-9B** (Q6_K/Q8_0 for production) — equal capability and best official BFCL/TAU2
  scores, **but requires the correct tool-call parser** (vLLM `qwen3_coder` or a shim);
  don't deploy it on raw llama.cpp `--jinja` without that fix.
- **Qwen3.5-4B Q4_K_M** — same capability, lowest footprint; ideal SLM-default/router.
