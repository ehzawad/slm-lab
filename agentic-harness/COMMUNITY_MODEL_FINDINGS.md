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
| gemma4-12B-agentic-v2  | llama.cpp | 7/8 | 29 | 24 | 0 | 72 | capable but flails |
| Qwythos-9B-Mythos      | llama.cpp | -   | -  |  - | - | -  | BLOCKED - GGUF convert fails |
| Qwythos-9B-Mythos      | vLLM      | (in progress) |||||| |

## Successes

- **gpt-oss-20b remains the reference local agent:** 8/8, fewest tool calls (8), zero parser
  workarounds (its harmony format parses natively in llama.cpp), fastest generation (177 t/s).
- **gemma4-12B-agentic-v2 (yuxinlu1)** loads and serves fine; Gemma 4's native tool protocol
  parses cleanly through `--jinja` (0 fallbacks). It is genuinely capable (7/8).
- **llama.cpp update fixed the Qwythos convert** at the file level (426 -> 427 tensors written).

## Failures & breaking points (with root cause)

1. **Qwythos GGUF is blocked by the hybrid architecture.** Qwen3.5 uses a 3:1 Gated-DeltaNet
   (linear-attention) to full-attention layer mix. The convert writes a GGUF, but the llama.cpp
   RUNTIME rejects it: `missing tensor 'blk.32.attn_norm.weight'` - the DeltaNet layers do not
   carry the norm tensors the loader expects, and this is unresolved even on the latest build
   (b9923, Jul-8). Pre-made base-Qwen3.5 GGUFs work only because their publisher converted them
   with patched tooling. Conclusion: **arbitrary Qwen3.5 fine-tunes cannot be self-converted to
   GGUF today** - they need vLLM/transformers with the Gated-DeltaNet kernels
   (flash-linear-attention + causal_conv1d). (vLLM run in progress.)

2. **gemma4 flails.** 7/8 with 24 tool calls (3x gpt-oss) and 7 tool-errors; `error_recover`
   failed by burning the loop budget on retries without emitting a final answer - matching the
   model card's own honest caveat ("still over-tries/retries"). Fairness note: our order-DB
   tasks are tau2-retail-shaped, outside its tuned telecom/terminal home domain, so this is an
   out-of-domain result, not a refutation of its telecom claim.

3. **The `pkill -f <pattern>` self-kill (recurring operational trap).** Commands containing
   `pkill -f 'llama-server'` (or any pattern that is also a substring of the shell's own command
   line) SIGTERM their own wrapper -> exit 144, before doing useful work. Fix: manage servers via
   subprocess handles / SIGINT, or use patterns that cannot match the invoking command.

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
- Qwythos safetensors (~18 GB) cached at `models/qwythos_src`; f16 GGUF builds but will not load.
