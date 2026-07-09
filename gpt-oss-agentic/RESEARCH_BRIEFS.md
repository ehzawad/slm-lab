# gpt-oss-20b Agentic Build - Research Briefs (4 parallel agents, cited)

Consolidated actionable findings for training, inference, harness reliability, and Granite-style
task adapters. These drive the ROBUST_PLAN and the build workflow. Evidence-quality flags kept.

## 1. Robust agentic TRAINING (fixing our failed run)

Our failure diagnosed: the SFT degeneration (adapter regurgitated the developer/tool-schema block)
is the classic signature of a **missing assistant-only loss mask** - loss was computed over
system/developer/tools tokens, so it learned to reproduce the schema. Compounded by 65 steps (~2
epochs) on 520 examples = memorization (loss 0.0055), and possibly wrong harmony rendering.

SFT fix recipe:
- **Assistant-only mask:** `train_on_responses_only` with the `<|start|>assistant` boundary - train
  on assistant analysis+commentary+final; mask developer/tools/user/tool-results. **VERIFY by
  decoding one masked batch** (non -100 tokens must be assistant-only) BEFORE training. Known sharp
  edges on multi-turn harmony (unsloth #823/#1017) - verify empirically.
- **Tool-call rendering (exact target):** `<|start|>assistant<|channel|>commentary to=functions.NAME
  <|constrain|>json<|message|>{...}<|call|>`; tools declared in the developer message as a
  TypeScript-style `namespace functions {...}`; tool results fed back as role
  `functions.NAME to=assistant<|channel|>commentary<|message|>{...}<|end|>`. Use the official
  renderer (apply_chat_template / openai-harmony / Unsloth encode_conversations_with_harmony) - never
  hand-rolled - and the SAME renderer at train and eval.
- **Multi-turn CoT:** train on the current turn's analysis, but STRIP prior turns' analysis from
  context (harmony spec drops stale CoT after a final/tool-result).
- **Hyperparams:** lr 1e-4 to 2e-4, LoRA r16 alpha16 (drop from 32), `all-linear` PLUS
  `target_parameters` for `mlp.experts.gate_up_proj/down_proj` (OpenAI uses layers 7/15/23),
  max_seq 2048-4096, eff-batch 16, cosine + 3% warmup, **1 epoch (~32 steps), at most 2**, early-stop
  on rising held-out loss, target train loss plateau 0.3-0.8 NOT <0.05.
- **Success metric:** free-running generation on a held-out split (valid-tool-call rate), not
  teacher-forced loss.

GRPO fix (only after solid SFT; our GRPO was a no-op - 256-token completions never terminated):
- max_completion_length >= 1024 (or reasoning_effort=low to shrink CoT); register harmony stop
  tokens (<|call|> <|return|> <|end|>) so a tool call ends the turn and can be scored.
- **Graded reward with partial credit:** reward a parseable correct-schema tool call BEFORE full task
  success (so early gradient exists); never a single sparse binary.
- lr 5e-5, num_generations >= 4, temp 1.0, hundreds of steps (zero reward for the first ~100-200 is
  normal; zero from truncation is a bug). Start single-turn RLVR; consider GSPO for long-completion
  stability. No first-party multi-turn tool-loop gpt-oss GRPO recipe exists (thin evidence).

Sources: OpenAI Cookbook (fine-tune-transformers, openai-harmony, handle-raw-cot), Unsloth gpt-oss
fine-tune + RL tutorials + GSPO docs, HF TRL, chat-template bug reports (gpt-oss-120b#69,
gpt-oss-20b#160), train_on_responses_only issues (unsloth #823/#1017/disc#2828).

## 2. INFERENCE performance (Ampere A6000/A5000)

- **MXFP4 on Ampere dequantizes to bf16** - no FP4 speed win (native FP4 tensor cores are
  Blackwell-only; Hopper via Triton). You get the disk/VRAM savings (~13GB), bf16-class speed.
- **llama.cpp is the smoothest primary server:** native MXFP4 GGUF, no crashes, best single-stream.
  `llama-server -ngl 99 --jinja -fa -ub 2048 -b 2048 --ctx-size <large> --temp 1.0 --top-p 1.0
  --top-k 0`. ~110-140 tok/s decode expected (3090-proxy; NO A6000 benchmark exists - thin).
- **vLLM works on Ampere with a RECENT STABLE build + `VLLM_ATTENTION_BACKEND=TRITON_ATTN` + Marlin
  MXFP4** (our earlier crash was a 0.10.x-gptoss preview / 0.24 warmup issue). Flags:
  `--tool-call-parser openai --enable-auto-tool-choice --enable-prefix-caching`.
- **Speculative decoding via MTP: FALSE PREMISE - gpt-oss ships no draft head.** Needs external
  EAGLE3 heads (Hopper-measured) or a trained draft; no smaller same-family sibling for the 20B.
  Do not count on it.
- **Do NOT tensor-parallel A6000+A5000** (heterogeneous; caps to the smaller card). Run one full
  replica per GPU behind a router (data parallel). For one pooled context, llama.cpp
  `--tensor-split 48,24` handles heterogeneous cards better than vLLM.
- **Highest-ROI throughput lever = prefix caching** (shared system prompt + tool schemas dominate
  agentic traffic). KV quant via llama.cpp `--cache-type-k q8_0 --cache-type-v q4_0` (needs -fa);
  fp8 e4m3 KV FAILS on sm_86. FA3/FlashInfer are Hopper-only.
- **Keep MXFP4** - Q8_0/bf16 "upgrades" buy ~nothing and one report saw higher refusal.

Sources: llama.cpp #15396, vLLM recipes/parallelism/structured-output docs, vLLM issues
#22414/#22502 (Ampere MXFP4 crash), #7714 (fp8 KV), Unsloth gpt-oss docs, OpenAI model card
(arXiv 2508.10925), Snowflake/Red Hat spec-decode blogs.

## 3. Harness RELIABILITY (make tool-calling bulletproof)

- **Reliability is dominated by harness correctness, not raw model skill.** gpt-oss tool-use is
  middling: tau-Bench Retail 54.8% (high), and NON-MONOTONIC - 20b Airline WORSE at high
  (42.6 med -> 38.0 high): over-reasoning hurts tools. Default **reasoning_effort MEDIUM** for tool
  loops; low for tight latency; high only for genuinely hard sub-tasks.
- **Sampling: temp 1.0, top_p 1.0, top_k 0, min_p 0, no penalties - OVERRIDE server defaults**
  (llama-server defaults top-k 40 / min-p 0.1; HF top_k 50). No source recommends low temp for tools.
- **Guarantee valid calls by constraining ONLY the JSON args** (between <|message|> and <|call|>),
  NOT the CoT: vLLM XGrammar `structural_tag`, or llama.cpp `json_schema`/`response_format`. Also
  DESCRIBE the schema in the prompt (llama.cpp does not inject it). XGrammar-2 structural_tag = 100%
  schema accuracy on tool calls with improved BFCL-V3 output accuracy.
- **Multi-turn CoT rule:** preserve `analysis` back to the last `final` WHILE a tool sequence is in
  flight; drop it once a `final` is produced; never show analysis to users; replace trailing
  `<|return|>` with `<|end|>` when persisting turns.
- **Server-side harmony parser** (`--tool-call-parser openai --reasoning-parser openai_gptoss` on
  vLLM; `--jinja` on llama.cpp) so the app never touches raw harmony tokens. Validator-first
  execution + retry/repair loop (LangGraph `ToolNode(handle_tool_errors=True)`; manual in Agents
  SDK). Avoid `strict=True` on Ollama gpt-oss.
- **No published schema-validity/executable-call rate exists for gpt-oss** - measuring ours
  (valid-JSON %, schema-valid %, dispatched %) across effort levels would be novel.

Sources: OpenAI model card (arXiv 2508.10925 / tau-Bench tables), harmony cookbook + handle-raw-cot,
vLLM structured-outputs docs, XGrammar-2 blog, llama.cpp grammars README + #22314/#15341/#15789,
"Say What You Mean" (dottxt) rebuttal of format-restriction-hurts-reasoning, framework bug reports
(ollama #11704, vllm #26967, langchain #32428/#34144, langgraph #1153).

## 4. Granite-style TASK ADAPTERS

- IBM ships many small swappable LoRA "intrinsics" over ONE frozen base (RAG library: query-rewrite,
  context-relevance, answerability, uncertainty, hallucination-detection, citation), chained.
- **Activated-LoRA (aLoRA)** adapts only tokens after an invocation sentinel -> reuses the base KV
  cache, hot-swaps without recomputing the prefix (~20-30x faster/call). BUT attention-only,
  higher rank (r~32), NOT interchangeable with LoRA, and **unproven on gpt-oss MoE+MXFP4**; full
  cross-model KV-reuse serving needs a custom vLLM fork -> **DEFER aLoRA to phase 2.**
- **Standard LoRA + vLLM multi-LoRA hot-swap works today:** `--enable-lora --max-loras N
  --max-lora-rank R`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, load/unload via
  `/v1/load_lora_adapter`; Punica SGMV batches heterogeneous adapters in one GEMM.
- Granite training recipe: PEFT LoRA r=8, lr 1e-5, 90/10 split, synthetic+filtered data, targets =
  the STRUCTURED task output.
- **Replicate for gpt-oss (2 adapters over the frozen base):** A "tool-call reliability" (hard
  negatives: malformed JSON, wrong tool, hallucinated name, no-tool-needed; harmony targets),
  B "incident-diagnosis" (structured root-cause + calibrated confidence + abstention). Serve both via
  one base with `--enable-lora --max-loras 2`, route per request.

Sources: aLoRA paper (arXiv 2504.12397), multi-adapter serving (2512.17910), RAG intrinsics
(2504.11704), IBM/activated-lora GitHub, HF ibm-granite collections, vLLM LoRA docs, Punica,
Unsloth LoRA hot-swap guide.

## Net corrections to the original plan
- DROP speculative decoding via MTP (no draft head on gpt-oss).
- vLLM is viable on Ampere with a RECENT STABLE build + TRITON_ATTN (earlier crash was a bad build) -
  which strengthens the case for vLLM as the same-path eval + multi-LoRA serving foundation.
- Fix sampling (temp 1.0/top_p 1.0/top_k 0) - the earlier harness likely mis-sampled.
- Constrain only JSON args, not CoT.
- Defer aLoRA; start with standard LoRA multi-adapter.
