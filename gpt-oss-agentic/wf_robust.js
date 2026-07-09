export const meta = {
  name: 'gpt-oss-robust-adapterA',
  description: 'Council-de-risked gpt-oss-20b build: FIRST prove a non-floored same-path eval foundation (vLLM+guided primary, transformers+grammar fallback) as a GATE, then robustly SFT Adapter A (tool-call reliability) with a VERIFIED assistant-only mask, then a VALID base-vs-adapter before/after (solved + executable-call rate) on that same path, with an inference-perf bench in parallel. GRPO/AdapterB/aLoRA/MTP explicitly deferred.',
  phases: [
    { title: 'Foundation', detail: 'GATE: stand up same-path eval, prove base is NON-FLOORED on the incident env' },
    { title: 'TrainA', detail: 'robust SFT Adapter A (tool-call reliability), assistant-only mask VERIFIED by decoding' },
    { title: 'EvalBench', detail: 'valid base-vs-AdapterA before/after (same path) + inference-perf bench in parallel' },
    { title: 'Commit', detail: 'findings, honest verdict, commit+push' },
  ],
}

const REPO = '/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench'
const OPS = REPO + '/gpt-oss-agentic'
const COMMON = [
  'Repo ' + REPO + ' on git branch gpt-oss-agentic; all new files under ' + OPS + '/. Venvs: ' + REPO + '/.venv (torch2.12/trl/peft/bitsandbytes/datasets), ' + REPO + '/.venv-unsloth (Unsloth gpt-oss QLoRA, PROVEN to train gpt-oss-20b on the A6000), ' + REPO + '/.venv-vllm (vLLM 0.24 - may need a version change).',
  'GPUS: GPU0=A5000 24GB, GPU1=A6000 48GB, both free, 90GB RAM. Always CUDA_VISIBLE_DEVICES=<n> CUDA_DEVICE_ORDER=PCI_BUS_ID. Never two agents on the same GPU at once. Manage servers via subprocess handle+SIGINT; NEVER pkill -f on a pattern matching your own shell (exit 144). Long GPU jobs: nohup + log + until-grep poll (a bash call caps ~10min).',
  'REUSE: ' + OPS + '/incident_sim.py + incident_harness.py (executable incident-response env, 24 verifiable scenarios, anti-brute-force, pluggable chat(messages,tools)->message backend). ' + OPS + '/sft_trajectories.json (520 harmony trajectories), gen_train_data.py, train_gptoss.py, eval_incident_gguf.py, eval_incident_adapter.py. ' + REPO + '/models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf (llama.cpp baseline scored 11/24). ' + OPS + '/RESEARCH_BRIEFS.md + ROBUST_PLAN.md (READ THESE - they carry the cited recipe).',
  'RESEARCH FACTS (do not re-derive): (a) The prior SFT degenerated because of a MISSING assistant-only loss mask (loss over developer/tool-schema tokens -> it learned to reproduce the schema). Fix: train_on_responses_only with the <|start|>assistant boundary; VERIFY by decoding one masked batch (the non -100 tokens must be assistant analysis+commentary+final only) BEFORE training. (b) Tool calls MUST render on the commentary channel: <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>, tools declared in the developer message; strip prior-turn analysis; SAME harmony renderer at train and eval. (c) 1 epoch (~32 steps), early stop, target loss 0.3-0.8 NOT <0.05; success = FREE-RUNNING held-out valid-tool-call rate, not teacher-forced loss. (d) LoRA r16 alpha16 all-linear + target_parameters mlp.experts.gate_up_proj/down_proj (layers 7/15/23), lr 1e-4-2e-4. (e) Sampling MUST be temp 1.0 top_p 1.0 top_k 0 min_p 0 - override llama-server defaults; reasoning_effort MEDIUM for tool loops (high can HURT). (f) Constrain ONLY the JSON args, not the CoT. (g) MXFP4 dequants to bf16 on Ampere (no FP4 speed win); gpt-oss has NO MTP head (no spec-decode-via-MTP).',
  'NO emojis. Your final message is DATA for the orchestrator: concise, structured, exact numbers and file paths.',
].join('\n')

const GATE_SCHEMA = {
  type: 'object',
  properties: {
    gate: { type: 'string', enum: ['GO', 'NOGO'] },
    path: { type: 'string', description: 'the working eval path: vllm | transformers | none' },
    base_solved: { type: 'integer', description: 'incident scenarios solved by BASE gpt-oss-20b on this path' },
    base_total: { type: 'integer' },
    non_floored: { type: 'boolean', description: 'true if base baseline is clearly non-zero (not floored)' },
    notes: { type: 'string' },
  },
  required: ['gate', 'path', 'base_solved', 'base_total', 'non_floored', 'notes'],
}

phase('Foundation')
const found = await agent(COMMON + '\n\n' + [
  'You are the FOUNDATION agent (GPU 1 / A6000). This is the GATE - the whole build depends on it. The prior run failed because the eval harness floored the BASE model to 0/24 on unconstrained transformers even though its GGUF twin scored 11/24. Your job: establish a SAME-PATH eval that reproduces a NON-FLOORED base baseline on the incident env, using the exact serving path the adapter will later use.',
  'Try paths in this order and STOP at the first that works:',
  'PATH 1 (preferred) recent-stable vLLM + guided decoding: in ' + REPO + '/.venv-vllm, ensure a RECENT STABLE vLLM (if the installed 0.24 crashes on the triton warmup, pip install a different recent stable version). Serve openai/gpt-oss-20b (or the local ' + REPO + '/models/gptoss20b if it maps) with CUDA_VISIBLE_DEVICES=1, VLLM_ATTENTION_BACKEND=TRITON_ATTN, --tool-call-parser openai --enable-auto-tool-choice --enable-prefix-caching --gpu-memory-utilization 0.85. Wire an OpenAI-endpoint chat() (temp 1.0 top_p 1.0 top_k 0, reasoning_effort medium) into incident_harness. If vLLM cannot be stabilized in a reasonable effort, move to PATH 2.',
  'PATH 2 (fallback) transformers + PEFT + grammar: load gpt-oss-20b (Unsloth FastLanguageModel 4-bit or transformers), generate with a grammar/structured constraint on the JSON args only (lm-format-enforcer or outlines), temp 1.0. This keeps the adapter unmerged and, crucially, GRAMMAR PREVENTS the flooring the prior unconstrained run hit.',
  'Do NOT use merge->GGUF (MXFP4 merge is untested; the council ruled it out for training conclusions).',
  'Whichever path comes up: run the BASE gpt-oss-20b on ALL 24 incident scenarios and report solved/24. The GATE PASSES only if the base is clearly NON-FLOORED (solved >= 6, ideally near the 11/24 llama.cpp reference). Write ' + OPS + '/eval_same_path.py (the reusable same-path evaluator: takes an optional adapter path, else base) and append the base row to ' + OPS + '/incident_scores.json labeled "gpt-oss-20b base (<path> same-path)". Tear down any server via SIGINT when done (free the A6000).',
  'Return the structured gate. If NOGO (both paths floor the base or crash), set gate=NOGO with the exact blocker in notes - we STOP rather than train blindly.',
].join(' '), { label: 'foundation-gate(gpu1)', phase: 'Foundation', schema: GATE_SCHEMA })

log('FOUNDATION gate=' + (found && found.gate) + ' path=' + (found && found.path) + ' base=' + (found && found.base_solved) + '/' + (found && found.base_total) + ' non_floored=' + (found && found.non_floored))

if (!found || found.gate !== 'GO' || !found.non_floored) {
  phase('Commit')
  const stop = await agent(COMMON + '\n\n' + [
    'The FOUNDATION gate did NOT pass: ' + JSON.stringify(found) + '.',
    'Per the council, we do NOT train on a floored eval path. Write ' + OPS + '/ROBUST_RESULTS.md documenting: the eval-path attempts and why each floored/crashed, that training is correctly HELD until a non-floored same-path eval exists, and the exact next action to unblock (e.g. specific vLLM version, or the grammar backend to fix). Then git add -A && git commit -m "Foundation gate: same-path eval not yet non-floored - training held (honest stop)" && git push origin gpt-oss-agentic. Confirm branch pushed + tree clean.',
    'Return: the blocker, the unblock action, and commit status.',
  ].join(' '), { label: 'stop+document', phase: 'Commit' })
  return { gate: found, outcome: 'STOPPED-at-gate', stop }
}

phase('TrainA')
const trainA = await agent(COMMON + '\n\n' + [
  'FOUNDATION passed: same-path eval = ' + JSON.stringify(found) + '. Now train ADAPTER A = "tool-call reliability" via Unsloth (' + REPO + '/.venv-unsloth, GPU 1 / A6000).',
  'STEP 1 - DATA: build ' + OPS + '/adapterA_data.py that produces harmony SFT data for reliable tool-calling from incident_sim + sft_trajectories: (i) the correct expert tool-call turns rendered on the COMMENTARY channel exactly as the inference path expects; (ii) HARD NEGATIVES turned into positives - for prompts where a naive model emits malformed JSON / wrong tool / hallucinated tool name / calls-when-no-tool-needed, include the CORRECT target (well-formed call, or an abstention when no tool is needed). Hold out 10% for eval. Print sizes.',
  'STEP 2 - MASK VERIFICATION (MANDATORY before training): write ' + OPS + '/train_adapterA.py using train_on_responses_only with the <|start|>assistant boundary. DECODE one masked batch and PRINT the tokens that carry loss (non -100). Confirm they are ONLY assistant analysis+commentary+final and that the developer/tools/user/tool-result tokens are masked. If the mask is wrong, fix the boundary strings until the decode is correct. Do NOT proceed to training until the printed mask is correct - include the decoded proof in your return.',
  'STEP 3 - SFT: LoRA r16 alpha16 all-linear + target_parameters (mlp.experts.gate_up_proj/down_proj layers 7/15/23), lr 1e-4, max_seq 2048, 1 epoch (early stop; STOP if train loss drops below 0.1 - that is memorization). Save adapter ' + OPS + '/adapters_gptoss/adapterA. After training, run FREE-RUNNING generation on the 10% held-out prompts and report the valid-tool-call rate (parseable commentary-channel call) - THIS is the success metric, not loss.',
  'Self-repair on Unsloth API errors (max 3 tries). Return: dataset sizes, the DECODED MASK PROOF, final train loss, held-out valid-tool-call rate, and the adapter path.',
].join(' '), { label: 'train-adapterA(gpu1)', phase: 'TrainA' })

phase('EvalBench')
const eb = await parallel([
  () => agent(COMMON + '\n\n' + [
    'FOUNDATION path = ' + JSON.stringify(found) + '. Adapter A trained: ' + (typeof trainA === 'string' ? trainA.slice(0, 400) : JSON.stringify(trainA).slice(0, 400)) + '.',
    'You are the VALID BEFORE/AFTER agent (GPU 1 / A6000). Using ' + OPS + '/eval_same_path.py (the SAME path that produced the non-floored base baseline), evaluate BASE vs ADAPTER A (' + OPS + '/adapters_gptoss/adapterA) on all 24 incident scenarios. This is the apples-to-apples comparison the prior run lacked.',
    'Report for BOTH base and Adapter A on the SAME path: solved/24, correct-root-cause rate, avg steps, AND the novel reliability metrics - executable-call rate (valid-JSON %, schema-valid %, dispatched %) and invalid-call rate (with the JSON-arg grammar on). Append both rows to ' + OPS + '/incident_scores.json labeled "... (adapterA same-path)". SANITY: the base row here must match the FOUNDATION base baseline (non-floored) - if it floors, the comparison is invalid; say so.',
    'Return: the before/after table (base vs Adapter A) with solved + executable-call-rate deltas, and whether the base stayed non-floored (comparison valid).',
  ].join(' '), { label: 'before-after(gpu1)', phase: 'EvalBench' }),

  () => agent(COMMON + '\n\n' + [
    'You are the INFERENCE-PERF agent (GPU 0 / A5000 - do NOT touch GPU 1). Benchmark gpt-oss-20b serving on the A5000 and report concrete numbers. Write ' + OPS + '/bench_inference.py.',
    'Measure with the CORRECT sampling (temp 1.0 top_p 1.0 top_k 0 min_p 0): (1) llama.cpp: llama-server ' + REPO + '/models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf -ngl 99 --jinja -fa -ub 2048 -b 2048 --ctx-size 8192 - measure decode tok/s and TTFT on a few agentic-style prompts, and the effect of --cache-type-k q8_0. (2) reasoning_effort low vs medium vs high: tokens generated + total latency (expect TTFT ~unchanged, total scales with CoT). (3) prefix-caching / prompt-cache effect on repeated system+tool-schema prefixes if measurable.',
    'Do NOT attempt MTP speculative decoding (gpt-oss has no draft head). Do NOT tensor-parallel across the two GPUs.',
    'Return: a small table of tok/s + TTFT for llama.cpp, the reasoning-effort latency/token tradeoff, and the single highest-ROI inference recommendation for an agentic server on this hardware.',
  ].join(' '), { label: 'inference-bench(gpu0)', phase: 'EvalBench' }),
])

phase('Commit')
const report = await agent(COMMON + '\n\n' + [
  'FOUNDATION: ' + JSON.stringify(found) + '.',
  'TRAIN A: ' + (typeof trainA === 'string' ? trainA : JSON.stringify(trainA)) + '.',
  'BEFORE/AFTER: ' + (typeof eb[0] === 'string' ? eb[0] : JSON.stringify(eb[0])) + '.',
  'INFERENCE BENCH: ' + (typeof eb[1] === 'string' ? eb[1] : JSON.stringify(eb[1])) + '.',
  'TASK: write ' + OPS + '/ROBUST_RESULTS.md and commit. Cover: (1) the same-path eval FOUNDATION and the non-floored base baseline (the fix for the prior confound); (2) Adapter A (tool-call reliability) - the DECODED MASK PROOF, training regime, held-out valid-tool-call rate; (3) the VALID before/after table (base vs Adapter A, same path) with solved + executable-call-rate deltas, and the honest verdict on whether Adapter A improved tool-call reliability / task success (report whichever way it falls - a null or negative result is fine if that is what the data shows, stated plainly); (4) the inference-perf numbers + recommendation; (5) explicitly list what was DEFERRED and why (GRPO until SFT+eval proven, Adapter B incident-diagnosis, aLoRA, MTP spec-decode false premise). Update ' + REPO + '/LEARNINGS.md experiment 8 with the corrected same-path result. NO emojis.',
  'Then git add -A && git commit -m "Robust gpt-oss-20b Adapter A: same-path eval + verified mask + valid before/after" && git push origin gpt-oss-agentic. Confirm branch pushed + tree clean.',
  'Return: the final before/after table, the honest verdict, the inference recommendation, and commit status.',
].join(' '), { label: 'findings+commit', phase: 'Commit' })

return { gate: found, trainA: (typeof trainA === 'string' ? trainA.slice(0, 400) : trainA),
         beforeAfter: eb[0], inference: eb[1], report }
