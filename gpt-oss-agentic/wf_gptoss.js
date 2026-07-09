export const meta = {
  name: 'gpt-oss-agentic-fullstack',
  description: 'Deep-dive gpt-oss-20b on a NEW harder agentic environment (executable incident-response sim, unsaturated, verified end-states), then run the training stack (CPT->SFT-on-trajectories->agentic-GRPO) on gpt-oss-20b via Unsloth with a feasibility gate, bigger data, on the free A6000. Commit to the gpt-oss-agentic branch.',
  phases: [
    { title: 'Recon', detail: 'Unsloth gpt-oss recipe + build harder agentic env + generate bigger datasets (parallel)' },
    { title: 'BaselineFeas', detail: 'baseline gpt-oss on the hard env + Unsloth QLoRA GO/NO-GO smoke (parallel, both GPUs)' },
    { title: 'Train', detail: 'CPT->SFT->agentic-GRPO on gpt-oss via Unsloth (adaptive on the gate), bigger data, A6000' },
    { title: 'EvalCommit', detail: 'before/after on the hard env, findings, commit to branch' },
  ],
}

const REPO = '/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench'
const OPS = REPO + '/gpt-oss-agentic'
const COMMON = [
  'Repo: ' + REPO + ' (currently on git branch gpt-oss-agentic - all new files go under ' + OPS + '/). Training venv ' + REPO + '/.venv (torch 2.12/cu130, transformers 5.13, trl 1.7, peft, bitsandbytes, datasets). llama.cpp bins at ' + REPO + '/llama.cpp/build/bin (updated build, GGUF agentic serving with --jinja).',
  'Existing assets to REUSE: ' + REPO + '/nimbus-agent/world.py (deterministic fictional-company world model: SERVICES dict, ERRORS dict, dependency graph, policies). ' + REPO + '/agentic-harness/mcp_server.py (minimal MCP JSON-RPC stdio server pattern). ' + REPO + '/models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf (works in llama.cpp). ' + REPO + '/models/q9b/Qwen3.5-9B-Q4_K_M.gguf and ' + REPO + '/models/qwythos/qwythos-nomtp-Q4_K_M.gguf.',
  'GPUS: GPU 0 = A5000 24GB, GPU 1 = A6000 48GB (BOTH FREE). 98GB RAM free. Always CUDA_VISIBLE_DEVICES=<n> CUDA_DEVICE_ORDER=PCI_BUS_ID. Never two agents on the same GPU at once. llama-server: manage via subprocess handle + SIGINT (NEVER pkill -f on a pattern matching your own shell - it self-kills, exit 144). Long GPU jobs: nohup to a log + until-grep poll (a single bash call caps around 10 min).',
  'KNOWN TRAINING FACTS (do not re-derive): gpt-oss-20b is MoE+MXFP4; native MXFP4 TRAINING is unsupported; the ONLY QLoRA path on Ampere is UNSLOTH with its LINEARIZED gpt-oss (unsloth/gpt-oss-20b, load in 4-bit), ~14GB VRAM, and Unsloth supports GRPO for gpt-oss. Install in a FRESH venv: uv pip install unsloth --torch-backend=auto (fallback: pip install unsloth). gpt-oss REQUIRES the harmony chat format. Thinking-Machines LoRA guidance: LoRA on ALL-LINEAR incl MoE layers, alpha=32, learning rate ~10x full-FT (about 2e-4 for LoRA), small batch, RL matches full-FT even at low rank.',
  'NO emojis anywhere (files, commits). Your final message is DATA for the orchestrator: concise, structured, exact numbers and paths.',
].join('\n')

phase('Recon')
const recon = await parallel([
  () => agent(COMMON + '\n\n' + [
    'TASK (CPU/web): produce the EXACT, verified Unsloth gpt-oss-20b QLoRA + GRPO recipe for an Ampere A6000 (48GB, compute 8.6, CUDA 13, torch 2.12).',
    'Read the Unsloth docs: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss and the gpt-oss-reinforcement-learning page.',
    'Extract concretely: (a) exact pip/uv install command(s) and the exact model id to load (linearized 4-bit gpt-oss) and FastLanguageModel.from_pretrained args (load_in_4bit, max_seq_length, dtype); (b) get_peft_model / LoRA config for gpt-oss MoE (target_modules incl MoE expert layers, r, alpha, gradient checkpointing); (c) SFT: how to format harmony chat data for gpt-oss + SFTTrainer/SFTConfig args; (d) GRPO: GRPOTrainer/GRPOConfig via Unsloth for gpt-oss, reward_funcs signature, vLLM-or-not for generation; (e) Ampere caveats: does A6000 (sm_86, NOT Hopper) actually work for the linearized QLoRA path, known errors and fixes.',
    'Write ' + OPS + '/UNSLOTH_RECIPE.md with the copy-pasteable recipe. Return: the install command, model id, LoRA config, and the single biggest risk for A6000.',
  ].join(' '), { label: 'recon:unsloth-recipe', phase: 'Recon' }),

  () => agent(COMMON + '\n\n' + [
    'TASK (CPU): BUILD THE NEW HARDER AGENTIC ENVIRONMENT - an executable Incident-Response simulator that is genuinely UNSATURATED (our old 8-task harness hits 8/8 for capable models; this must not). Reuse ' + REPO + '/nimbus-agent/world.py (import its SERVICES/ERRORS/dependency graph).',
    'Write ' + OPS + '/incident_sim.py: a stateful simulator where each service has a state record (status healthy|degraded|down, error_rate, replicas, config_ok flag, deployed_bad flag). A SCENARIO injects a ROOT-CAUSE fault at ONE service (bad config, bad deploy, dependency outage, pool exhausted) that CASCADES to dependents (a downstream service appears down because its dependency is the real cause).',
    'TOOLS (python callables with docstrings, also exposed MCP-style): get_status(service), get_logs(service) returns the error CODE hinting the cause, get_dependencies(service), restart(service), scale(service,n), rollback(service), set_config(service,key,value), check_all(). Tools mutate/read sim state.',
    'The CORRECT fix depends on fault TYPE: bad config -> set_config+restart; bad deploy -> rollback; pool exhausted -> set_config(pool_max)+restart; dependency outage -> fix the DEPENDENCY not the symptom. ANTI-BRUTE-FORCE: blind restart-everything must NOT fix config/deploy faults; cap tool calls (about 15) and penalize restart-looping.',
    'SCORING: a scenario is SOLVED only if after the actions ALL services are healthy AND the correct root-cause fix was applied within budget. Return per-scenario {solved, steps, correct_root_cause, redundant_calls}. build_scenarios(n): about 24 diverse verifiable scenarios (fault type x service x cascade depth), deterministic seed. expert_trajectory(scenario): the gold diagnose->fix->verify tool sequence (for SFT).',
    'Put the multi-turn agent loop (model emits tool call, execute, feed result, repeat, verify) in ' + OPS + '/incident_harness.py taking a chat(messages, tools) callable that returns a message, so backends (OpenAI endpoint for GGUF, or local transformers/Unsloth generate for an adapter) just supply chat().',
    'Smoke-test (CPU, no model): run 3 scenarios with the EXPERT trajectory as the agent and confirm solved=True; run a dumb restart-everything agent and confirm it does NOT solve config/deploy faults (proves anti-brute-force).',
    'Return: file paths, the 3 expert-solved + brute-force-fails smoke numbers, and one example scenario (fault, correct fix, expert steps).',
  ].join(' '), { label: 'recon:build-env', phase: 'Recon' }),
])

log('Recon done')

phase('BaselineFeas')
const bf = await parallel([
  () => agent(COMMON + '\n\n' + [
    'Recon built ' + OPS + '/incident_sim.py + ' + OPS + '/incident_harness.py (executable incident-response env, about 24 verifiable scenarios, pluggable chat() backend, anti-brute-force scoring).',
    'You are the GPU-1 (A6000) BASELINE agent. Write ' + OPS + '/eval_incident_gguf.py that: given (label, gguf_path, port), starts llama-server (CUDA_VISIBLE_DEVICES=1, --jinja) on that port, wires an OpenAI-endpoint chat() into incident_harness, runs ALL scenarios, prints solved/total + avg steps + correct-root-cause rate + redundant-call rate, appends to ' + OPS + '/incident_scores.json, kills the server via subprocess SIGINT (never pkill).',
    'Run the DEEP-DIVE baseline on gpt-oss-20b FIRST: ' + REPO + '/models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf on port 18470. Then ALSO baseline Qwen3.5-9B (' + REPO + '/models/q9b/Qwen3.5-9B-Q4_K_M.gguf) and Qwythos-9B (' + REPO + '/models/qwythos/qwythos-nomtp-Q4_K_M.gguf, temp 0.6) on the SAME env, port 18470, sequential (one server at a time on GPU-1).',
    'CRITICAL: this env must NOT saturate - if gpt-oss solves ALL scenarios, the env is too easy; report that so we can harden it. We WANT an unsaturated score that leaves room to improve via training.',
    'Return: incident-response solved/total (and root-cause rate, avg steps) for gpt-oss-20b, Qwen3.5-9B, Qwythos - the deep-dive numbers - and whether the env is appropriately hard (not saturated).',
  ].join(' '), { label: 'baseline:gpt-oss-hard-env(gpu1)', phase: 'BaselineFeas' }),

  () => agent(COMMON + '\n\n' + [
    'Recon produced ' + OPS + '/UNSLOTH_RECIPE.md and the incident env. Do TWO things.',
    '(1) GENERATE BIGGER TRAINING DATA (CPU): write ' + OPS + '/gen_train_data.py that builds from incident_sim + world.py: cpt_corpus.json (ops-domain corpus: service handbooks, runbooks per error code, dependency facts, policies - a few hundred docs); sft_trajectories.json (expert incident-resolution TRAJECTORIES as harmony/chat conversations: user symptom -> assistant reasons -> tool calls -> tool results -> verified fix, over 500+ scenario instances, formatted as messages arrays with tool-call turns); grpo_scenarios.json (about 300 scenario specs for agentic-GRPO with execution reward = solved). Run it, print sizes.',
    '(2) UNSLOTH FEASIBILITY SMOKE on the A6000 (the GO/NO-GO gate): create a FRESH venv ' + REPO + '/.venv-unsloth, install per UNSLOTH_RECIPE.md (uv pip install unsloth --torch-backend=auto; if uv missing, pip install uv then that, or pip install unsloth). Write+run ' + OPS + '/smoke_unsloth_gptoss.py: CUDA_VISIBLE_DEVICES=1, FastLanguageModel.from_pretrained the linearized 4-bit gpt-oss-20b (max_seq_length 1024, load_in_4bit=True), add LoRA (all-linear incl MoE, r=8, alpha=16), a 3-step SFTTrainer run on 8 tiny harmony examples, confirm loss prints and no crash.',
    'Return CLEARLY: dataset sizes; and the FEASIBILITY VERDICT as the FIRST line of your response: exactly "GATE: GO" if the Unsloth gpt-oss 3-step smoke trained without crashing (give the loss + VRAM used), or "GATE: NOGO" with the exact error if it failed. This verdict decides whether phase Train runs on gpt-oss or falls back.',
  ].join(' '), { label: 'feas:unsloth-gate+data(gpu0)', phase: 'BaselineFeas' }),
])

const gate = (typeof bf[1] === 'string' && bf[1].includes('GATE: GO')) ? 'GO' : 'NOGO'
log('Feasibility gate for gpt-oss training: ' + gate)

phase('Train')
const trained = await agent(COMMON + '\n\n' + [
  'Baseline+Feasibility results follow.',
  '=== deep-dive baseline (gpt-oss on hard env) ===',
  (typeof bf[0] === 'string' ? bf[0] : JSON.stringify(bf[0])),
  '=== unsloth gate + data ===',
  (typeof bf[1] === 'string' ? bf[1] : JSON.stringify(bf[1])),
  'FEASIBILITY GATE = ' + gate + '.',
  'TASK (GPU 1 / A6000, sequential): run the agentic training stack on the incident-response domain with the BIGGER datasets in ' + OPS + '/ (cpt_corpus.json, sft_trajectories.json, grpo_scenarios.json).',
  'IF GATE == GO: train gpt-oss-20b via Unsloth (' + REPO + '/.venv-unsloth). Write ' + OPS + '/train_gptoss.py with stages by arg: sft = SFT on sft_trajectories (harmony), LoRA all-linear+MoE r=16 alpha=32 lr 2e-4, 1-2 epochs or a few hundred steps (use the free 48GB/98GB: seq len up to 2048, as many steps as fit ~30-40 min), save adapter ' + OPS + '/adapters_gptoss/sft; grpo = agentic-GRPO from the sft adapter using the incident env execution reward (solved=1) over grpo_scenarios via GRPOTrainer (multi-turn tool loop as rollout, or a single-turn verifiable proxy if multi-turn is too heavy - document which), save ' + OPS + '/adapters_gptoss/grpo. Run sft then grpo; self-repair on Unsloth API errors (max 3 tries/stage).',
  'IF GATE == NOGO: document the gpt-oss training blocker in ' + OPS + '/UNSLOTH_RECIPE.md, then run the SAME stack (sft -> agentic-GRPO on the incident env) on Qwen3.5-9B via ' + REPO + '/.venv (QLoRA 4-bit, trl SFTTrainer+GRPOTrainer) so we still demonstrate the full pipeline on the harder problem. Save adapters under ' + OPS + '/adapters_qwen/.',
  'Either way: after training, SAVE the final adapter path and print stage losses.',
  'Return: which model was trained (gpt-oss or Qwen fallback), the stage losses, the final adapter path, and any stage you had to shorten/skip with the honest reason.',
].join(' '), { label: 'train-stack', phase: 'Train' })

phase('EvalCommit')
const report = await agent(COMMON + '\n\n' + [
  'Deep-dive baseline: ' + (typeof bf[0] === 'string' ? bf[0] : JSON.stringify(bf[0])),
  'Training result: ' + (typeof trained === 'string' ? trained : JSON.stringify(trained)),
  'Gate was: ' + gate + '.',
  'TASK: measure BEFORE vs AFTER on the harder incident-response env, write findings, commit to the gpt-oss-agentic branch.',
  '1. Eval the TRAINED adapter on the incident env: write ' + OPS + '/eval_incident_adapter.py that loads base+trained-adapter (Unsloth FastLanguageModel if gpt-oss, else transformers+peft for Qwen), wires a local transformers generate into incident_harness.chat(), runs ALL scenarios, appends {label:"<model> TRAINED", ...} to ' + OPS + '/incident_scores.json (GPU 1). If loading the adapter for generation is blocked, say so and report training losses + baseline as the result.',
  '2. Build the deliverable table from ' + OPS + '/incident_scores.json: solved/total, root-cause rate, avg steps, redundant calls for gpt-oss-20b baseline, Qwen3.5-9B, Qwythos, and the TRAINED model, with the BEFORE->AFTER delta for the trained model.',
  '3. Write ' + OPS + '/GPT_OSS_AGENTIC_FINDINGS.md: the new harder environment (why unsaturated + anti-brute-force), the gpt-oss-20b deep-dive baseline (its agentic capacity on a real unsaturated problem), the training stack that ran (gpt-oss via Unsloth OR the honest gate=NOGO fallback to Qwen with the documented blocker), the before/after delta, and honest caveats (what the delta supports/does not; dataset sizes; any shortened stage). NO emojis.',
  '4. git add -A && git commit -m "gpt-oss deep-dive + harder agentic env + training stack + before/after" && git push -u origin gpt-oss-agentic. Confirm the branch pushed and working tree clean.',
  'Return: the final before/after table, the key delta, the honest verdict on whether training improved gpt-oss agentic capacity on the hard env, and the commit/branch status.',
].join(' '), { label: 'eval+findings+commit', phase: 'EvalCommit' })

return { gate, baseline: (typeof bf[0] === 'string' ? bf[0].slice(0, 600) : bf[0]),
         trained: (typeof trained === 'string' ? trained.slice(0, 600) : trained), report }
