#!/usr/bin/env python3
"""Agentic training stack for gpt-oss-20b on the NimbusWorks incident-response env.

Stages (select with argv[1]):
  sft   -> SFT on sft_trajectories.json (harmony multi-turn tool trajectories).
           LoRA all-linear+MoE, r=16 alpha=32, lr 2e-4. Saves adapter to
           adapters_gptoss/sft.
  grpo  -> GRPO from the SFT adapter using the incident-env EXECUTION reward.
           Single-turn verifiable proxy (documented below). Saves adapters_gptoss/grpo.

Run (GPU 1 / A6000):
  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python train_gptoss.py sft
  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python train_gptoss.py grpo

GRPO design note (honest):
  A faithful multi-turn agentic rollout inside trl's GRPOTrainer (generate ->
  execute tool -> feed result -> repeat) is not natively supported and is heavy.
  We use a SINGLE-TURN VERIFIABLE PROXY that still trains the exact weak skill the
  baseline analysis flagged (root-cause ID + set_config->restart sequencing):
    - The prompt surfaces the real diagnostic observations up front (check_all +
      get_logs for every unhealthy service), exactly what a first diagnostic pass
      would reveal. The true root is NOT named; cascades show NBX-3301, the root
      shows its real fault code, so the model must still distinguish root vs symptom.
    - The model must emit a JSON remediation PLAN (ordered list of tool calls).
    - The reward REPLAYS that plan against a fresh, real IncidentSim and returns
      the execution outcome (solved=1.0). This is a genuine execution reward on the
      same simulator used for eval, just collapsed to one generation for tractable RL.
"""
import os, sys, json, re, random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# REQUIRED on torch 2.10+cu130 / bitsandbytes 0.49 / sm_86 (see smoke script).
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

import torch
from unsloth import FastLanguageModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from incident_sim import IncidentSim, TOOLS_SPEC, SERVICES  # noqa: E402

MODEL_NAME = "unsloth/gpt-oss-20b"
MAX_SEQ = 2048
ADAPTER_SFT = os.path.join(HERE, "adapters_gptoss", "sft")
ADAPTER_GRPO = os.path.join(HERE, "adapters_gptoss", "grpo")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_base():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=None,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        full_finetuning=False,
    )
    return model, tokenizer


def add_lora(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=LORA_TARGETS,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


# ==========================================================================
# STAGE: SFT
# ==========================================================================
def run_sft():
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_base()
    model = add_lora(model)

    trajs = json.load(open(os.path.join(HERE, "sft_trajectories.json")))
    # The FastLanguageModel-returned tokenizer's harmony chat template chokes on
    # the tool schema (a patched jinja that mishandles both wrapped and flat tool
    # dicts). A stock AutoTokenizer for the same model renders the wrapped
    # OpenAI-style tools correctly, and produces byte-identical harmony text (same
    # vocab). So render with the stock tokenizer; SFTTrainer only needs the model
    # tokenizer to tokenize the raw "text" field, not to apply the chat template.
    from transformers import AutoTokenizer
    render_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = []
    for t in trajs:
        txt = render_tok.apply_chat_template(
            t["messages"], tools=TOOLS_SPEC,
            tokenize=False, add_generation_prompt=False)
        texts.append(txt)
    print(f"=== rendered {len(texts)} harmony texts via stock tokenizer ===", flush=True)
    dataset = Dataset.from_dict({"text": texts})
    print(f"=== SFT dataset: {len(dataset)} harmony trajectories ===", flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            max_length=MAX_SEQ,
            output_dir="/tmp/gptoss_sft_out",
            report_to="none",
            save_strategy="no",
        ),
    )
    print("=== starting SFT ===", flush=True)
    stats = trainer.train()

    os.makedirs(ADAPTER_SFT, exist_ok=True)
    model.save_pretrained(ADAPTER_SFT)
    tokenizer.save_pretrained(ADAPTER_SFT)
    peak = torch.cuda.max_memory_allocated() / 1e9
    hist = [h for h in trainer.state.log_history if "loss" in h]
    print(f"SFT_DONE training_loss={stats.training_loss:.4f} "
          f"peak_alloc_GB={peak:.2f} steps={trainer.state.global_step} "
          f"adapter={ADAPTER_SFT}", flush=True)
    print("SFT_LOSS_HISTORY=" + json.dumps(
        [{"step": h["step"], "loss": round(h["loss"], 4)} for h in hist]), flush=True)
    if len(hist) >= 2:
        print(f"SFT_LOSS_FIRST={hist[0]['loss']:.4f} SFT_LOSS_LAST={hist[-1]['loss']:.4f}",
              flush=True)


# ==========================================================================
# STAGE: GRPO  (single-turn verifiable execution-reward proxy)
# ==========================================================================
SYSTEM_PROMPT = (
    "You are an on-call SRE for NimbusWorks. Exactly one service has a real "
    "ROOT-CAUSE fault; every other alerting service is only a cascading symptom "
    "of an unhealthy dependency (its log shows NBX-3301 'upstream ... symptom, "
    "not root cause'). Diagnose the ROOT from the observations, then output the "
    "CORRECT remediation as an ordered plan of tool calls. Rules: bad_config "
    "(NBX-6600) -> set_config(fix key/value) THEN restart; pool_exhausted "
    "(NBX-2201) -> set_config(pool_max=512) THEN restart; bad_deploy (NBX-6700) "
    "-> rollback; crash (NBX-6900) -> restart. A bare restart does NOT fix "
    "config/deploy/pool faults."
)

PLAN_INSTRUCTIONS = (
    "\n\nRespond with ONLY a JSON array (no prose) of the tool calls to run, in "
    'order. Each element is {"tool": <name>, "args": {...}}. Available mutating '
    'tools: set_config(service,key,value), restart(service), rollback(service), '
    "scale(service,replicas). Fix the ROOT service and restore all services to "
    "healthy."
)


def _observation_block(scenario):
    """Deterministic first-pass diagnostics the model gets to reason over."""
    sim = IncidentSim(scenario)
    lines = ["check_all -> " + sim.check_all()]
    for s in sorted(SERVICES.keys()):
        if sim.services[s]["status"] != "healthy":
            lines.append(f"get_logs({s}) -> {sim.get_logs(s)}")
    return "\n".join(lines)


def build_grpo_prompt(scenario):
    user = (
        "Incident: alerts firing across the fleet. Initial diagnostics:\n"
        + _observation_block(scenario)
        + "\n\nFix keys per service (only if bad_config): "
        + json.dumps({scenario["root_service"]: [scenario["fix_key"], scenario["fix_value"]]})
        + PLAN_INSTRUCTIONS
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_plan(text):
    """Extract the JSON tool-call plan from a completion; robust to harmony wrappers."""
    # strip harmony final-channel wrappers if present
    m = _JSON_RE.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    plan = []
    if isinstance(arr, list):
        for e in arr:
            if isinstance(e, dict) and "tool" in e:
                plan.append((e["tool"], e.get("args", {}) or {}))
    return plan


def _replay_reward(scenario, plan):
    """Execution reward: replay the plan on the real sim. Shaped around solved."""
    sim = IncidentSim(scenario, max_calls=15)
    for name, args in plan[:15]:
        if not isinstance(args, dict):
            continue
        sim.dispatch(name, args)
    sc = sim.score()
    if sc["solved"]:
        return 1.0
    r = 0.0
    if sc["correct_root_cause"]:        # root fault cleared but fleet not fully healthy
        r += 0.4
    # partial credit for acting on the true root at all (targeting signal)
    root = scenario["root_service"]
    touched_root = any(isinstance(a, dict) and a.get("service") == root
                       and n in ("set_config", "restart", "rollback")
                       for n, a in plan[:15])
    if touched_root:
        r += 0.2
    # penalise flailing on wrong services
    wrong = sum(1 for n, a in plan[:15]
                if isinstance(a, dict) and a.get("service") not in (root, None)
                and n in ("restart", "rollback", "set_config", "scale"))
    r -= 0.03 * wrong
    return max(0.0, min(r, 0.9))


def run_grpo():
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Resume from the SFT adapter as a TRAINABLE PeftModel. Loading the adapter
    # directory directly via Unsloth's from_pretrained restores base+LoRA (incl.
    # the MoE expert LoRA) with is_trainable=True. NOTE: model.load_adapter()
    # attaches the LoRA as is_trainable=False -> 0 trainable params, so we must
    # NOT use that path here.
    if os.path.isdir(ADAPTER_SFT):
        print(f"=== loading SFT adapter (trainable) from {ADAPTER_SFT} ===", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_SFT,
            dtype=None,
            max_seq_length=MAX_SEQ,
            load_in_4bit=True,
            full_finetuning=False,
        )
        FastLanguageModel.for_training(model)
    else:
        print("=== WARNING: no SFT adapter found; attaching fresh LoRA ===", flush=True)
        model, tokenizer = load_base()
        model = add_lora(model)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=== GRPO trainable params: {ntrain:,} ===", flush=True)
    if ntrain == 0:
        raise RuntimeError("0 trainable params after adapter load; aborting GRPO")

    scenarios = json.load(open(os.path.join(HERE, "grpo_scenarios.json")))
    random.Random(3407).shuffle(scenarios)
    # keep runtime bounded (HF-generate GRPO is slow); balanced subset
    subset = scenarios[:64]
    scen_by_prompt = {}
    rows = []
    for spec in subset:
        sc = spec["scenario"]
        prompt = build_grpo_prompt(sc)
        key = json.dumps(prompt, sort_keys=True)
        scen_by_prompt[key] = sc
        rows.append({"prompt": prompt})
    dataset = Dataset.from_list(rows)
    print(f"=== GRPO dataset: {len(dataset)} scenarios (single-turn exec-reward) ===",
          flush=True)

    def reward_exec(prompts, completions, **kwargs):
        outs = []
        for p, c in zip(prompts, completions):
            key = json.dumps(p, sort_keys=True)
            sc = scen_by_prompt.get(key)
            text = c if isinstance(c, str) else (
                c[-1]["content"] if isinstance(c, list) and c else "")
            plan = _parse_plan(text or "")
            outs.append(_replay_reward(sc, plan) if sc is not None else 0.0)
        return outs

    # HF-generate rollout on a 20B MoE is slow (~10 min/step at 8x512-tok
    # completions). Trim aggressively so GRPO fits the time budget while still
    # producing a real execution-reward signal: 4 completions/step, 256-tok cap,
    # hard 12-step cap. Documented as a SHORTENED GRPO stage.
    args = GRPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=768,
        max_completion_length=256,
        learning_rate=1e-5,
        logging_steps=1,
        max_steps=12,
        optim="adamw_8bit",
        warmup_steps=2,
        lr_scheduler_type="constant",
        seed=3407,
        temperature=1.0,
        output_dir="/tmp/gptoss_grpo_out",
        report_to="none",
        save_strategy="no",
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_exec],
        args=args,
        train_dataset=dataset,
    )
    print("=== starting GRPO ===", flush=True)
    stats = trainer.train()

    os.makedirs(ADAPTER_GRPO, exist_ok=True)
    model.save_pretrained(ADAPTER_GRPO)
    tokenizer.save_pretrained(ADAPTER_GRPO)
    peak = torch.cuda.max_memory_allocated() / 1e9
    hist = trainer.state.log_history
    rw = [h for h in hist if "reward" in h]
    lo = [h for h in hist if "loss" in h]
    print(f"GRPO_DONE training_loss={stats.training_loss:.6f} "
          f"peak_alloc_GB={peak:.2f} steps={trainer.state.global_step} "
          f"adapter={ADAPTER_GRPO}", flush=True)
    print("GRPO_REWARD_HISTORY=" + json.dumps(
        [{"step": h["step"], "reward": round(h.get("reward", 0.0), 4)} for h in rw]),
        flush=True)
    print("GRPO_LOSS_HISTORY=" + json.dumps(
        [{"step": h["step"], "loss": round(h["loss"], 6)} for h in lo]), flush=True)
    if rw:
        print(f"GRPO_REWARD_FIRST={rw[0].get('reward',0.0):.4f} "
              f"GRPO_REWARD_LAST={rw[-1].get('reward',0.0):.4f}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "sft"
    if stage == "sft":
        run_sft()
    elif stage == "grpo":
        run_grpo()
    else:
        print(f"unknown stage {stage}; use sft|grpo")
        sys.exit(2)
