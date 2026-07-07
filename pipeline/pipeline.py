#!/usr/bin/env python3
"""OpenAI-style 7-stage post-training pipeline — PROOF OF FLOW on one A5000.
CPT -> SFT -> Reasoning -> Tool-calling -> MCP -> Preference(DPO) -> Agentic-RL(GRPO).
One LoRA adapter accumulates across stages (each resumes the previous). Tiny slices,
few steps: the goal is a green end-to-end flow, not accuracy. Resumable: a stage with
an existing adapters/<stage>/ dir is skipped. Base = Qwen3-4B (swap for gpt-oss on AWS)."""
import os, sys, json, re, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from datasets import load_dataset, Dataset
from common import load_tokenizer, model_with_adapter, ADAPTERS
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
from agentic_grpo import TOOLS as AGENTIC_TOOLS, build_ds as agentic_ds, success_reward

TOK = load_tokenizer()
LOG = f"{os.path.dirname(os.path.abspath(__file__))}/pipeline_log.json"
RESULTS = json.load(open(LOG)) if os.path.exists(LOG) else {}

def done(stage):  return os.path.exists(f"{ADAPTERS}/{stage}")
def save(model, stage):
    # selected_adapters=["default"] avoids saving trl's auto-created frozen "ref"
    # DPO reference adapter alongside the policy (council finding, version-independent).
    try: model.save_pretrained(f"{ADAPTERS}/{stage}", selected_adapters=["default"])
    except Exception: model.save_pretrained(f"{ADAPTERS}/{stage}")
def record(stage, **kw):
    RESULTS[stage] = kw; json.dump(RESULTS, open(LOG, "w"), indent=2)
    print(f"  >> {stage}: {kw}", flush=True)
def free(model):
    del model; gc.collect(); torch.cuda.empty_cache()

def chat_text(msgs):
    return TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

def sft_config(out, steps):
    return SFTConfig(output_dir=out, max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=5, max_length=1024, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")

def run_sft(stage, prev, ds, steps):
    print(f"\n===== STAGE {stage} (SFT-type, {len(ds)} ex, {steps} steps) =====", flush=True)
    model = model_with_adapter(prev)
    tr = SFTTrainer(model=model, args=sft_config(f"/tmp/{stage}", steps),
                    train_dataset=ds, processing_class=TOK)
    out = tr.train()
    save(model, stage); record(stage, loss=round(out.training_loss, 4), n=len(ds), steps=steps)
    free(model)

# ---------------- Stage datasets ----------------
def ds_cpt():   # 1. Continual pretraining on raw domain text
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    rows = [t for t in d["text"] if len(t) > 250][:256]
    return Dataset.from_dict({"text": rows})

def ds_sft():   # 2. Instruction SFT (harmony-style reasoning chats; OpenAI cookbook set)
    d = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:256]")
    return d.map(lambda e: {"text": chat_text(e["messages"])}, remove_columns=d.column_names)

def ds_reasoning():  # 3. Math CoT (DeepSeek-R1 traces)
    d = load_dataset("open-r1/OpenR1-Math-220k", split="train[:256]")
    cols = d.column_names
    def f(e):
        if "messages" in cols and e["messages"]:
            return {"text": chat_text(e["messages"])}
        prob = e.get("problem") or e.get("question") or ""
        sol = e.get("solution") or e.get("answer") or ""
        return {"text": chat_text([{"role": "user", "content": prob},
                                   {"role": "assistant", "content": str(sol)}])}
    return d.map(f, remove_columns=cols)

def ds_tools():  # 4. Function calling (xLAM; fallback to glaive)
    try:
        d = load_dataset("Salesforce/xlam-function-calling-60k", split="train[:256]")
        def f(e):
            sys_m = {"role": "system", "content": "Tools:\n" + str(e["tools"])}
            return {"text": chat_text([sys_m, {"role": "user", "content": e["query"]},
                                       {"role": "assistant", "content": str(e["answers"])}])}
        return d.map(f, remove_columns=d.column_names)
    except Exception as ex:
        print("  xlam gated/unavailable -> glaive fallback:", ex, flush=True)
        d = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:256]")
        return d.map(lambda e: {"text": (e.get("system", "") + "\n" + e.get("chat", ""))[:4000]},
                     remove_columns=d.column_names)

def ds_mcp():  # 5. MCP tool-use (synthesized from our mcp_server.py order DB)
    sysm = ("You are an agent using MCP tools. Available: get_order(order_id), "
            "cancel_order(order_id).")
    rows = []
    orders = [("A1002", "processing", 80), ("A1003", "processing", 12), ("A1001", "shipped", 25)]
    for oid, status, price in orders * 22:
        conv = [{"role": "system", "content": sysm},
                {"role": "user", "content": f"What is the status of order {oid}?"},
                {"role": "assistant", "content": f'<tool_call>{{"name":"get_order","arguments":{{"order_id":"{oid}"}}}}</tool_call>'},
                {"role": "tool", "content": f'{{"status":"{status}","price":{price}}}'},
                {"role": "assistant", "content": f"Order {oid} is {status}."}]
        rows.append({"text": chat_text(conv)})
    return Dataset.from_dict({"text": [r["text"] for r in rows][:256]})

def run_dpo(stage, prev, steps):  # 6. Preference optimization
    print(f"\n===== STAGE {stage} (DPO, {steps} steps) =====", flush=True)
    d = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:256]")
    def f(e):
        return {"prompt": e["prompt"],
                "chosen": e["chosen"][-1]["content"],
                "rejected": e["rejected"][-1]["content"]}
    d = d.map(f, remove_columns=d.column_names)
    model = model_with_adapter(prev)
    cfg = DPOConfig(output_dir=f"/tmp/{stage}", max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=5e-5, logging_steps=5, beta=0.1,
        max_length=1024, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = DPOTrainer(model=model, args=cfg, train_dataset=d, processing_class=TOK)
    out = tr.train()
    save(model, stage); record(stage, loss=round(out.training_loss, 4), steps=steps)
    free(model)

def run_grpo(stage, prev, steps):  # 7. Agentic RL with verifiable reward
    print(f"\n===== STAGE {stage} (GRPO/RLVR, {steps} steps) =====", flush=True)
    d = load_dataset("openai/gsm8k", "main", split="train[:128]")
    def gold(a):
        m = re.findall(r"-?\d[\d,]*", a.split("####")[-1]); return m[-1].replace(",", "") if m else None
    d = d.map(lambda e: {"prompt": e["question"] + "\nEnd with 'Answer: <int>'.",
                         "gold": gold(e["answer"])}, remove_columns=d.column_names)
    def reward(completions, gold, **kw):
        out = []
        for c, g in zip(completions, gold):
            m = re.search(r"[Aa]nswer:\s*(-?\d[\d,]*)", c) or re.search(r"(-?\d[\d,]*)\s*$", c.strip())
            out.append(1.0 if (m and m.group(1).replace(",", "") == g) else 0.0)
        return out
    model = model_with_adapter(prev)
    cfg = GRPOConfig(output_dir=f"/tmp/{stage}", max_steps=steps, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, num_generations=2, learning_rate=1e-5, logging_steps=5,
        max_completion_length=256, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=d, processing_class=TOK)
    out = tr.train()
    save(model, stage); record(stage, loss=round(out.training_loss, 4), reward=out.metrics.get("train_reward"), steps=steps)
    free(model)

def run_agentic_grpo(stage, prev, steps):  # 7b. TRUE agentic RL: multi-turn tool loop
    print(f"\n===== STAGE {stage} (agentic GRPO, tool loop, {steps} steps) =====", flush=True)
    model = model_with_adapter(prev)
    cfg = GRPOConfig(output_dir=f"/tmp/{stage}", max_steps=steps, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, num_generations=2, learning_rate=1e-5, logging_steps=4,
        max_completion_length=384, max_tool_calling_iterations=4, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = GRPOTrainer(model=model, reward_funcs=success_reward, args=cfg,
                     train_dataset=agentic_ds(), processing_class=TOK, tools=AGENTIC_TOOLS)
    out = tr.train()
    save(model, stage); record(stage, loss=round(out.training_loss, 4), steps=steps)
    free(model)

STAGES = [
    ("1_cpt",       lambda p: run_sft("1_cpt", p, ds_cpt(), 30)),
    ("2_sft",       lambda p: run_sft("2_sft", p, ds_sft(), 40)),
    ("3_reasoning", lambda p: run_sft("3_reasoning", p, ds_reasoning(), 40)),
    ("4_tools",     lambda p: run_sft("4_tools", p, ds_tools(), 40)),
    ("5_mcp",       lambda p: run_sft("5_mcp", p, ds_mcp(), 30)),
    ("6_dpo",       lambda p: run_dpo("6_dpo", p, 30)),
    ("7_grpo",      lambda p: run_grpo("7_grpo", p, 15)),
    ("7b_agentic_grpo", lambda p: run_agentic_grpo("7b_agentic_grpo", p, 12)),
]

def main():
    prev = None
    for name, fn in STAGES:
        if done(name):
            print(f"[done] {name}", flush=True); prev = name; continue
        fn(prev); prev = name
    print("\n===== PIPELINE COMPLETE =====", flush=True)
    print(json.dumps(RESULTS, indent=2))

if __name__ == "__main__":
    main()
