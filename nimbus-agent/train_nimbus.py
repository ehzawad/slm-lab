#!/usr/bin/env python3
"""NimbusWorks staged post-training: the TM curve, reproduced. One accumulating QLoRA
adapter through CPT -> SFT -> reasoning -> tools -> MCP -> DPO -> GRPO; every stage is
evaluated (by run_all.sh) on all four metrics so stage-induced regressions and repairs
are visible. Usage: python train_nimbus.py <stage>. Resumable per stage."""
import os, sys, json, re, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"
ADAPTERS = f"{HERE}/adapters"
ORDER = ["cpt", "sft", "reasoning", "tools", "mcp", "dpo", "grpo"]

TOK = AutoTokenizer.from_pretrained(BASE)
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token

def prev_stage(stage):
    i = ORDER.index(stage)
    for p in reversed(ORDER[:i]):
        if os.path.exists(f"{ADAPTERS}/{p}"):
            return p
    return None

def load_model(stage):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        dtype=torch.bfloat16, device_map={"": 0})
    m.config.use_cache = False
    m = prepare_model_for_kbit_training(m)
    p = prev_stage(stage)
    if p:
        model = PeftModel.from_pretrained(m, f"{ADAPTERS}/{p}", is_trainable=True)
        print(f"  [resumed adapter: {p}]", flush=True)
    else:
        model = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
        print("  [fresh adapter]", flush=True)
    return model

def save(model, stage):
    try: model.save_pretrained(f"{ADAPTERS}/{stage}", selected_adapters=["default"])
    except Exception: model.save_pretrained(f"{ADAPTERS}/{stage}")

def sft_cfg(stage, steps, maxlen=1024, lr=2e-4):
    return SFTConfig(output_dir=f"/tmp/nb_{stage}", max_steps=steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=10, max_length=maxlen,
        report_to=[], bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        save_strategy="no")

def run_sft_stage(stage, ds, steps, maxlen=1024, lr=2e-4):
    model = load_model(stage)
    tr = SFTTrainer(model=model, args=sft_cfg(stage, steps, maxlen, lr),
                    train_dataset=ds, processing_class=TOK)
    out = tr.train()
    save(model, stage)
    print(f">> {stage} done. loss={round(out.training_loss, 4)}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

def chats(path):
    rows = json.load(open(f"{HERE}/{path}"))
    return Dataset.from_list([{"text": TOK.apply_chat_template(r, tokenize=False)} for r in rows])

def chats_with_replay(path, n_replay=30):
    """Stage data + a replay slice of the SFT chat mix. This is the continual-learning
    fix for the v1 catastrophic format collapse (tools stage: domain 68->0, IF 30):
    single-format stages must keep seeing normal QA/chat, per TM's 70/30 mixing lesson."""
    import random
    rows = json.load(open(f"{HERE}/{path}"))
    replay = json.load(open(f"{HERE}/train_sft.json"))
    random.Random(0).shuffle(replay)
    all_rows = rows + replay[:n_replay]
    random.Random(1).shuffle(all_rows)
    return Dataset.from_list([{"text": TOK.apply_chat_template(r, tokenize=False)}
                              for r in all_rows])

# ---------------- stages ----------------
def cpt():
    # Pure-docs mid-training, deliberately WITHOUT chat mixing: we want the TM
    # knowledge-up / instruction-following-down tension to be visible, then repaired by SFT.
    docs = json.load(open(f"{HERE}/corpus.json"))
    ds = Dataset.from_list([{"text": d} for d in docs])
    run_sft_stage("cpt", ds, steps=120, maxlen=768)

def sft():
    # Domain QA chats + general chat (restores assistant behavior, per TM recipe).
    dom = json.load(open(f"{HERE}/train_sft.json"))
    gen = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:80]")
    rows = [{"text": TOK.apply_chat_template(r, tokenize=False)} for r in dom]
    rows += [{"text": TOK.apply_chat_template(e["messages"], tokenize=False,)} for e in gen]
    run_sft_stage("sft", Dataset.from_list(rows), steps=90)

def reasoning():
    run_sft_stage("reasoning", chats_with_replay("train_reasoning.json"), steps=50)

def tools():
    run_sft_stage("tools", chats_with_replay("train_tools.json", n_replay=40), steps=50)

def mcp():
    run_sft_stage("mcp", chats_with_replay("train_mcp.json"), steps=50)

def dpo():
    d = json.load(open(f"{HERE}/train_dpo.json"))
    ds = Dataset.from_list(d)  # prompt/chosen/rejected strings
    model = load_model("dpo")
    # v2: gentler DPO (v1 at 60 steps / 5e-5 eroded domain QA 42->24)
    cfg = DPOConfig(output_dir="/tmp/nb_dpo", max_steps=30, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-5, logging_steps=10, beta=0.1,
        max_length=1024, report_to=[], bf16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, save_strategy="no")
    tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=TOK)
    out = tr.train()
    save(model, "dpo")
    print(f">> dpo done. loss={round(out.training_loss, 4)}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

def grpo():
    d = json.load(open(f"{HERE}/train_grpo.json"))
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": t["q"]}],
                             "gold": t["gold"]} for t in d])
    def _text(c):
        return " ".join(m.get("content") or "" for m in c if m.get("role") == "assistant") \
               if isinstance(c, list) else (c or "")
    def reward(completions, gold, **kw):
        out = []
        for c, g in zip(completions, gold):
            m = re.search(r"[Aa]nswer:\s*(-?\d[\d,]*)", _text(c))
            out.append(1.0 if (m and m.group(1).replace(",", "") == g) else 0.0)
        return out
    model = load_model("grpo")
    cfg = GRPOConfig(output_dir="/tmp/nb_grpo", max_steps=40, per_device_train_batch_size=4,
        gradient_accumulation_steps=2, num_generations=4, learning_rate=1e-5, beta=0.04,
        logging_steps=10, max_completion_length=192, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=ds,
                     processing_class=TOK)
    out = tr.train()
    save(model, "grpo")
    print(f">> grpo done. loss={round(out.training_loss, 4)}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    stage = sys.argv[1]
    if os.path.exists(f"{ADAPTERS}/{stage}"):
        print(f"[done] {stage}")
    else:
        {"cpt": cpt, "sft": sft, "reasoning": reasoning, "tools": tools,
         "mcp": mcp, "dpo": dpo, "grpo": grpo}[stage]()
