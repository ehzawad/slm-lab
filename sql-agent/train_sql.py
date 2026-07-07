#!/usr/bin/env python3
"""NL->SQL quality training on the A5000. Two aligned stages (single domain -> no
interference): SFT on (schema+question -> gold SQL), then GRPO with an EXECUTION reward
(run predicted SQL, reward = result-set match vs gold). QLoRA 4-bit + LoRA-all-linear.
Usage: python train_sql.py {sft|grpo}. GRPO resumes the SFT adapter."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from sql_exec import make_prompt, make_cot_prompt, exec_reward

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"
ADAPTERS = f"{HERE}/adapters"
TOK = AutoTokenizer.from_pretrained(BASE)
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token
TRAIN = json.load(open(f"{HERE}/train.json"))

def load_lora(prev=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0})
    m.config.use_cache = False
    m = prepare_model_for_kbit_training(m)
    if prev and os.path.exists(f"{ADAPTERS}/{prev}"):
        model = PeftModel.from_pretrained(m, f"{ADAPTERS}/{prev}", is_trainable=True)
        print(f"  [resumed {prev}]", flush=True)
    else:
        model = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()
    return model

def sft():
    print("\n===== SQL SFT (schema+question -> gold SQL) =====", flush=True)
    ds = Dataset.from_list([{"text": TOK.apply_chat_template(
        [{"role": "user", "content": make_prompt(r["context"], r["question"])},
         {"role": "assistant", "content": r["gold"]}], tokenize=False)} for r in TRAIN])
    model = load_lora()
    args = SFTConfig(output_dir="/tmp/sql_sft", num_train_epochs=1, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.03, logging_steps=20, max_length=1024, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=TOK)
    out = tr.train()
    model.save_pretrained(f"{ADAPTERS}/sft", selected_adapters=["default"])
    print(f">> SFT done. loss={round(out.training_loss,4)}", flush=True)

def cotsft():
    print("\n===== SQL CoT-SFT (reasoning + SQL, from sql_explanation) =====", flush=True)
    ds = Dataset.from_list([{"text": TOK.apply_chat_template(
        [{"role": "user", "content": make_cot_prompt(r["context"], r["question"])},
         {"role": "assistant", "content": f"Reasoning: {r['explanation']}\nSQL: {r['gold']}"}],
        tokenize=False)} for r in TRAIN])
    model = load_lora()  # fresh from base
    args = SFTConfig(output_dir="/tmp/sql_cotsft", num_train_epochs=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.03, logging_steps=20, max_length=1536, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=TOK)
    out = tr.train()
    model.save_pretrained(f"{ADAPTERS}/cotsft", selected_adapters=["default"])
    print(f">> CoT-SFT done. loss={round(out.training_loss,4)}", flush=True)

def grpo():
    print("\n===== SQL GRPO (execution reward) =====", flush=True)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": make_prompt(r["context"], r["question"])}],
                             "context": r["context"], "gold": r["gold"]} for r in TRAIN[:800]])
    def _text(c):
        return " ".join(m.get("content") or "" for m in c if m.get("role") == "assistant") if isinstance(c, list) else (c or "")
    def reward(completions, context, gold, **kw):
        return [exec_reward(ctx, g, _text(c)) for c, ctx, g in zip(completions, context, gold)]
    model = load_lora("sft")
    # Council-reconciled: explicit KL anchor to SFT (beta), 8 generations for signal
    # (cuts zero-std groups 18%->1.4%), execution reward already hardened vs degenerate hacks.
    cfg = GRPOConfig(output_dir="/tmp/sql_grpo", max_steps=80, per_device_train_batch_size=8,
        gradient_accumulation_steps=4, num_generations=8, learning_rate=1e-5, logging_steps=10,
        beta=0.04, temperature=0.9, log_completions=False,
        max_completion_length=192, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=ds, processing_class=TOK)
    out = tr.train()
    model.save_pretrained(f"{ADAPTERS}/grpo", selected_adapters=["default"])
    print(f">> GRPO done. loss={round(out.training_loss,4)}", flush=True)

if __name__ == "__main__":
    {"sft": sft, "cotsft": cotsft, "grpo": grpo}[sys.argv[1]]()
