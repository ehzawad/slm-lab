#!/usr/bin/env python3
"""NimbusWorks staged post-training v3: same TM curve as train_nimbus.py, but SFT-type
stages (sft/reasoning/tools/mcp) train with assistant-only loss via a {% generation %}-
patched chat template (template_masked.jinja). Datasets are conversational ({'messages':
[...]}) so TRL builds assistant token masks; prompts/tool-responses no longer receive
gradient. cpt is NOT retrained: adapters_v3/cpt is copied from the v2 adapters/cpt.
dpo and grpo are unchanged from v2 except for the adapter root.
Usage: python train_nimbus_v3.py <stage> [--smoke]. Resumable per stage."""
import os, sys, json, re, gc, shutil
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
ADAPTERS = f"{HERE}/adapters_v3"
V2_ADAPTERS = f"{HERE}/adapters"
ORDER = ["cpt", "sft", "reasoning", "tools", "mcp", "dpo", "grpo"]
SMOKE = "--smoke" in sys.argv or os.environ.get("SMOKE") == "1"

TOK = AutoTokenizer.from_pretrained(BASE)
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token
# Masked template: render-identical to the original, but wraps assistant completions
# (content, tool_calls serialization, and <|im_end|>\n) in {% generation %} markers so
# TRL's assistant_only_loss can build token masks. Verified by mask_audit.py.
TOK.chat_template = open(f"{HERE}/template_masked.jinja").read()

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
    return SFTConfig(output_dir=f"/tmp/nb3_{stage}", max_steps=steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=10, max_length=maxlen,
        report_to=[], bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        save_strategy="no", assistant_only_loss=True, use_liger_kernel=False)

def audit_masks(ds, maxlen, stage):
    """Truncation guard: replicate TRL's tokenization (apply_chat_template with
    return_assistant_tokens_mask) and assert every example keeps at least one trainable
    (assistant) token after truncation to max_length. Also report the active-label
    fraction; full-text training (v2) would be ~1.0."""
    active, total = 0, 0
    for i, row in enumerate(ds):
        out = TOK.apply_chat_template(row["messages"], tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True)
        mask = out["assistant_masks"][:maxlen]
        n = sum(mask)
        assert n > 0, (f"[{stage}] example {i} has zero trainable tokens after "
                       f"truncation to {maxlen}: {row['messages'][0]}")
        active += n; total += len(mask)
    frac = active / max(total, 1)
    print(f"[{stage}] mask audit: {len(ds)} examples OK, "
          f"active-label fraction = {frac:.3f}", flush=True)
    return frac

def run_sft_stage(stage, ds, steps, maxlen=1024, lr=2e-4):
    if SMOKE:
        steps = 3
        print(f"[{stage}] SMOKE mode: max_steps=3", flush=True)
    audit_masks(ds, maxlen, stage)
    model = load_model(stage)
    tr = SFTTrainer(model=model, args=sft_cfg(stage, steps, maxlen, lr),
                    train_dataset=ds, processing_class=TOK)
    out = tr.train()
    save(model, stage)
    print(f">> {stage} done. loss={round(out.training_loss, 4)}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

def msgs(path):
    rows = json.load(open(f"{HERE}/{path}"))
    return Dataset.from_list([{"messages": r} for r in rows])

def msgs_with_replay(path, n_replay=30):
    """Stage data + a replay slice of the SFT chat mix (same continual-learning fix as
    v2, same seeds), but conversational: rows are message lists, not rendered text."""
    import random
    rows = json.load(open(f"{HERE}/{path}"))
    replay = json.load(open(f"{HERE}/train_sft.json"))
    random.Random(0).shuffle(replay)
    all_rows = rows + replay[:n_replay]
    random.Random(1).shuffle(all_rows)
    return Dataset.from_list([{"messages": r} for r in all_rows])

# ---------------- stages ----------------
def cpt():
    # v3 does not retrain CPT (pure-docs LM loss is unaffected by assistant masking).
    # Reuse the v2 post-CPT adapter as the v3 starting point.
    src, dst = f"{V2_ADAPTERS}/cpt", f"{ADAPTERS}/cpt"
    assert os.path.exists(src), f"missing v2 cpt adapter at {src}"
    shutil.copytree(src, dst)
    print(f">> cpt: copied v2 adapter {src} -> {dst}", flush=True)

def sft():
    # Domain QA chats + general chat, conversational form (role/content only).
    dom = json.load(open(f"{HERE}/train_sft.json"))
    gen = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:80]")
    rows = [{"messages": r} for r in dom]
    rows += [{"messages": [{"role": m["role"], "content": m["content"]}
                           for m in e["messages"]]} for e in gen]
    run_sft_stage("sft", Dataset.from_list(rows), steps=90)

def reasoning():
    run_sft_stage("reasoning", msgs_with_replay("train_reasoning.json"), steps=50)

def tools():
    run_sft_stage("tools", msgs_with_replay("train_tools.json", n_replay=40), steps=50)

def mcp():
    run_sft_stage("mcp", msgs_with_replay("train_mcp.json"), steps=50)

def dpo():
    d = json.load(open(f"{HERE}/train_dpo.json"))
    ds = Dataset.from_list(d)  # prompt/chosen/rejected strings
    model = load_model("dpo")
    cfg = DPOConfig(output_dir="/tmp/nb3_dpo", max_steps=30, per_device_train_batch_size=1,
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
    cfg = GRPOConfig(output_dir="/tmp/nb3_grpo", max_steps=40, per_device_train_batch_size=4,
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
        os.makedirs(ADAPTERS, exist_ok=True)
        {"cpt": cpt, "sft": sft, "reasoning": reasoning, "tools": tools,
         "mcp": mcp, "dpo": dpo, "grpo": grpo}[stage]()
