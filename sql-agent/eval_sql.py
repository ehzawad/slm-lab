#!/usr/bin/env python3
"""Execution-accuracy eval for the NL->SQL agent. Loads a model (base or a LoRA adapter),
generates SQL for held-out questions, executes predictions, and reports EXECUTION ACCURACY
(result-set match vs gold) + valid-SQL rate. A5000 only. Usage: python eval_sql.py [adapter]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sql_exec import make_prompt, exec_reward, clean_sql, build_db, run

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"
ADAPTERS = f"{HERE}/adapters"
TOK = AutoTokenizer.from_pretrained(BASE); TOK.padding_side = "left"
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token

def load(adapter=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0})
    if adapter:
        m = PeftModel.from_pretrained(m, f"{ADAPTERS}/{adapter}")
    m.eval(); return m

@torch.no_grad()
def evaluate(adapter=None, n=150):
    data = json.load(open(f"{HERE}/eval.json"))[:n]
    model = load(adapter)
    preds = []
    for i in range(0, len(data), 8):
        batch = data[i:i+8]
        texts = [TOK.apply_chat_template([{"role": "user", "content": make_prompt(r["context"], r["question"])}],
                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
        enc = TOK(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        out = model.generate(**enc, max_new_tokens=192, do_sample=False, pad_token_id=TOK.pad_token_id)
        for j in range(len(batch)):
            preds.append(TOK.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    exec_acc = valid = 0
    for r, p in zip(data, preds):
        exec_acc += exec_reward(r["context"], r["gold"], p)
        try:
            c = build_db(r["context"]); run(c, clean_sql(p)); c.close(); valid += 1
        except Exception:
            pass
    label = adapter or "base"
    res = {"model": label, "n": len(data), "exec_acc": round(100*exec_acc/len(data), 1),
           "valid_sql": round(100*valid/len(data), 1)}
    print(f"  {label:16s} exec_acc={res['exec_acc']}%  valid_sql={res['valid_sql']}%", flush=True)
    rp = f"{HERE}/eval_scores.json"
    scores = json.load(open(rp)) if os.path.exists(rp) else {}
    scores[label] = res; json.dump(scores, open(rp, "w"), indent=2)
    del model; torch.cuda.empty_cache()
    return res

if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(adapter)
