#!/usr/bin/env python3
"""Multi-metric eval across NimbusWorks pipeline checkpoints - the TM curve, measured.
Metrics: domain-QA recall, instruction-following, hallucination handling (refuse fake /
answer real), tool-call validity. Usage: python eval_nimbus.py <label> [adapter_dir]"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"
TOK = AutoTokenizer.from_pretrained(BASE); TOK.padding_side = "left"
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token

TOOLS = [{"type": "function", "function": {"name": n, "description": d,
          "parameters": {"type": "object", "properties": {"service": {"type": "string"},
                         "replicas": {"type": "integer"}}, "required": ["service"]}}}
         for n, d in [("nbx_status", "show health of a service"),
                      ("nbx_deploy_rollback", "roll a service back to previous release"),
                      ("nbx_scale", "set replica count of a service"),
                      ("nbx_restart", "rolling restart of a service"),
                      ("nbx_logs", "show recent log lines of a service")]]

def load(adapter=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval(); return m

@torch.no_grad()
def gen(model, prompts, tools=None, max_new=160):
    outs = []
    for i in range(0, len(prompts), 8):
        batch = prompts[i:i+8]
        texts = [TOK.apply_chat_template([{"role": "user", "content": p}], tools=tools,
                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in batch]
        enc = TOK(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=TOK.pad_token_id)
        outs += [TOK.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                 for j in range(len(batch))]
    return outs

def ev_domain(model):
    d = json.load(open(f"{HERE}/eval_domain_qa.json"))
    outs = gen(model, [x["q"] for x in d], max_new=60)
    ok = sum(1 for x, o in zip(d, outs) if x["a"].lower() in o.lower())
    return round(100 * ok / len(d), 1)

def ev_if(model):
    d = json.load(open(f"{HERE}/eval_if.json"))
    outs = gen(model, [x["q"] for x in d], max_new=40)
    ok = 0
    for x, o in zip(d, outs):
        o = o.strip()
        if not x["check"]:
            ok += (0 < len(o.split()) <= 25)          # answered, concisely
        else:
            ok += (x["check"].lower() in o.lower() and len(o) < 120)
    return round(100 * ok / len(d), 1)

REFUSAL = re.compile(r"(don'?t have|no record|not aware|couldn'?t find|no service|doesn'?t exist|"
                     r"not familiar|unknown|no information|cannot find|may be misspelled)", re.I)

def ev_halluc(model):
    d = json.load(open(f"{HERE}/eval_halluc.json"))
    outs = gen(model, [x["q"] for x in d], max_new=90)
    ok = 0
    for x, o in zip(d, outs):
        if x["fake"]:
            ok += bool(REFUSAL.search(o))              # must refuse the fake entity
        else:
            ok += (not REFUSAL.search(o)) and (x["a_substr"].lower()[:15] in o.lower())
    return round(100 * ok / len(d), 1)

def ev_tools(model):
    d = json.load(open(f"{HERE}/eval_tools.json"))
    outs = gen(model, [x["q"] for x in d], tools=TOOLS, max_new=120)
    ok = 0
    for x, o in zip(d, outs):
        low = o.lower()
        ok += (x["tool"] in low and x["arg_substr"].lower() in low)
    return round(100 * ok / len(d), 1)

def main():
    label = sys.argv[1]
    adapter = sys.argv[2] if len(sys.argv) > 2 else None
    rp = f"{HERE}/scores.json"
    scores = json.load(open(rp)) if os.path.exists(rp) else {}
    model = load(adapter)
    row = {"domain_qa": ev_domain(model), "instruction_following": ev_if(model),
           "hallucination_handling": ev_halluc(model), "tool_validity": ev_tools(model)}
    scores[label] = row
    json.dump(scores, open(rp, "w"), indent=2)
    print(f"{label}: {row}", flush=True)

if __name__ == "__main__":
    main()
