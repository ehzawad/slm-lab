#!/usr/bin/env python3
"""Quality evaluation across pipeline checkpoints — measure capability, not loss.
Scores held-out REASONING accuracy (GSM8K) and TOOL-CALL validity/correctness for each
stage's adapter, so we can see (a) where quality is built and (b) whether later stages
(DPO/GRPO/agentic) DEGRADE earlier capabilities — the single-adapter interference risk.
A5000 only. Resumable: checkpoints already in eval_results.json are skipped."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from common import BASE, ADAPTERS, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = load_tokenizer(); TOK.padding_side = "left"

# Checkpoints to compare (label, adapter dir or None for untrained base)
CHECKPOINTS = [
    ("base",        None),
    ("3_reasoning", "3_reasoning"),
    ("4_tools",     "4_tools"),
    ("5_mcp",       "5_mcp"),
    ("6_dpo",       "6_dpo"),
    ("7b_agentic",  "7b_agentic_grpo"),
]

# Tool-call eval set (prompt, expected_tool, arg_substring)
EVAL_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather.",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web.",
      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate arithmetic.",
      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
TOOL_PROMPTS = [
    ("What's the weather in Tokyo?", "get_weather", "tokyo"),
    ("Search for news about the RTX A5000.", "search_web", "a5000"),
    ("What is 3847 times 219?", "calculator", "3847"),
    ("Tell me the temperature in Paris.", "get_weather", "paris"),
    ("Find recent papers on GRPO.", "search_web", "grpo"),
    ("Compute 15% of 2400.", "calculator", "2400"),
    ("Is it raining in London?", "get_weather", "london"),
    ("Look up the 2022 World Cup winner.", "search_web", "world cup"),
]

def load_model(adapter):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0})
    if adapter:
        m = PeftModel.from_pretrained(m, f"{ADAPTERS}/{adapter}")
    m.eval()
    return m

@torch.no_grad()
def gen(model, chats, tools=None, max_new=512):
    outs = []
    for i in range(0, len(chats), 8):
        batch = chats[i:i+8]
        texts = [TOK.apply_chat_template(c, tools=tools, tokenize=False, add_generation_prompt=True) for c in batch]
        enc = TOK(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=TOK.pad_token_id)
        for j in range(len(batch)):
            gen_ids = out[j][enc["input_ids"].shape[1]:]
            outs.append(TOK.decode(gen_ids, skip_special_tokens=True))
    return outs

def eval_reasoning(model, n=40):
    d = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    chats = [[{"role": "user", "content": q + "\nReason step by step, end with 'Answer: <integer>'."}]
             for q in d["question"]]
    golds = [re.findall(r"-?\d[\d,]*", a.split("####")[-1])[-1].replace(",", "") for a in d["answer"]]
    ok = 0
    for txt, g in zip(gen(model, chats), golds):
        m = re.search(r"[Aa]nswer:\s*(-?\d[\d,]*)", txt)
        got = m.group(1).replace(",", "") if m else (re.findall(r"-?\d[\d,]*", txt) or [""])[-1].replace(",", "")
        ok += (got == g)
    return round(100 * ok / n, 1)

def eval_tools(model):
    chats = [[{"role": "user", "content": p}] for p, _, _ in TOOL_PROMPTS]
    valid = correct = 0
    for txt, (_, exp_tool, arg) in zip(gen(model, chats, tools=EVAL_TOOLS, max_new=256), TOOL_PROMPTS):
        low = txt.lower()
        called = ("tool_call" in low) or ('"name"' in low and exp_tool in low) or (f"<function={exp_tool}" in low)
        if called: valid += 1
        if exp_tool in low and arg in low: correct += 1
    n = len(TOOL_PROMPTS)
    return round(100 * valid / n, 1), round(100 * correct / n, 1)

def main():
    rp = f"{HERE}/eval_results.json"
    results = json.load(open(rp)) if os.path.exists(rp) else {}
    for label, adapter in CHECKPOINTS:
        if label in results:
            print(f"[done] {label}", flush=True); continue
        if adapter and not os.path.exists(f"{ADAPTERS}/{adapter}"):
            print(f"[skip] {label}: adapter missing", flush=True); continue
        print(f"\n=== eval {label} ===", flush=True)
        model = load_model(adapter)
        racc = eval_reasoning(model)
        tvalid, tcorrect = eval_tools(model)
        results[label] = {"reasoning_acc": racc, "tool_valid": tvalid, "tool_correct": tcorrect}
        print(f"  reasoning_acc={racc}%  tool_valid={tvalid}%  tool_correct={tcorrect}%", flush=True)
        json.dump(results, open(rp, "w"), indent=2)
        del model; torch.cuda.empty_cache()
    print("\n=== QUALITY TRAJECTORY ===", flush=True)
    print(f"{'checkpoint':14s} {'reason%':>8} {'toolvalid%':>11} {'toolcorrect%':>13}")
    for label, _ in CHECKPOINTS:
        if label in results:
            r = results[label]
            print(f"{label:14s} {r['reasoning_acc']:>8} {r['tool_valid']:>11} {r['tool_correct']:>13}")

if __name__ == "__main__":
    main()
