#!/usr/bin/env python3
"""GPU: pre-filter OPD candidate prompts through the teacher (base Qwen3-4B, no adapter).

For every candidate the teacher generates a greedy response (max_new_tokens=120).
Keep rules:
  fake      - kept ONLY if the teacher's output matches the refusal regex
              (otherwise the teacher would distill fabrication into the student)
  if / no_invent / chat - kept unconditionally
The teacher response is stored as the assistant turn, because GKDTrainer's
DataCollatorForChatML needs a full 'messages' conversation (last message =
completion) for the off-policy (1-lmbda) steps.

Reads opd_prompts_candidates.json, writes opd_prompts.json as
  [{"messages": [...user..., {"role":"assistant","content": teacher_out}], "category": ...}]
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"

REFUSAL_RE = re.compile(
    r"no record|don'?t have|do not have|not aware|unknown|"
    r"does ?n[o']t (?:seem to )?exist|no information|not familiar|"
    r"couldn'?t find|could not find|don'?t know|do not know|"
    r"i'?m not sure (?:what|which|there)|no such service|not a (?:known|real) service|"
    r"unable to find|cannot find|can'?t find|have no data|not in my",
    re.IGNORECASE)


def apply_template(tok, msgs):
    try:  # Qwen3: suppress thinking so the 120-token budget is spent on the answer
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    cands = json.load(open(f"{HERE}/opd_prompts_candidates.json"))
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    kept, dropped = [], Counter()
    kept_cnt = Counter()
    for i, row in enumerate(cands):
        text = apply_template(tok, row["prompt"])
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        resp = strip_think(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
        cat = row["category"]
        if cat == "fake" and not REFUSAL_RE.search(resp):
            dropped[cat] += 1
            print(f"[{i+1}/{len(cands)}] DROP {cat}: {row['prompt'][-1]['content'][:60]!r}", flush=True)
            continue
        kept.append({"messages": row["prompt"] + [{"role": "assistant", "content": resp}],
                     "category": cat})
        kept_cnt[cat] += 1
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(cands)}] kept={len(kept)}", flush=True)

    path = f"{HERE}/opd_prompts.json"
    json.dump(kept, open(path, "w"), indent=1)
    print(f"wrote {len(kept)} prompts -> {path}")
    for cat in sorted(set(kept_cnt) | set(dropped)):
        print(f"  {cat}: kept={kept_cnt.get(cat, 0)} dropped={dropped.get(cat, 0)}")


if __name__ == "__main__":
    main()
