import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES","1")
os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID")
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT","1")
import torch
from unsloth import FastLanguageModel
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import train_gptoss as T
model,tok=FastLanguageModel.from_pretrained(model_name=T.ADAPTER_SFT,dtype=None,
    max_seq_length=2048,load_in_4bit=True,full_finetuning=False)
FastLanguageModel.for_inference(model)
scen=json.load(open(os.path.join(HERE,"grpo_scenarios.json")))
for spec in scen[:2]:
    sc=spec["scenario"]
    msgs=T.build_grpo_prompt(sc)
    ids=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt").to(model.device)
    out=model.generate(input_ids=ids,max_new_tokens=256,do_sample=True,temperature=1.0,
        pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=False)
    plan=T._parse_plan(gen)
    r=T._replay_reward(sc,plan) if plan else 0.0
    print("==== SCENARIO",sc["id"],"====")
    print("COMPLETION:",repr(gen[:900]))
    print("PARSED_PLAN:",plan,"REWARD:",r)
    print()
