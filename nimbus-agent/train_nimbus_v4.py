#!/usr/bin/env python3
"""v4: the tools-dip fix. v3's curve showed the format-narrow tools stage erodes domain
knowledge (44 -> 36) before MCP's grounded-answer format recovers it (-> 51). v4 does not
sequence the narrow format at all: tools + MCP traces + a richer replay slice (sft chats
AND reasoning traces) are BLENDED into one 'toolsmix' stage. Everything else (masking,
steps budget, dpo, grpo) matches v3. Reuses v3's cpt/sft/reasoning adapters unchanged.
Usage: python train_nimbus_v4.py <cpt|sft|reasoning|toolsmix|dpo|grpo>. Resumable."""
import os, sys, json, shutil, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nimbus_v3 as t3
from datasets import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = f"{HERE}/adapters_v3"
V4 = f"{HERE}/adapters_v4"
# Repoint the v3 machinery at the v4 adapter root and stage order.
t3.ADAPTERS = V4
t3.ORDER = ["cpt", "sft", "reasoning", "toolsmix", "dpo", "grpo"]

def copy_stage(stage):
    src, dst = f"{V3}/{stage}", f"{V4}/{stage}"
    assert os.path.exists(src), f"missing v3 adapter {src}"
    shutil.copytree(src, dst)
    print(f">> {stage}: copied v3 adapter -> {dst}", flush=True)

def toolsmix():
    """Blend tool-calls + MCP traces + replay (sft chats AND reasoning traces) into one
    stage. Steps 90 ~= v3's tools(50) + mcp(50) budget, slightly under, same LR."""
    tools = json.load(open(f"{HERE}/train_tools.json"))
    mcp = json.load(open(f"{HERE}/train_mcp.json"))
    sft_replay = json.load(open(f"{HERE}/train_sft.json"))
    reas_replay = json.load(open(f"{HERE}/train_reasoning.json"))
    random.Random(0).shuffle(sft_replay)
    random.Random(2).shuffle(reas_replay)
    rows = tools + mcp + sft_replay[:40] + reas_replay[:15]
    random.Random(1).shuffle(rows)
    ds = Dataset.from_list([{"messages": r} for r in rows])
    print(f"[toolsmix] blended dataset: {len(tools)} tools + {len(mcp)} mcp + "
          f"40 sft-replay + 15 reasoning-replay = {len(ds)}", flush=True)
    t3.run_sft_stage("toolsmix", ds, steps=90)

if __name__ == "__main__":
    stage = sys.argv[1]
    if os.path.exists(f"{V4}/{stage}"):
        print(f"[done] {stage}")
    else:
        os.makedirs(V4, exist_ok=True)
        {"cpt": lambda: copy_stage("cpt"), "sft": lambda: copy_stage("sft"),
         "reasoning": lambda: copy_stage("reasoning"),
         "toolsmix": toolsmix, "dpo": t3.dpo, "grpo": t3.grpo}[stage]()
