#!/usr/bin/env python3
"""Stage 7b: TRUE agentic RL — GRPO with a real multi-turn tool loop.
Unlike stage 7 (single-turn GSM8K math RLVR), here TRL's GRPOTrainer is given Python
callable tools; it lets the policy call them, executes them, appends results, regenerates,
and we reward END-TO-END task success. Resumes the accumulated adapter from stage 6 (DPO).
Proof-of-flow: tiny prompt set, few steps. Base = Qwen3-4B, QLoRA 4-bit, A5000."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from datasets import Dataset
from common import load_tokenizer, model_with_adapter, ADAPTERS
from trl import GRPOTrainer, GRPOConfig

TOK = load_tokenizer()
_ORDERS = {"A1001": ("Wireless Mouse", 25, "shipped"),
           "A1002": ("Mechanical Keyboard", 80, "processing"),
           "A1003": ("USB-C Cable", 12, "processing")}

# --- Tools as plain callables (TRL derives the schema from signature + docstring) ---
def get_order(order_id: str) -> str:
    """Look up an order's item, price, and status.

    Args:
        order_id: The order identifier, e.g. 'A1002'.
    """
    if order_id not in _ORDERS:
        return json.dumps({"error": f"order {order_id} not found"})
    item, price, status = _ORDERS[order_id]
    return json.dumps({"order_id": order_id, "item": item, "price": price, "status": status})

def cancel_order(order_id: str) -> str:
    """Cancel an order if it has not shipped; returns the refund amount.

    Args:
        order_id: The order identifier, e.g. 'A1002'.
    """
    if order_id not in _ORDERS:
        return json.dumps({"error": f"order {order_id} not found"})
    item, price, status = _ORDERS[order_id]
    if status == "shipped":
        return json.dumps({"error": f"order {order_id} already shipped"})
    return json.dumps({"ok": True, "refund": price})

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Args:
        expression: An arithmetic expression, e.g. '80 * 0.15'.
    """
    if not re.fullmatch(r"[0-9+\-*/(). %]+", expression or ""):
        return json.dumps({"error": "invalid expression"})
    try:
        return json.dumps({"result": eval(expression, {"__builtins__": {}}, {})})
    except Exception as e:
        return json.dumps({"error": str(e)})

TOOLS = [get_order, cancel_order, calculator]

# --- Agentic tasks with verifiable end-state ---
def build_ds():
    tasks = [
        ("Cancel order A1002 if it has not shipped, then reply 'REFUND: <number>'.", r"REFUND:\s*80"),
        ("Look up order A1003 and reply with its price as 'PRICE: <number>'.", r"PRICE:\s*12"),
        ("Is order A1001 shipped? Reply exactly 'SHIPPED' or 'NOT'.", r"\bSHIPPED\b"),
        ("Compute 15% of order A1002's price and reply 'RESULT: <number>'.", r"RESULT:\s*12"),
        ("Cancel order A1003 and reply 'REFUND: <number>'.", r"REFUND:\s*12"),
        ("Look up order A1001 and reply with its status as 'STATUS: <status>'.", r"STATUS:\s*shipped"),
    ]
    rows = [{"prompt": [{"role": "user", "content": t}], "pat": p} for t, p in tasks * 8]
    return Dataset.from_list(rows)

def _text(c):
    if isinstance(c, list):  # conversational completion -> concat assistant text
        return " ".join(m.get("content") or "" for m in c if m.get("role") == "assistant")
    return c or ""

def success_reward(completions, pat, **kwargs):
    return [1.0 if re.search(p, _text(c), re.I) else 0.0 for c, p in zip(completions, pat)]

def main():
    stage = "7b_agentic_grpo"
    if os.path.exists(f"{ADAPTERS}/{stage}"):
        print(f"[done] {stage}"); return
    print(f"\n===== STAGE {stage}: agentic GRPO (multi-turn tool loop) =====", flush=True)
    model = model_with_adapter("6_dpo")  # resume the accumulated post-DPO policy
    cfg = GRPOConfig(output_dir=f"/tmp/{stage}", max_steps=12, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, num_generations=2, learning_rate=1e-5, logging_steps=2,
        max_completion_length=384, max_tool_calling_iterations=4, report_to=[], bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, save_strategy="no")
    tr = GRPOTrainer(model=model, reward_funcs=success_reward, args=cfg,
                     train_dataset=build_ds(), processing_class=TOK, tools=TOOLS)
    out = tr.train()
    model.save_pretrained(f"{ADAPTERS}/{stage}", selected_adapters=["default"])
    rewards = [m for k, m in out.metrics.items() if "reward" in k]
    print(f"\n>> {stage} done. loss={round(out.training_loss,4)} metrics_reward={rewards}", flush=True)

if __name__ == "__main__":
    main()
