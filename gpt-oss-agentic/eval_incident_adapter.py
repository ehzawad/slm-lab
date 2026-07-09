#!/usr/bin/env python3
"""Evaluate a TRAINED gpt-oss adapter (or the base model) on the incident-response
env using a LOCAL transformers/Unsloth generate wired into incident_harness.chat().

This is an attempted same-backend BEFORE/AFTER check: the same 24 scenarios, the
same harness, the same 4-bit gpt-oss-20b weights, differing only in whether the
SFT LoRA adapter is attached. It does not reproduce the grammar-constrained GGUF
serving path used for the main baseline, so if this local-generate path floors
the base model it cannot provide a valid task-capacity delta. The GGUF baseline
in incident_scores.json is a separate, backend-different reference point.

gpt-oss uses the harmony chat format. We render OpenAI-style messages+tools with the
stock `unsloth/gpt-oss-20b` tokenizer (the adapter-dir tokenizer's patched jinja
rejects the tool schema; the stock one produces byte-identical harmony text), generate
one turn, stop at `<|call|>` (tool call) or `<|return|>` (final), and parse the harmony
continuation back into an OpenAI assistant message with tool_calls.

Usage (GPU 1 / A6000):
  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python eval_incident_adapter.py --adapter adapters_gptoss/sft --label "gpt-oss-20b TRAINED"
  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python eval_incident_adapter.py --adapter none --label "gpt-oss-20b base (transformers)"
"""
import argparse, json, os, re, sys, time
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")   # required (sm_86 / bnb / cu130)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from incident_harness import run_episode  # noqa: E402
from incident_sim import build_scenarios, TOOLS_SPEC  # noqa: E402

MODEL_NAME = "unsloth/gpt-oss-20b"
MAX_SEQ = 8192
SCORES_PATH = os.path.join(HERE, "incident_scores.json")

# harmony markers
RET_ID = 200002  # <|return|>  (eos)
END_ID = 200007  # <|end|>

# Handles both harmony tool-call header forms:
#   ...to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>
#   ...to=functions.NAME<|channel|>commentary <|constrain|>json<|message|>ARGS<|call|>
_TOOLCALL_RE = re.compile(
    r"to=functions\.([A-Za-z0-9_\-]+).*?<\|message\|>(.*?)<\|call\|>", re.DOTALL)
_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL)
_TAG_RE = re.compile(r"<\|[^>]*\|>")


def parse_continuation(cont, ncall):
    """Parse a harmony continuation (text after the final `<|start|>assistant`)
    into an OpenAI-style assistant message dict."""
    m = _TOOLCALL_RE.search(cont)
    if m:
        name = m.group(1)
        args = m.group(2).strip()
        return {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"call_{ncall}", "type": "function",
                "function": {"name": name, "arguments": args or "{}"}}],
        }
    fm = _FINAL_RE.search(cont)
    if fm:
        content = fm.group(1).strip()
    else:
        content = _TAG_RE.sub("", cont).strip()
    return {"role": "assistant", "content": content or "done"}


def make_chat(model, tok, render_tok, temp=1.0, max_new_tokens=256, verbose=False):
    state = {"n": 0}

    def chat(messages, tools):
        prompt = render_tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=MAX_SEQ - max_new_tokens).to(model.device)
        # IMPORTANT: stop only on <|call|> (tool call complete) or <|return|> (final
        # answer / eos). Do NOT stop on <|end|>: in harmony an assistant turn emits
        # `analysis ...<|end|>` and THEN continues `<|start|>assistant to=functions...`,
        # so stopping at <|end|> would cut generation off before the tool call.
        gen_kw = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=[RET_ID],
            pad_token_id=RET_ID,
            stop_strings=["<|call|>"],
            tokenizer=tok,
        )
        if temp and temp > 0:
            gen_kw.update(do_sample=True, temperature=temp, top_p=1.0)
        else:
            gen_kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw)
        cont = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        state["n"] += 1
        msg = parse_continuation(cont, state["n"])
        if verbose:
            print("    RAWCONT:", repr(cont[:300]))
        return msg

    return chat


def load(adapter):
    """Return (model, tok). adapter='none' -> base 4-bit; else load adapter dir."""
    name = MODEL_NAME if adapter in (None, "none", "") else adapter
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, dtype=None, max_seq_length=MAX_SEQ,
        load_in_4bit=True, full_finetuning=False)
    FastLanguageModel.for_inference(model)
    return model, tok


def evaluate(label, adapter, temp, n_scenarios=24, max_calls=15, verbose=False):
    print(f"=== loading model (adapter={adapter}) ===", flush=True)
    t_load = time.time()
    model, tok = load(adapter)
    render_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"=== loaded in {time.time()-t_load:.1f}s ===", flush=True)

    scenarios = build_scenarios(n_scenarios)
    chat = make_chat(model, tok, render_tok, temp=temp, verbose=verbose)
    per = []
    t0 = time.time()
    for i, sc in enumerate(scenarios, 1):
        r = run_episode(sc, chat, max_calls=max_calls, verbose=verbose)
        per.append(r)
        print(f"  [{i:>2}/{len(scenarios)}] {sc['id']:<28} "
              f"solved={int(r['solved'])} steps={r['steps']:>2} "
              f"root_cause={int(r['correct_root_cause'])} "
              f"redundant={r['redundant_calls']}", flush=True)
    elapsed = time.time() - t0

    n = len(per)
    solved = sum(x["solved"] for x in per)
    rc = sum(x["correct_root_cause"] for x in per)
    avg_steps = sum(x["steps"] for x in per) / n
    total_calls = sum(x["steps"] for x in per)
    total_redundant = sum(x["redundant_calls"] for x in per)
    redundant_rate = (total_redundant / total_calls) if total_calls else 0.0

    summary = {
        "label": label,
        "backend": "transformers/unsloth 4bit",
        "adapter": adapter,
        "temp": temp,
        "n_scenarios": n,
        "max_calls": max_calls,
        "solved": solved,
        "solved_rate": round(solved / n, 4),
        "correct_root_cause": rc,
        "root_cause_rate": round(rc / n, 4),
        "avg_steps": round(avg_steps, 3),
        "redundant_calls_total": total_redundant,
        "redundant_call_rate": round(redundant_rate, 4),
        "elapsed_s": round(elapsed, 1),
        "gpu": "GPU%s" % os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_scenario": [
            {"id": x["id"], "solved": bool(x["solved"]), "steps": x["steps"],
             "correct_root_cause": bool(x["correct_root_cause"]),
             "redundant_calls": x["redundant_calls"]}
            for x in per
        ],
    }
    print(f"\n== {label} ==")
    print(f"  solved             : {solved}/{n}  ({summary['solved_rate']*100:.1f}%)")
    print(f"  correct root-cause : {rc}/{n}  ({summary['root_cause_rate']*100:.1f}%)")
    print(f"  avg steps          : {summary['avg_steps']}")
    print(f"  redundant-call rate: {summary['redundant_call_rate']*100:.1f}%  "
          f"({total_redundant}/{total_calls})")
    print(f"  elapsed            : {summary['elapsed_s']}s")
    _append_score(summary)
    return summary


def _append_score(summary):
    data = []
    if os.path.exists(SCORES_PATH):
        try:
            with open(SCORES_PATH) as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        except (json.JSONDecodeError, OSError):
            data = []
    data.append(summary)
    with open(SCORES_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[scores] appended -> {SCORES_PATH}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none",
                    help="'none' for base, or path to adapter dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-calls", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    adapter = args.adapter
    if adapter not in (None, "none", "") and not os.path.isabs(adapter):
        adapter = os.path.join(HERE, adapter)
    evaluate(args.label, adapter, args.temp,
             n_scenarios=args.n, max_calls=args.max_calls, verbose=args.verbose)


if __name__ == "__main__":
    main()
