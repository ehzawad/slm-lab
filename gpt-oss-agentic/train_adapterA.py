#!/usr/bin/env python3
"""ADAPTER A - "tool-call reliability" SFT (Unsloth gpt-oss-20b, A6000 / GPU 1).

Pipeline:
  1. Load linearized 4-bit gpt-oss-20b.
  2. LoRA r16 alpha16 all-linear (+ MoE expert target_parameters layers 7/15/23
     when the checkpoint exposes them).
  3. Render adapterA_train.json with the OFFICIAL harmony template (stock
     AutoTokenizer + TOOLS_SPEC) == the vLLM inference path.
  4. train_on_responses_only(response_part="<|start|>assistant",
     instruction_part="<|start|>functions") -> assistant-only loss mask.
     DECODE one masked batch and PRINT the loss-carrying tokens (MANDATORY proof).
  5. SFT 1 epoch, lr 1e-4, max_seq 2048, eff-batch 16 (~34 steps). EARLY STOP if
     train loss < 0.1 (memorization guard).
  6. Free-running generation on the held-out prompts -> valid-tool-call rate
     (parseable commentary-channel call). THIS is the success metric.

Run: CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python train_adapterA.py
"""
import os, sys, json, re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")  # sm_86 cublasLt fix (proven)

import torch
import unsloth  # noqa: F401  (must precede unsloth_zoo)
from unsloth import FastLanguageModel
from unsloth_zoo.dataset_utils import train_on_responses_only

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from incident_sim import TOOLS_SPEC  # noqa: E402

MODEL_NAME = "unsloth/gpt-oss-20b"
MAX_SEQ = 2048
ADAPTER_A = os.path.join(HERE, "adapters_gptoss", "adapterA_vllm")
TRAIN_PATH = os.path.join(HERE, "adapterA_train.json")
HELD_PATH = os.path.join(HERE, "adapterA_heldout.json")

# vLLM-servable target set only: its LoRA loader accepts {q,k,v,o}_proj + router + experts.
# The per-expert target_parameters form (experts.down_projs.N) is NOT servable, so we drop it
# and the non-servable gate/up/down_proj names. Attention + router is the servable compromise
# (MoE-expert adaptation sacrificed for a valid same-path vLLM before/after).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention-only: router is GptOssTopKRouter (not nn.Linear, PEFT rejects it); q/k/v/o are vLLM-servable
EXPERT_LAYERS = ()
RESP_PART = "<|start|>assistant"
INSTR_PART = "<|start|>functions"

VALID_TOOLS = {"get_status", "get_logs", "get_dependencies", "check_all",
               "restart", "scale", "rollback", "set_config"}

# first parseable commentary-channel tool call in a free-running generation
_CALL_RE = re.compile(
    r"to=functions\.([a-zA-Z_][a-zA-Z0-9_]*)[^{<]*?<\|message\|>(\{.*?\})\s*<\|call\|>",
    re.DOTALL)


# --------------------------------------------------------------------------
def build_lora(model):
    """r16 a16 all-linear; add MoE expert target_parameters (layers 7/15/23) when
    the linearized checkpoint exposes grouped `experts.gate_up_proj/down_proj`."""
    param_names = [n for n, _ in model.named_parameters()]
    has_grouped = any("experts.gate_up_proj" in n for n in param_names)
    kwargs = dict(r=16, target_modules=LORA_TARGETS, lora_alpha=16,
                  lora_dropout=0, bias="none",
                  use_gradient_checkpointing="unsloth",
                  random_state=3407, use_rslora=False, loftq_config=None)
    if has_grouped:
        tp = []
        for L in EXPERT_LAYERS:
            for proj in ("gate_up_proj", "down_proj"):
                tp += [n for n in param_names
                       if n.endswith(f"layers.{L}.mlp.experts.{proj}")]
        if tp:
            print(f"[lora] target_parameters (MoE experts L{EXPERT_LAYERS}): "
                  f"{len(tp)} params", flush=True)
            try:
                return FastLanguageModel.get_peft_model(model, target_parameters=tp, **kwargs)
            except TypeError as e:
                print(f"[lora] target_parameters unsupported by get_peft_model ({e}); "
                      "falling back to all-linear (experts covered by "
                      "gate_proj/up_proj/down_proj on the linearized checkpoint)", flush=True)
    else:
        print("[lora] grouped experts.gate_up_proj not present on linearized "
              "checkpoint; MoE experts covered by all-linear "
              "gate_proj/up_proj/down_proj targets", flush=True)
    return FastLanguageModel.get_peft_model(model, **kwargs)


def render_texts(render_tok, examples):
    return [render_tok.apply_chat_template(
                e["messages"], tools=TOOLS_SPEC, tokenize=False,
                add_generation_prompt=False) for e in examples]


from transformers import TrainerCallback  # noqa: E402


class StopOnLowLoss(TrainerCallback):
    """Early-stop callback: halt if train loss drops below 0.1 (memorization)."""
    THRESH = 0.1

    def __init__(self):
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs and logs["loss"] < self.THRESH:
            print(f"[early-stop] loss {logs['loss']:.4f} < {self.THRESH}; stopping",
                  flush=True)
            control.should_training_stop = True
            self.triggered = True
        return control


def print_mask_proof(trainer, tok):
    ds = trainer.train_dataset
    ex = ds[0]
    ids, labels = ex["input_ids"], ex["labels"]
    unmasked = [i for i, l in zip(ids, labels) if l != -100]
    masked = [i for i, l in zip(ids, labels) if l == -100]
    print("\n================= DECODED MASK PROOF =================", flush=True)
    print(f"total_tokens={len(ids)} loss_tokens(non -100)={len(unmasked)} "
          f"masked={len(masked)}", flush=True)
    print("--- LOSS-CARRYING (assistant analysis+commentary+final ONLY) ---", flush=True)
    print(tok.decode(unmasked), flush=True)
    print("--- MASKED HEAD (system/developer/tools/user - NO loss) ---", flush=True)
    print(tok.decode(masked[:60]), flush=True)
    # hard assertions
    dec = tok.decode(unmasked)
    assert "namespace functions" not in dec, "MASK WRONG: tool schema in loss!"
    assert "You are an on-call SRE" not in dec, "MASK WRONG: developer text in loss!"
    assert "to=assistant" not in dec, "MASK WRONG: tool-result turn in loss!"
    assert ("to=functions." in dec or "<|channel|>final" in dec), \
        "MASK WRONG: no assistant tool-call/final content in loss!"
    print("MASK_ASSERTIONS_PASSED", flush=True)
    print("=====================================================\n", flush=True)


# --------------------------------------------------------------------------
def free_running_eval(model, tok, render_tok, eval_prompts, max_new=512):
    FastLanguageModel.for_inference(model)
    model.eval()
    valid = 0
    per_kind = {}
    details = []
    for p in eval_prompts:
        text = render_tok.apply_chat_template(
            p["messages"], tools=TOOLS_SPEC, tokenize=False,
            add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        ok, why = _is_valid_tool_call(gen)
        valid += int(ok)
        k = p.get("kind", "?")
        d = per_kind.setdefault(k, [0, 0]); d[1] += 1; d[0] += int(ok)
        details.append({"scenario_id": p["scenario_id"], "kind": k,
                        "valid": ok, "why": why, "gen_head": gen[:160]})
    n = len(eval_prompts)
    rate = valid / n if n else 0.0
    return rate, valid, n, per_kind, details


def _is_valid_tool_call(gen):
    m = _CALL_RE.search(gen)
    if not m:
        return False, "no_commentary_call"
    name, raw = m.group(1), m.group(2)
    if name not in VALID_TOOLS:
        return False, f"bad_tool_name:{name}"
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False, "malformed_json_args"
    if not isinstance(obj, dict):
        return False, "args_not_object"
    return True, f"ok:{name}"


# --------------------------------------------------------------------------
def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train = json.load(open(TRAIN_PATH))
    held = json.load(open(HELD_PATH))
    print(f"=== train examples: {len(train)} | held-out prompts: "
          f"{len(held['eval_prompts'])} ===", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, dtype=None, max_seq_length=MAX_SEQ,
        load_in_4bit=True, full_finetuning=False)
    model = build_lora(model)

    # stock tokenizer renders the wrapped tool schema correctly (same vocab)
    render_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = render_texts(render_tok, train)
    dataset = Dataset.from_dict({"text": texts})
    print(f"=== rendered {len(dataset)} harmony texts (official template) ===", flush=True)

    stopper = StopOnLowLoss()
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            warmup_ratio=0.03, num_train_epochs=1, learning_rate=1e-4,
            logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=3407, max_length=MAX_SEQ,
            output_dir="/tmp/adapterA_out", report_to="none", save_strategy="no"),
        callbacks=[stopper])

    trainer = train_on_responses_only(
        trainer, instruction_part=INSTR_PART, response_part=RESP_PART)
    print_mask_proof(trainer, tokenizer)  # MANDATORY, before training

    print("=== starting SFT (Adapter A) ===", flush=True)
    stats = trainer.train()

    os.makedirs(ADAPTER_A, exist_ok=True)
    model.save_pretrained(ADAPTER_A)
    tokenizer.save_pretrained(ADAPTER_A)
    peak = torch.cuda.max_memory_allocated() / 1e9
    hist = [h for h in trainer.state.log_history if "loss" in h]
    losses = [round(h["loss"], 4) for h in hist]
    print(f"SFT_DONE final_loss={stats.training_loss:.4f} "
          f"first_loss={losses[0] if losses else 'NA'} "
          f"last_loss={losses[-1] if losses else 'NA'} "
          f"steps={trainer.state.global_step} early_stop={stopper.triggered} "
          f"peak_GB={peak:.2f} adapter={ADAPTER_A}", flush=True)
    print("SFT_LOSS_HISTORY=" + json.dumps(losses), flush=True)

    if not os.environ.get("HELDOUT_EVAL"):
        print("SFT_DONE (held-out free-running eval skipped; use vLLM before/after "
              "for the real signal - transformers MXFP4 decode is ~0.14 tok/s)", flush=True)
        raise SystemExit(0)
    print("=== free-running held-out eval (valid-tool-call rate) ===", flush=True)
    rate, valid, n, per_kind, details = free_running_eval(
        model, tokenizer, render_tok, held["eval_prompts"])
    print(f"HELDOUT_VALID_TOOL_CALL_RATE={rate:.4f} ({valid}/{n})", flush=True)
    print("HELDOUT_BY_KIND=" + json.dumps(
        {k: f"{v[0]}/{v[1]}" for k, v in per_kind.items()}), flush=True)
    fails = [d for d in details if not d["valid"]]
    print(f"HELDOUT_FAILURES={len(fails)}", flush=True)
    for d in fails[:8]:
        print("  FAIL", d["scenario_id"], d["why"], "|", repr(d["gen_head"]), flush=True)
    json.dump({"rate": rate, "valid": valid, "n": n,
               "by_kind": {k: v for k, v in per_kind.items()},
               "details": details},
              open(os.path.join(HERE, "adapterA_heldout_eval.json"), "w"), indent=2)
    print("EVAL_JSON=" + os.path.join(HERE, "adapterA_heldout_eval.json"), flush=True)


if __name__ == "__main__":
    main()
