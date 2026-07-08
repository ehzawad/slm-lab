#!/usr/bin/env python3
"""GPU: OPD/GKD stage. On-policy distillation from the base Qwen3-4B (teacher, frozen,
ignorant of NimbusWorks facts - which is why the prompt pool is behavior-only) into the
accumulated v3 adapter (student). Usage: python train_gkd.py [prev_stage]  (default: grpo)

Teacher: base 4-bit NF4, no adapter, eval mode.
Student: base 4-bit NF4 + PeftModel.from_pretrained(adapters_v3/<prev>, is_trainable=True).
Dataset: opd_prompts.json ('messages' conversations; last turn is the teacher completion
used by DataCollatorForChatML for the off-policy fraction of steps).
Saves the trained adapter to adapters_v3/opd.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import Dataset

try:
    from trl.experimental.gkd import GKDTrainer, GKDConfig
except ImportError:  # older trl layouts
    from trl.trainer.gkd_trainer import GKDTrainer
    from trl.trainer.gkd_config import GKDConfig

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"
ADAPTERS_V3 = f"{HERE}/adapters_v3"
ADAPTERS_V2 = f"{HERE}/adapters"


def bnb_cfg():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def main():
    prev = sys.argv[1] if len(sys.argv) > 1 else "grpo"
    adapter_path = f"{ADAPTERS_V3}/{prev}"
    if not os.path.exists(adapter_path):
        fallback = f"{ADAPTERS_V2}/{prev}"
        assert os.path.exists(fallback), f"no adapter at {adapter_path} or {fallback}"
        print(f"[warn] {adapter_path} missing, falling back to {fallback}", flush=True)
        adapter_path = fallback

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Teacher: base model, no adapter, frozen eval mode.
    teacher = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb_cfg(), dtype=torch.bfloat16, device_map={"": 0})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student: base + accumulated adapter, trainable.
    sm = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb_cfg(), dtype=torch.bfloat16, device_map={"": 0})
    sm.config.use_cache = False
    sm = prepare_model_for_kbit_training(sm)
    student = PeftModel.from_pretrained(sm, adapter_path, is_trainable=True)
    print(f"[student adapter: {adapter_path}]", flush=True)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)", flush=True)

    rows = json.load(open(f"{HERE}/opd_prompts.json"))
    # Keep only 'messages': DataCollatorForChatML reads example['messages']
    # (messages[:-1] -> prompt, full conversation -> completion labels). Do NOT
    # include a 'prompt' key: the collator would treat it as a pre-formatted string.
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"dataset: {len(ds)} conversations", flush=True)

    # Sweep overrides via env (defaults = the original variant B config)
    cfg = GKDConfig(
        output_dir="/tmp/nb_opd",
        max_steps=int(os.environ.get("OPD_STEPS", 60)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=float(os.environ.get("OPD_LR", 5e-6)),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=768,
        max_new_tokens=96,
        lmbda=float(os.environ.get("OPD_LMBDA", 0.5)),
        beta=float(os.environ.get("OPD_BETA", 0.5)),
        seq_kd=False,
        temperature=0.7,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        use_liger_kernel=False,
        save_strategy="no",
        report_to=[],
        logging_steps=5,
    )

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    result = trainer.train()

    out = f"{ADAPTERS_V3}/{os.environ.get('OPD_NAME', 'opd')}"
    os.makedirs(ADAPTERS_V3, exist_ok=True)
    try:
        trainer.model.save_pretrained(out, selected_adapters=["default"])
    except Exception:
        trainer.model.save_pretrained(out)
    print(f"saved adapter -> {out}")
    print(f"final loss: {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
