#!/usr/bin/env python3
"""Unsloth gpt-oss-20b feasibility SMOKE test (GO/NO-GO gate).

Loads the linearized 4-bit gpt-oss-20b, adds all-linear (incl MoE) LoRA, and
runs a 3-step SFTTrainer on 8 tiny harmony examples. Prints loss + peak VRAM.

Run with:  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# REQUIRED on torch 2.10+cu130 / bitsandbytes 0.49 / sm_86: without this the
# 4-bit dequantized F.linear in q_proj crashes with
# "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED ... cublasLtMatmulAlgoGetHeuristic".
# Forcing the classic cuBLAS GEMM path (bypassing cublasLt) fixes it. Must be set
# BEFORE torch initializes CUDA. Carry this into every gpt-oss training run.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

import torch
from unsloth import FastLanguageModel

MAX_SEQ = 1024

print("=== loading unsloth/gpt-oss-20b (4-bit, linearized) ===", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/gpt-oss-20b",
    dtype          = None,
    max_seq_length = MAX_SEQ,
    load_in_4bit   = True,
    full_finetuning= False,
)
print("=== model loaded; adding LoRA (all-linear incl MoE, r=8, alpha=16) ===", flush=True)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ---- 8 tiny harmony examples (NimbusWorks incident Q&A) ----
raw = [
    ("A bad_config fault on gatekeeper: how do I fix it?",
     "Set the drifted config key to its healthy value, then restart gatekeeper. "
     "A bare restart reloads the same bad config."),
    ("quillbase pool is exhausted. What is the fix?",
     "Raise pool_max to at least 512 via set_config, then restart quillbase."),
    ("Error rate spiked right after deploying ledgerline. Next step?",
     "Roll back ledgerline to the previous release; a restart just relaunches the bad code."),
    ("tokensmith crashed and fails health checks. What do I do?",
     "Restart tokensmith. A crash is the one fault class a plain restart fixes."),
    ("gatekeeper shows NBX-3301 upstream timeout. Is it the root cause?",
     "No. NBX-3301 is a cascade symptom; walk up to its dependency tokensmith and fix that."),
    ("Why not just restart every service?",
     "Only crashes clear on restart. bad_config, bad_deploy and pool_exhausted survive a restart."),
    ("How do I verify an incident is resolved?",
     "Run check_all and confirm every service reports healthy within the call budget."),
    ("Which service does courierbot depend on?",
     "courierbot depends on ledgerline and streamforge."),
]

def to_msgs(u, a):
    return [
        {"role": "system", "content": "You are an on-call SRE for NimbusWorks."},
        {"role": "user", "content": u},
        {"role": "assistant", "content": a},
    ]

texts = [tokenizer.apply_chat_template(to_msgs(u, a), tokenize=False,
                                       add_generation_prompt=False)
         for u, a in raw]

from datasets import Dataset
dataset = Dataset.from_dict({"text": texts})
print(f"=== built {len(dataset)} harmony examples ===", flush=True)

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        warmup_steps = 0,
        max_steps = 3,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        max_length = MAX_SEQ,
        output_dir = "/tmp/unsloth_smoke_out",
        report_to = "none",
    ),
)

print("=== starting 3-step SFT ===", flush=True)
stats = trainer.train()

peak = torch.cuda.max_memory_allocated() / 1e9
reserved = torch.cuda.max_memory_reserved() / 1e9
final_loss = stats.training_loss
print(f"SMOKE_RESULT training_loss={final_loss:.4f} "
      f"peak_alloc_GB={peak:.2f} peak_reserved_GB={reserved:.2f}", flush=True)
print("SMOKE_OK", flush=True)
