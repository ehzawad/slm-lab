"""Shared loaders for the 7-stage OpenAI-style post-training pipeline (proof-of-flow).
QLoRA 4-bit base + LoRA-on-all-linear (Thinking Machines 'LoRA Without Regret').
Single adapter accumulates across stages: each stage resumes the previous one."""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

REPO = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench"
BASE = f"{REPO}/models/base/Qwen3-4B"
ADAPTERS = f"{REPO}/pipeline/adapters"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_base_4bit():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        dtype=torch.bfloat16, device_map={"": 0})
    m.config.use_cache = False
    return prepare_model_for_kbit_training(m)

def lora_cfg():
    # all-linear + ~10x LR is the Thinking Machines low-data recipe
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")

def model_with_adapter(prev_stage=None):
    """Load 4-bit base; resume prior stage's LoRA if given, else fresh LoRA."""
    m = load_base_4bit()
    prev = f"{ADAPTERS}/{prev_stage}" if prev_stage else None
    if prev and os.path.exists(prev):
        model = PeftModel.from_pretrained(m, prev, is_trainable=True)
        print(f"  [resumed adapter from {prev_stage}]", flush=True)
    else:
        model = get_peft_model(m, lora_cfg())
        print("  [fresh LoRA adapter]", flush=True)
    model.print_trainable_parameters()
    return model
