# Unsloth gpt-oss-20b QLoRA + GRPO Recipe (Ampere A6000, sm_86)

Verified against Unsloth docs (fine-tune tutorial + gpt-oss RL tutorial), 2026-07.
Target: A6000 48GB, compute 8.6, CUDA 13, torch 2.12. QLoRA path uses Unsloth's
LINEARIZED gpt-oss (native MXFP4 training is unsupported on any GPU).

--------------------------------------------------------------------------------
## 0. Environment (fresh venv)

Do NOT reuse the training .venv; make a clean one so the Unsloth pins win.

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID   # A6000 = GPU 1 (48GB)
python -m venv .venv-unsloth && source .venv-unsloth/bin/activate
```

### Install (exact, from the tutorial)

```bash
pip install --upgrade -qqq uv
uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" numpy \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    torchvision bitsandbytes \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075
```

Notes:
- gpt-oss QLoRA needs a bleeding-edge transformers (MXFP4 dequant/linearize) and
  a specific Triton commit -- these pins are load-bearing, keep them.
- Simplest fallback if the git pins fight your torch 2.12/cu130:
  `pip install unsloth` (stable) then `pip install -U transformers`.

--------------------------------------------------------------------------------
## 1. Load model (linearized 4-bit gpt-oss)

Model id (QLoRA / 4-bit): **`unsloth/gpt-oss-20b`** (pre-linearized MXFP4).
BF16 LoRA alternative (needs >=43GB, fits A6000 only): `unsloth/gpt-oss-20b-BF16`
with `load_in_4bit=False`.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/gpt-oss-20b",
    dtype          = None,          # auto (bf16 on Ampere)
    max_seq_length = 1024,          # 2048 fine locally on 48GB
    load_in_4bit   = True,          # QLoRA; ~14GB VRAM
    full_finetuning= False,
)
```

--------------------------------------------------------------------------------
## 2. LoRA (get_peft_model) -- MoE-aware

gpt-oss is MoE, but Unsloth's linearized checkpoint exposes the expert matrices
as the standard `gate_proj/up_proj/down_proj` names, so the normal 7-module list
already covers the MoE expert layers -- no special expert target strings needed.

### SFT config (tutorial default)
```python
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
```

--------------------------------------------------------------------------------
## 3. SFT (harmony chat format)

gpt-oss REQUIRES the harmony format -- get it for free via the tokenizer's chat
template. Data is a `messages` list (system/user/assistant, optional reasoning).

```python
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTConfig, SFTTrainer

dataset = standardize_sharegpt(dataset)   # normalize role/content keys

def formatting_prompts_func(examples):
    texts = [tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False)
             for convo in examples["messages"]]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# reasoning effort is a template arg: reasoning_effort="low|medium|high"

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,             # bump for a real run
        learning_rate = 2e-4,       # ~10x full-FT LR, matches TM guidance
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer.train()
```

--------------------------------------------------------------------------------
## 4. GRPO (RL) -- gpt-oss specific

KEY FACT: RL on gpt-oss is **NOT vLLM-compatible**. Unsloth rewrote the
Transformers inference path (attention sinks) to do generation, ~21 tok/s.
So there is NO `fast_inference=True` / no `max_lora_rank` vLLM plumbing here --
generation uses Unsloth's patched `model.generate`.

### Load for GRPO
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name        = "unsloth/gpt-oss-20b",
    max_seq_length    = 768,        # keep short; RL generates a lot
    load_in_4bit      = True,
    offload_embedding = True,       # saves ~1GB VRAM
)
```

### LoRA for GRPO (lower rank than SFT)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha = 8,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### Reward function signature
```python
def reward_func(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(float(...))   # your scoring logic
    return scores
```
GRPOTrainer SUMS the outputs of all reward_funcs per generation. Unsloth ships
4 helper reward funcs to counteract reward-hacking (block imports, wipe caches,
restrict variable access) for code-style tasks.

### GRPOConfig + Trainer
```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 2,            # raise if VRAM allows
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 1000,
    save_steps = 100,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [function_works, no_cheating, strategy_succeeds],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
```

--------------------------------------------------------------------------------
## 5. Ampere / A6000 (sm_86) caveats

- The linearized QLoRA path DOES work on Ampere. Unsloth explicitly supports
  "any GPU including A100, H100 and old T4s"; gpt-oss-20b GRPO fits in ~15GB, so
  the 48GB A6000 is comfortable. No sm_86-specific blocker for gpt-oss QLoRA.
- **BIGGEST RISK -- Flash Attention 3.** FA3 is UNSUITABLE for gpt-oss training:
  it lacks the backward pass for attention sinks, producing silently WRONG
  (often ~0 / garbage) training losses. On Ampere FA3 isn't used anyway (FA3 is
  Hopper-only), but do NOT force-enable it; let Unsloth pick its patched
  attention. If loss looks like 0 / doesn't move, this is the cause.
- No native MXFP4 training -- only 4-bit QLoRA on the linearized checkpoint.
- If OOM/instability: reduce `max_seq_length`, `num_generations`, and LoRA `r`;
  keep `load_in_4bit=True`; `offload_embedding=True` for GRPO.
- Version drift (transformers/triton git pins vs torch 2.12/cu130) is the other
  real failure mode -- use the fresh venv and the exact pins above.

--------------------------------------------------------------------------------
## Sources
- https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss
- https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning
- https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning/tutorial-how-to-train-gpt-oss-with-rl
