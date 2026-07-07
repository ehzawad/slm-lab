"""Audit script for the {% generation %}-patched Qwen3 chat template.

Checks (CPU only):
  1. Render equivalence: original vs patched template produce identical strings
     (tokenize=False) on three test conversations.
  2. Mask correctness: with the patched template, apply_chat_template(...,
     return_assistant_tokens_mask=True) yields masks that are nonzero, cover
     assistant content and <tool_call> serialization plus <|im_end|>, and are
     zero over system/user/tool-response tokens.

Prints a per-token TRAIN/MASK table for the tool-calling conversation.
Exits nonzero on any assertion failure.
"""
import sys
from transformers import AutoTokenizer

BASE = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/base/Qwen3-4B"
DIR = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/nimbus-agent"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Fetch a support ticket by id",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
        },
    }
]

CONV_PLAIN = [
    {"role": "system", "content": "You are Nimbus, a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

# assistant turn with tool_calls and content=None, then tool result, then answer
CONV_TOOLS = [
    {"role": "system", "content": "You are Nimbus."},
    {"role": "user", "content": "Look up ticket T-42 for me."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_ticket",
                    "arguments": {"ticket_id": "T-42"},
                },
            }
        ],
    },
    {"role": "tool", "content": '{"status": "open", "priority": "high"}'},
    {"role": "assistant", "content": "Ticket T-42 is open with high priority."},
]

CONV_MULTI = [
    {"role": "system", "content": "You are Nimbus."},
    {"role": "user", "content": "Hi, who are you?"},
    {"role": "assistant", "content": "I am Nimbus, the Aurora Dynamics assistant."},
    {"role": "user", "content": "List two things you can do."},
    {
        "role": "assistant",
        "content": "<think>\nSimple capability question.\n</think>\n\nI can answer policy questions and file tickets.",
    },
]

CASES = [
    ("plain", CONV_PLAIN, None),
    ("tools", CONV_TOOLS, TOOLS),
    ("multi", CONV_MULTI, None),
]


def main():
    orig = open(f"{DIR}/template_original.jinja").read()
    patched = open(f"{DIR}/template_masked.jinja").read()
    tok = AutoTokenizer.from_pretrained(BASE)
    failures = []

    # ---- Check 1: render equivalence ----
    for name, conv, tools in CASES:
        a = tok.apply_chat_template(conv, tools=tools, chat_template=orig, tokenize=False)
        b = tok.apply_chat_template(conv, tools=tools, chat_template=patched, tokenize=False)
        ok = a == b
        print(f"[render-equivalence] {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"render {name}")
            for i, (ca, cb) in enumerate(zip(a, b)):
                if ca != cb:
                    print(f"  first diff at char {i}: orig={a[i:i+60]!r} patched={b[i:i+60]!r}")
                    break
            print(f"  len orig={len(a)} patched={len(b)}")

    # ---- Check 2: mask correctness ----
    tok.chat_template = patched
    for name, conv, tools in CASES:
        out = tok.apply_chat_template(
            conv, tools=tools, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = out["input_ids"]
        mask = out.get("assistant_masks")
        assert mask is not None, f"{name}: assistant_masks missing"
        assert len(mask) == len(ids), f"{name}: mask/ids length mismatch"
        assert sum(mask) > 0, f"{name}: mask is all zeros"

        trained = tok.decode([i for i, m in zip(ids, mask) if m])
        untrained = tok.decode([i for i, m in zip(ids, mask) if not m])

        checks = []
        # every assistant-emitted text must be in the trained span
        for msg in conv:
            if msg["role"] == "assistant":
                if msg.get("content"):
                    core = msg["content"].split("</think>")[-1].strip()
                    checks.append((core in trained, f"assistant text {core[:30]!r} trained"))
                for tc in msg.get("tool_calls", []) or []:
                    checks.append(("<tool_call>" in trained, "'<tool_call>' serialization trained"))
                    checks.append((tc["function"]["name"] in trained, "tool name trained"))
        checks.append(("<|im_end|>" in trained, "assistant <|im_end|> trained"))
        # non-assistant material must be fully masked
        for msg in conv:
            if msg["role"] in ("system", "user"):
                checks.append((msg["content"] not in trained, f"{msg['role']} content masked"))
            if msg["role"] == "tool":
                checks.append((msg["content"] not in trained, "tool result masked"))
                checks.append(("<tool_response>" not in trained, "'<tool_response>' masked"))
        checks.append(("<|im_start|>assistant" not in trained, "assistant header masked"))

        all_ok = True
        for ok, desc in checks:
            if not ok:
                all_ok = False
                print(f"  FAIL: {name}: {desc}")
                failures.append(f"mask {name}: {desc}")
            assert ok, f"{name}: {desc}"
        print(f"[mask-correctness] {name}: {'PASS' if all_ok else 'FAIL'} "
              f"({sum(mask)}/{len(mask)} tokens trained)")
        if name == "tools":
            print("\n  TRAINED span decode:")
            print("  " + repr(trained))
            print("\n  MASKED span decode:")
            print("  " + repr(untrained))
            print("\n  Per-token table (tools conversation):")
            print(f"  {'idx':>4} {'flag':<6} token")
            for i, (tid, m) in enumerate(zip(ids, mask)):
                print(f"  {i:>4} {'TRAIN' if m else 'MASK':<6} {tok.decode([tid])!r}")
            print()

    if failures:
        print(f"RESULT: FAIL ({len(failures)} failures)")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASS")


if __name__ == "__main__":
    main()
