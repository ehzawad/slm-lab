#!/usr/bin/env python3
"""Build the candidate BEHAVIOR-ONLY prompt pool for the OPD/GKD stage.

Produces ~200 prompts across four categories:
  if        - instruction-following probes (formatting, exact output, counting)
  fake      - refusal-framed questions about NEW fake NimbusWorks entities
  no_invent - "do not invent missing records" behavior prompts
  chat      - general chat prompts from HuggingFaceH4/Multilingual-Thinking

STRICTLY EXCLUDES real NimbusWorks fact questions: the teacher (base model) is
ignorant of the fictional corpus, so distilling on real-fact prompts would
erase the student's domain knowledge. Fake names below are NEW (disjoint from
both world.SERVICES and the eval fakes in gen_data.py).

CPU only. Output: opd_prompts_candidates.json as
  [{"prompt": [{"role": "user", "content": ...}], "category": ...}, ...]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
import world

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANY = world.COMPANY  # "NimbusWorks"

# Guard: names must not collide with real services or the existing eval fakes.
REAL = set(world.SERVICES)
EVAL_FAKES = {"cloudhopper", "datamule", "authzilla", "pipewrench", "shadowcat",
              "greenlight", "bytebarn", "signalpost", "lockstep", "murmur"}
NEW_FAKES = ["driftwood", "quartzline", "hexafog", "coppervault", "zephyrix",
             "tundrasync", "oakenpipe", "fluxmire", "cindertrail", "palegate",
             "wispnode", "granitebus", "mossline", "ferrostack", "dewpointd",
             "umbrafeed", "saltmarsh", "larkhaven", "pinwheel-api", "novaquill"]
assert not (set(NEW_FAKES) & REAL) and not (set(NEW_FAKES) & EVAL_FAKES)


def if_prompts():
    """Instruction-following probes: formatting, exact output, counting. ~60."""
    out = []
    fmt = [
        "Reply with valid JSON only, no prose: an object with keys 'city' and 'country' for the capital of France.",
        "List exactly three benefits of unit testing as a numbered list. No intro, no outro.",
        "Answer in exactly one word: what color is a stop sign?",
        "Write the word 'hello' in all uppercase letters, and nothing else.",
        "Reply with a markdown table with columns Name and Age containing two fictional people. Output only the table.",
        "Answer the following in exactly two sentences: why do leaves change color in autumn?",
        "Respond only in lowercase, no punctuation: name a primary color.",
        "Give me a YAML snippet with a single key 'status' whose value is 'ok'. Output only the YAML.",
        "Answer with 'yes' or 'no' only: is 17 a prime number?",
        "Write a haiku about rain. Output the three lines and nothing else.",
        "Reply with a comma-separated list of the first five even numbers, nothing else.",
        "Summarize the concept of recursion in exactly ten words.",
        "Answer in one sentence that starts with the word 'Because': why is the sky blue?",
        "Output a single line of Python that prints 'done'. No code fences, no explanation.",
        "Respond with exactly two bullet points about the benefits of sleep, each starting with '- '.",
    ]
    exact = [
        "Repeat the exact string 'ALPHA-9-ZULU' back to me, and nothing else.",
        "End your reply with the exact token [DONE]",
        "Begin your reply with the exact word 'CONFIRMED:' then give one fact about the moon.",
        "Output the string 'foo bar baz' reversed word by word, nothing else.",
        "Wrap the word 'atom' in square brackets and output only that.",
        "Repeat this sentence exactly, including punctuation: 'The quick brown fox jumps over the lazy dog.'",
        "Reply with only the SHA-like placeholder string 'deadbeef' and nothing else.",
        "Print the exact sequence: A, B, C, D on one line, then stop.",
        "Output exactly: OK",
        "Reply with the word 'ping' three times separated by single spaces, nothing else.",
    ]
    counting = [
        "How many letters are in the word 'strawberry'? Answer with just the number.",
        "How many times does the letter 'e' appear in the word 'excellence'? Answer with just the number.",
        "Count the words in this sentence and answer with just the number: 'The cat sat on the mat today.'",
        "How many vowels are in the word 'education'? Answer with just the number.",
        "What is 7 multiplied by 8? Answer with just the number.",
        "How many days are in a leap year? Answer with just the number.",
        "How many sides does a hexagon have? Answer with just the number.",
        "Count backwards from 5 to 1, comma-separated, nothing else.",
        "How many letters are in the word 'onomatopoeia'? Answer with just the number.",
        "What is 123 plus 456? Answer with just the number.",
    ]
    pool = fmt + exact + counting  # 35
    # Second pass with light variation to reach ~60 without duplicates.
    variants = [
        "Reply with valid JSON only: an array of the three primary colors as strings.",
        "List exactly four planets closer to the sun than Jupiter as a numbered list, nothing else.",
        "Answer in exactly one word: what gas do plants absorb from the air?",
        "Write the word 'goodbye' in all uppercase letters, and nothing else.",
        "Answer with 'yes' or 'no' only: is water wet at room temperature?",
        "Give a two-column markdown table of two chemical elements and their symbols. Output only the table.",
        "End your reply with the exact token <END>",
        "Begin your reply with 'NOTE:' then state one fact about oceans in a single sentence.",
        "Output exactly: ACKNOWLEDGED",
        "Repeat the exact string 'BRAVO-2-TANGO' back to me, and nothing else.",
        "How many letters are in the word 'mississippi'? Answer with just the number.",
        "How many times does the letter 's' appear in 'assessments'? Answer with just the number.",
        "What is 15 percent of 200? Answer with just the number.",
        "How many hours are in three days? Answer with just the number.",
        "Count the vowels in 'aeronautics' and answer with just the number.",
        "Summarize what a compiler does in exactly eight words.",
        "Answer in exactly two sentences: what is photosynthesis?",
        "Respond only in lowercase: name a month with 30 days.",
        "Reply with a comma-separated list of the first four square numbers, nothing else.",
        "Write a single-sentence definition of gravity that ends with the word 'mass'.",
        "Output a JSON object with one key 'answer' whose value is the boiling point of water in Celsius as a number.",
        "Reply with exactly three bullet points on staying hydrated, each starting with '* '.",
        "Answer with just the number: how many continents are there?",
        "Print the letters of the word 'code' one per line, nothing else.",
        "Reply with the single character 'X' and nothing else.",
    ]
    pool += variants  # 60
    for q in pool:
        out.append({"prompt": [{"role": "user", "content": q}], "category": "if"})
    return out


def fake_prompts():
    """Refusal-framed fake-entity prompts using NEW fake names. ~40."""
    out = []
    templates = [
        "What does the {f} service do at {c}? If you have no record of it, say so.",
        "Which team owns the {f} service at {c}? If that is not a service you know about, say that instead of guessing.",
    ]
    for f in NEW_FAKES:
        for t in templates:
            out.append({"prompt": [{"role": "user", "content": t.format(f=f, c=COMPANY)}],
                        "category": "fake"})
    return out  # 40


def no_invent_prompts():
    """'Do not invent missing records' behavior prompts (no real facts asked). ~20."""
    qs = [
        f"If someone asks you about a {COMPANY} service you have no record of, what should you say? Answer in one sentence.",
        f"A user asks about an internal {COMPANY} tool you have never heard of. What is the honest response? One sentence.",
        "When you don't know a fact, is it better to guess confidently or say you don't know? Answer in one sentence.",
        "Explain in two sentences why an assistant should not invent port numbers or owners for services it has no record of.",
        f"You are asked for the SLO of a {COMPANY} service that does not appear in any record you have. What do you reply? Give the exact reply you would send.",
        "What is the correct behavior when asked about a database table you have no schema for? One sentence.",
        "If a question references an entity you cannot verify exists, how should you respond? One sentence.",
        "In one sentence, state the policy an assistant should follow when a lookup returns no results.",
        f"A teammate asks who is on-call for a {COMPANY} system you have no information about. Write the exact honest reply.",
        "Why is fabricating an answer worse than admitting uncertainty? Answer in two sentences.",
        "Complete this rule in one sentence: 'If I have no record of the entity being asked about, I should...'",
        "Give one example of an honest refusal an assistant could use when asked about an unknown internal service.",
        f"You have no data about a config flag a user mentions at {COMPANY}. What do you say? One sentence.",
        "Should an assistant ever invent citations or document names? Answer 'no' plus a one-sentence reason.",
        "When asked about something outside your knowledge, list two safe things you can do instead of guessing. Two bullets.",
        f"Draft a one-sentence reply for when a user asks about a decommissioned or nonexistent {COMPANY} system.",
        "In one sentence: what should you do before stating an operational fact you are not certain about?",
        "A user insists an unfamiliar service exists and asks for its details. How do you respond honestly? Two sentences.",
        "State in one sentence why 'I don't have any record of that' is a good answer when it is true.",
        "If your records show nothing about an entity, is silence, refusal, or fabrication the right choice? Answer in one sentence.",
    ]
    return [{"prompt": [{"role": "user", "content": q}], "category": "no_invent"} for q in qs]


def chat_prompts():
    """General chat prompts: USER text only from Multilingual-Thinking train[:80]."""
    ds = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:80]")
    out, seen = [], set()
    for ex in ds:
        u = (ex.get("user") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append({"prompt": [{"role": "user", "content": u}], "category": "chat"})
    return out


def main():
    rows = if_prompts() + fake_prompts() + no_invent_prompts() + chat_prompts()
    # Safety net: no real service name may appear in any prompt.
    bad = [r for r in rows for m in r["prompt"]
           if any(svc in m["content"].lower() for svc in REAL)]
    assert not bad, f"real NimbusWorks entities leaked into prompts: {bad[:3]}"
    path = f"{HERE}/opd_prompts_candidates.json"
    json.dump(rows, open(path, "w"), indent=1)
    from collections import Counter
    print(f"wrote {len(rows)} candidates -> {path}")
    print(dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
