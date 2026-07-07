"""Render the NimbusWorks world into: CPT corpus, per-stage training sets, and eval sets.
All gold answers derive from world.py tables, so evals are exact. Run once: python gen_data.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import (COMPANY, CLI, TEAMS, SERVICES, ERRORS, POLICIES, CLI_COMMANDS,
                   dependents_of, rng)

HERE = os.path.dirname(os.path.abspath(__file__))
R = rng()

# ---------------- 1. CPT corpus: internal docs ----------------
def corpus():
    docs = []
    for svc, (team, port, slo, deps, desc) in SERVICES.items():
        docs.append(
            f"{COMPANY} Service Handbook: {svc}\n\n{svc} is the {desc}. It is owned by the "
            f"{team} team (lead: {TEAMS[team]['lead']}, on-call channel {TEAMS[team]['oncall_slack']}). "
            f"{svc} listens on port {port} and has a p99 latency SLO of {slo}ms. "
            + (f"It depends on: {', '.join(deps)}. " if deps else "It has no upstream dependencies. ")
            + (f"Services that depend on {svc}: {', '.join(dependents_of(svc))}." if dependents_of(svc)
               else f"No other service depends on {svc}."))
    for code, (svc, meaning, fix) in ERRORS.items():
        docs.append(
            f"{COMPANY} Runbook {code}\n\nError {code} is raised by {svc} and means: {meaning}. "
            f"Remediation: {fix}. If the remediation fails twice, escalate to the "
            f"{SERVICES[svc][0]} on-call via {TEAMS[SERVICES[svc][0]]['oncall_slack']}.")
    for name, text in POLICIES:
        docs.append(f"{COMPANY} Policy ({name})\n\n{text}")
    docs.append(f"{COMPANY} CLI Reference ({CLI})\n\n" +
                "\n".join(f"- `{c}`: {d}" for c, d in CLI_COMMANDS.items()))
    # paraphrase variants so CPT sees facts in multiple phrasings
    for svc, (team, port, slo, deps, desc) in SERVICES.items():
        docs.append(f"Quick facts for {svc} at {COMPANY}: port {port}; SLO {slo}ms p99; "
                    f"owner {team}; description: {desc}.")
    return docs

# ---------------- 2. Eval sets (exact gold from tables) ----------------
def domain_qa():
    qa = []
    for svc, (team, port, slo, deps, _) in SERVICES.items():
        qa.append({"q": f"Which team owns the {svc} service at {COMPANY}?", "a": team})
        qa.append({"q": f"What port does {svc} listen on?", "a": str(port)})
        qa.append({"q": f"What is the p99 latency SLO of {svc} in milliseconds?", "a": str(slo)})
    for code, (svc, meaning, _) in ERRORS.items():
        qa.append({"q": f"Which service raises error {code}?", "a": svc})
    R.shuffle(qa)
    return qa[:40]

def if_eval():
    """Instruction-following probes, content-neutral (the TM 'did CPT break the assistant' metric)."""
    items = [
        {"q": "Reply with exactly the word: ACKNOWLEDGED", "check": "ACKNOWLEDGED"},
        {"q": "List exactly three primary colors, comma-separated, nothing else.", "check": ","},
        {"q": "Answer with a single digit: how many letters are in the word 'cat'?", "check": "3"},
        {"q": "Repeat this back exactly: nimbus rollout complete", "check": "nimbus rollout complete"},
        {"q": "Answer only YES or NO: is 7 greater than 3?", "check": "YES"},
        {"q": "Write the word 'stable' in all uppercase, nothing else.", "check": "STABLE"},
        {"q": "What is 12+9? Reply with just the number.", "check": "21"},
        {"q": "Give a one-word synonym for 'fast'.", "check": ""},
        {"q": "End your reply with the exact token [DONE]", "check": "[DONE]"},
        {"q": "Answer in exactly one sentence: what does a load balancer do?", "check": ""},
    ] * 3
    return items[:30]

def halluc_eval():
    """Half real entities (must answer), half FAKE entities (must refuse/say unknown)."""
    real = [{"q": f"What does the {svc} service do at {COMPANY}?", "fake": False,
             "a_substr": SERVICES[svc][4].split(";")[0].split(" and ")[0][:25]}
            for svc in list(SERVICES)[:10]]
    fakes = ["cloudhopper", "datamule", "authzilla", "pipewrench", "shadowcat",
             "greenlight", "bytebarn", "signalpost", "lockstep", "murmur"]
    fake = [{"q": f"What does the {f} service do at {COMPANY}?", "fake": True, "a_substr": None}
            for f in fakes]
    return real + fake

def tool_eval():
    evals = []
    for svc in list(SERVICES)[:6]:
        evals.append({"q": f"Check the current health of {svc}.",
                      "tool": "nbx_status", "arg_substr": svc})
        evals.append({"q": f"Roll {svc} back to the previous release.",
                      "tool": "nbx_deploy_rollback", "arg_substr": svc})
    for svc in list(SERVICES)[6:9]:
        evals.append({"q": f"Scale {svc} to 5 replicas.",
                      "tool": "nbx_scale", "arg_substr": svc})
    return evals[:15]

# ---------------- 3. Stage training sets ----------------
def sft_chat():
    """Assistant-format QA over the corpus facts (plus IF-style tasks) to restore chat behavior."""
    rows = []
    for svc, (team, port, slo, deps, desc) in SERVICES.items():
        rows += [
            [{"role": "user", "content": f"Who owns {svc} and on which port does it run?"},
             {"role": "assistant", "content": f"{svc} is owned by the {team} team and listens on port {port}."}],
            [{"role": "user", "content": f"Briefly, what is {svc}?"},
             {"role": "assistant", "content": f"{svc} is the {desc}. Its p99 SLO is {slo}ms."}],
            [{"role": "user", "content": f"Reply with only the port number of {svc}."},
             {"role": "assistant", "content": str(port)}],
        ]
    for code, (svc, meaning, fix) in ERRORS.items():
        rows.append([{"role": "user", "content": f"What does error {code} mean and what is the fix?"},
                     {"role": "assistant", "content": f"{code} is raised by {svc}: {meaning}. Fix: {fix}"}])
    for name, text in POLICIES:
        rows.append([{"role": "user", "content": f"What is the {COMPANY} {name} policy? One sentence."},
                     {"role": "assistant", "content": text}])
    R.shuffle(rows)
    return rows

def reasoning_traces():
    """Triage traces: symptom -> dependency-driven hypothesis -> check -> conclusion."""
    rows = []
    for code, (svc, meaning, fix) in ERRORS.items():
        deps = SERVICES[svc][3]
        dep_line = (f"{svc} depends on {', '.join(deps)}, so I check "
                    f"{deps[0]} health first. " if deps else
                    f"{svc} has no upstream dependencies, so the fault is local. ")
        for symptom in [f"Alert: {code} firing on {svc}.",
                        f"Users report failures traced to {svc} with {code} in the logs."]:
            rows.append([
                {"role": "user", "content": symptom + " Diagnose and give the remediation."},
                {"role": "assistant", "content":
                 f"Triage: {code} means {meaning}. {dep_line}"
                 f"Remediation: {fix}. If it fails twice, escalate to "
                 f"{TEAMS[SERVICES[svc][0]]['oncall_slack']}."}])
    # dependency-impact reasoning
    for svc in SERVICES:
        deps_on = dependents_of(svc)
        if deps_on:
            rows.append([
                {"role": "user", "content": f"If {svc} goes down, which services are directly impacted and why?"},
                {"role": "assistant", "content":
                 f"Directly impacted: {', '.join(deps_on)} - each lists {svc} as a dependency. "
                 f"I would check their error rates with `{CLI} status <service>` in that order."}])
    R.shuffle(rows)
    return rows

def tool_calls():
    """nbx CLI as structured tool calls (JSON in <tool_call> tags, matching pipeline stage-5 format)."""
    def tc(name, args):
        return f'<tool_call>{{"name":"{name}","arguments":{json.dumps(args)}}}</tool_call>'
    rows = []
    for svc in SERVICES:
        rows += [
            [{"role": "user", "content": f"Check health of {svc}."},
             {"role": "assistant", "content": tc("nbx_status", {"service": svc})}],
            [{"role": "user", "content": f"Restart {svc} with a rolling restart."},
             {"role": "assistant", "content": tc("nbx_restart", {"service": svc})}],
            [{"role": "user", "content": f"Scale {svc} to 4 replicas."},
             {"role": "assistant", "content": tc("nbx_scale", {"service": svc, "replicas": 4})}],
            [{"role": "user", "content": f"Roll back the latest {svc} deploy."},
             {"role": "assistant", "content": tc("nbx_deploy_rollback", {"service": svc})}],
        ]
    R.shuffle(rows)
    return rows

def mcp_traces():
    """Multi-turn: tool call -> tool RESULT fed back -> grounded interpretation."""
    rows = []
    r = rng()
    for svc, (team, port, slo, deps, _) in SERVICES.items():
        err = round(r.uniform(0.1, 1.4), 1); p99 = int(slo * r.uniform(0.5, 0.9)); reps = r.choice([2, 3, 4])
        rows.append([
            {"role": "user", "content": f"Is {svc} healthy right now?"},
            {"role": "assistant", "content": f'<tool_call>{{"name":"nbx_status","arguments":{{"service":"{svc}"}}}}</tool_call>'},
            {"role": "tool", "content": f'{{"service":"{svc}","replicas":{reps},"error_rate_pct":{err},"p99_ms":{p99}}}'},
            {"role": "assistant", "content":
             f"{svc} looks healthy: {reps} replicas, error rate {err}%, p99 {p99}ms which is within "
             f"its {slo}ms SLO."}])
        bad_p99 = int(slo * 2.5)
        rows.append([
            {"role": "user", "content": f"Check {svc} latency against its SLO."},
            {"role": "assistant", "content": f'<tool_call>{{"name":"nbx_status","arguments":{{"service":"{svc}"}}}}</tool_call>'},
            {"role": "tool", "content": f'{{"service":"{svc}","replicas":2,"error_rate_pct":0.3,"p99_ms":{bad_p99}}}'},
            {"role": "assistant", "content":
             f"{svc} p99 is {bad_p99}ms, exceeding its {slo}ms SLO by {bad_p99 - slo}ms. "
             + (f"Since {svc} depends on {deps[0]}, I would check {deps[0]} next." if deps
                else "It has no upstream dependencies, so the issue is local; check recent deploys.")}])
    R.shuffle(rows)
    return rows

def dpo_pairs():
    """Prefer grounded/cited answers and honest refusal over confident fabrication;
    prefer diagnose-before-mutate over blind action."""
    pairs = []
    fakes = ["cloudhopper", "datamule", "authzilla", "pipewrench", "shadowcat"]
    for f in fakes:
        pairs.append({
            "prompt": f"What does the {f} service do at {COMPANY}?",
            "chosen": f"I don't have any record of a service called {f} at {COMPANY}. "
                      f"It may be misspelled - can you check the name? Known services include "
                      f"{', '.join(list(SERVICES)[:4])}, among others.",
            "rejected": f"{f} is {COMPANY}'s internal scheduling service. It runs on port 7515, "
                        f"is owned by Core Platform, and has a 50ms SLO."})
    for svc, (team, port, slo, deps, desc) in list(SERVICES.items())[:8]:
        pairs.append({
            "prompt": f"{svc} is erroring in prod. What should I do first?",
            "chosen": f"First diagnose, don't mutate: run `{CLI} status {svc}` and "
                      f"`{CLI} logs {svc} --tail 100` to identify the error code"
                      + (f", and check its dependency {deps[0]} as well" if deps else "")
                      + ". Then apply the matching runbook. Roll back only if a recent deploy "
                        "raised the error rate above 2% (rollback policy).",
            "rejected": f"Just restart it: `{CLI} restart {svc}`. If that doesn't work, "
                        f"delete the pods and redeploy from scratch."})
        pairs.append({
            "prompt": f"Quick: what port is {svc} on? If you're not sure, say so.",
            "chosen": f"Port {port}.",
            "rejected": f"Port {port + 111}."})
    R.shuffle(pairs)
    return pairs

def grpo_tasks():
    """Single-step verifiable ops tasks (exact-answer, for RLVR)."""
    tasks = []
    for svc, (team, port, slo, deps, _) in SERVICES.items():
        tasks.append({"q": f"A request to {svc} took {slo * 3}ms. By how many milliseconds does "
                           f"that exceed its p99 SLO? End with 'Answer: <number>'.",
                      "gold": str(slo * 3 - slo)})
        tasks.append({"q": f"{COMPANY} capacity policy requires how many minimum prod replicas for "
                           f"{svc}? End with 'Answer: <number>'.",
                      "gold": "3" if team == "Payments" else "2"})
        if deps:
            tasks.append({"q": f"How many direct upstream dependencies does {svc} have? "
                               f"End with 'Answer: <number>'.", "gold": str(len(deps))})
    R.shuffle(tasks)
    return tasks

def main():
    out = {
        "corpus.json": corpus(),
        "eval_domain_qa.json": domain_qa(),
        "eval_if.json": if_eval(),
        "eval_halluc.json": halluc_eval(),
        "eval_tools.json": tool_eval(),
        "train_sft.json": sft_chat(),
        "train_reasoning.json": reasoning_traces(),
        "train_tools.json": tool_calls(),
        "train_mcp.json": mcp_traces(),
        "train_dpo.json": dpo_pairs(),
        "train_grpo.json": grpo_tasks(),
    }
    for name, data in out.items():
        json.dump(data, open(f"{HERE}/{name}", "w"), indent=1)
        print(f"  {name}: {len(data)} items")

if __name__ == "__main__":
    main()
