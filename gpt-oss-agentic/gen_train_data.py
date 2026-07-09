#!/usr/bin/env python3
"""Generate bigger training data for the gpt-oss agentic incident-response track.

Emits three JSON files under gpt-oss-agentic/:

  1. cpt_corpus.json          - ops-domain continued-pretraining corpus:
       service handbooks, per-error-code runbooks, dependency facts, policies,
       CLI references, fault-playbook docs. A few hundred plain-text docs.

  2. sft_trajectories.json    - expert incident-resolution TRAJECTORIES rendered
       as harmony/chat `messages` arrays (system/user/assistant-with-tool_calls/
       tool). 500+ scenario instances: user symptom -> assistant reasons ->
       tool calls -> tool results -> verified fix. Each trajectory is REPLAYED
       through IncidentSim so every tool result is the real environment output
       and the final state is verified solved.

  3. grpo_scenarios.json      - ~300 scenario specs for agentic GRPO. Each carries
       the full scenario dict (root_service, fault_type, fix_key/value, cascade)
       plus the system/user prompt so the RL loop can run episodes and reward
       execution success (solved).

Everything derives from nimbus-agent/world.py + incident_sim.py, so the domain
is guaranteed absent from any pretraining corpus and all gold answers are exact.
"""
import os, sys, json, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "nimbus-agent"))

import world
from world import (SERVICES, ERRORS, POLICIES, CLI_COMMANDS, TEAMS, COMPANY, CLI,
                   dependents_of)
import incident_sim as isim
from incident_sim import (IncidentSim, TOOLS_SPEC, expert_trajectory,
                          FAULT_TYPES, CONFIG_FIX, POOL_SERVICES, POOL_MAX_MIN,
                          FAULT_CODE, FAULT_HINT, CASCADE_CODE, _make_scenario,
                          _transitive_dependents, _cascade_depth)

OUT_CPT   = os.path.join(HERE, "cpt_corpus.json")
OUT_SFT   = os.path.join(HERE, "sft_trajectories.json")
OUT_GRPO  = os.path.join(HERE, "grpo_scenarios.json")

SEED = 1337

SYSTEM_PROMPT = (
    "You are an on-call SRE for NimbusWorks. Multiple services are alerting. "
    "Exactly one service has a real ROOT-CAUSE fault; other alerting services "
    "are only cascading symptoms of an unhealthy dependency. Use the tools to "
    "diagnose the root cause, apply the CORRECT fix for the fault type, and "
    "verify every service is healthy. Do NOT blindly restart everything: a "
    "restart does not fix a bad config, a bad deploy, or an exhausted pool. "
    "You have a limited tool-call budget."
)


# ==========================================================================
# 1. CPT CORPUS  -- ops-domain documents
# ==========================================================================
def gen_cpt_corpus():
    docs = []

    def add(doc_id, title, text):
        docs.append({"id": doc_id, "title": title,
                     "text": text.strip() + "\n"})

    # ---- company overview ----
    team_lines = "\n".join(
        f"- {name}: lead {m['lead']}, on-call {m['oncall_slack']}"
        for name, m in TEAMS.items())
    add("overview", f"{COMPANY} platform overview", f"""
{COMPANY} runs a microservice fleet operated via the `{CLI}` CLI. There are
{len(SERVICES)} production services across {len(TEAMS)} teams. Each service has a
health status (healthy/degraded/down), an error rate, a replica count, and a
per-service connection pool. Teams and on-call ownership:
{team_lines}
Exactly one service is ever the ROOT CAUSE of an incident; every other alerting
service is a cascading symptom of an unhealthy dependency.
""")

    # ---- per-service handbooks ----
    for svc, (team, port, slo, deps, desc) in SERVICES.items():
        dependents = dependents_of(svc)
        dep_txt = ", ".join(deps) if deps else "none (leaf dependency)"
        dpt_txt = ", ".join(dependents) if dependents else "none (leaf consumer)"
        cfg = CONFIG_FIX.get(svc)
        cfg_txt = (f"Canonical config key `{cfg[0]}` with healthy value {cfg[1]}."
                   if cfg else "No standard tunable config key.")
        pool_txt = ("This service maintains a database connection pool; the "
                    f"healthy pool_max is >= {POOL_MAX_MIN}."
                    if svc in POOL_SERVICES else
                    "This service does not front a sized connection pool.")
        add(f"handbook:{svc}", f"Service handbook: {svc}", f"""
Service `{svc}` is owned by the {team} team. {desc}. It listens on port {port}
and has an SLO of {slo} ms p99. It depends on: {dep_txt}. Services that depend on
{svc} (its dependents): {dpt_txt}. {cfg_txt} {pool_txt}
When {svc} is the root cause, its dependents show upstream connect timeouts
({CASCADE_CODE}) even though they are individually healthy; restarting a
dependent does nothing until {svc} itself is fixed.
""")

    # ---- per-error-code runbooks (world.ERRORS) ----
    for code, (svc, meaning, fix) in ERRORS.items():
        add(f"runbook:{code}", f"Runbook {code}", f"""
Error {code} on `{svc}`: {meaning}. Remediation: {fix}. Confirm recovery with
`{CLI} status {svc}` showing status healthy and error rate below 1%.
""")

    # ---- fault-type playbooks (the four incident classes) ----
    playbooks = {
        "bad_config": (
            f"Error code {FAULT_CODE['bad_config']}. A configuration value drifted "
            "out of its valid range. A bare restart re-loads the SAME bad value and "
            "does NOT fix it. Correct sequence: `set_config` the canonical key to its "
            "healthy value, THEN `restart` so the new value takes effect."),
        "bad_deploy": (
            f"Error code {FAULT_CODE['bad_deploy']}. Error rate spiked immediately "
            "after the most recent deploy. A restart just re-launches the same bad "
            "release. Correct fix: `rollback` the service to the previous release."),
        "pool_exhausted": (
            f"Error code {FAULT_CODE['pool_exhausted']}. The connection pool is "
            "saturated. A restart alone keeps the tiny pool. Correct sequence: "
            f"`set_config` pool_max to >= {POOL_MAX_MIN}, THEN `restart`."),
        "crash": (
            f"Error code {FAULT_CODE['crash']}. The process stopped answering health "
            "checks. This is the ONLY class a plain `restart` of the ROOT service "
            "fixes. Restarting a cascaded symptom instead of the root does nothing."),
    }
    for ft, body in playbooks.items():
        add(f"playbook:{ft}", f"Fault playbook: {ft}", f"""
Incident class `{ft}`. {body}
Always identify the single root-cause service first (the one whose logs show a
real fault code, not the cascade code {CASCADE_CODE}), fix it, then verify with
check_all that every service returned to healthy.
""")

    # ---- dependency facts (one per directed edge) ----
    for svc, meta in SERVICES.items():
        for dep in meta[3]:
            add(f"depfact:{svc}->{dep}", f"Dependency fact: {svc} -> {dep}", f"""
`{svc}` depends on `{dep}`. If `{dep}` is unhealthy, `{svc}` will report
{CASCADE_CODE} upstream connect timeouts and appear down even though `{svc}`
itself has no root-cause fault. Fixing `{dep}` restores `{svc}` automatically.
""")

    # ---- policies ----
    for pid, text in POLICIES:
        add(f"policy:{pid}", f"Policy: {pid}", text)

    # ---- CLI reference ----
    for cmd, meaning in CLI_COMMANDS.items():
        add(f"cli:{cmd.split()[1]}:{abs(hash(cmd))%9999}",
            f"CLI: {cmd}", f"`{cmd}` -- {meaning}.")

    # ---- diagnostic-methodology docs (general reasoning corpus) ----
    add("method:triage", "Incident triage methodology", f"""
Triage at {COMPANY}: (1) run check_all to see which services are unhealthy.
(2) Distinguish root from symptom: pull get_logs on the deepest-looking broken
service; a {CASCADE_CODE} means it is only a symptom -- follow its dependency
upward. (3) The root's logs carry a real fault code
({', '.join(FAULT_CODE[f] for f in FAULT_TYPES)}). (4) Apply the fault-type fix.
(5) Re-run check_all to verify full recovery within the tool-call budget.
""")
    add("method:antibrute", "Why restart-everything fails", f"""
Blindly restarting every service is an anti-pattern at {COMPANY}. Only a `crash`
({FAULT_CODE['crash']}) is cleared by a restart. bad_config
({FAULT_CODE['bad_config']}), bad_deploy ({FAULT_CODE['bad_deploy']}) and
pool_exhausted ({FAULT_CODE['pool_exhausted']}) all survive a restart and need a
set_config+restart, a rollback, or a pool bump + restart respectively.
""")

    # ---- per-(service, fault_type) remediation docs + incident postmortems ----
    # These are the concrete, scenario-grounded documents (the bulk of the
    # corpus): for every valid root x fault combination, a remediation card and
    # a short postmortem describing the cascade and the exact fix steps.
    for ft in FAULT_TYPES:
        roots = POOL_SERVICES if ft == "pool_exhausted" else list(SERVICES.keys())
        for root in roots:
            sc = _make_scenario(root, ft)
            cascade = sc["cascade"]
            casc_txt = ", ".join(cascade) if cascade else "no downstream services"
            team = SERVICES[root][0]
            # remediation card
            if ft == "bad_config":
                fix_steps = (f"1) set_config {root} {sc['fix_key']}={sc['fix_value']}; "
                             f"2) restart {root}.")
            elif ft == "pool_exhausted":
                fix_steps = (f"1) set_config {root} pool_max={POOL_MAX_MIN}; "
                             f"2) restart {root}.")
            elif ft == "bad_deploy":
                fix_steps = f"1) rollback {root}."
            else:
                fix_steps = f"1) restart {root}."
            add(f"remediation:{ft}:{root}",
                f"Remediation card: {ft} on {root}", f"""
When `{root}` ({team}) suffers a {ft} fault it emits {sc['error_code']}. Its
transitive dependents ({casc_txt}) go down with {CASCADE_CODE} as pure symptoms.
Correct remediation (do NOT restart the symptoms): {fix_steps}
Verify with check_all that all {len(SERVICES)} services report healthy.
""")
            # postmortem narrative
            add(f"postmortem:{ft}:{root}",
                f"Postmortem: {ft} at {root}", f"""
Incident summary. Root cause: a {ft} fault on `{root}`, owned by {team}. The
first symptom the on-call saw was multiple alerting services; check_all showed
{root} plus its dependents ({casc_txt}) unhealthy. get_logs on the dependents
returned {CASCADE_CODE} (upstream connect timeout) -- a symptom, not the cause.
get_logs on `{root}` returned {sc['error_code']}: {FAULT_HINT[ft].format(key=sc['fix_key'])}.
Resolution followed the {ft} playbook: {fix_steps} After the fix, check_all
confirmed all services healthy. Lesson: never restart a {CASCADE_CODE} symptom;
walk the dependency chain to the one service with a real fault code.
""")

    return docs


# ==========================================================================
# 2. SFT TRAJECTORIES  -- harmony/chat messages, replayed through the sim
# ==========================================================================
def build_user_prompt():
    return (
        "Incident: alerts firing across the fleet. Services: "
        + ", ".join(sorted(SERVICES.keys()))
        + ".\nDiagnose the root cause and restore all services to healthy. "
        "Fix types: bad_config -> set_config then restart; bad_deploy -> "
        "rollback; pool_exhausted -> set_config(pool_max>=512) then restart; "
        "dependency outage -> restart the ROOT service (not the symptom). "
        "Call tools now."
    )


TOOL_ID = {"n": 0}
def _next_id():
    TOOL_ID["n"] += 1
    return f"call_{TOOL_ID['n']:06d}"


def _reason_for_step(scenario, name, args, step_idx, total):
    """A short natural-language rationale that precedes each tool call, so the
    assistant turn carries harmony-style reasoning + an action."""
    root = scenario["root_service"]
    ft = scenario["fault_type"]
    if name == "check_all":
        if step_idx == 0:
            return "Start by surveying the whole fleet to see which services are alerting."
        return "Fixed the root cause; re-checking every service to confirm full recovery."
    if name == "get_logs":
        svc = args["service"]
        if svc == root:
            return (f"`{root}` looks like the root. Reading its logs to confirm the "
                    f"exact fault code and required remediation.")
        return (f"`{svc}` is alerting; reading its logs. If it shows {CASCADE_CODE} "
                f"it is only a cascading symptom of an unhealthy dependency.")
    if name == "get_dependencies":
        return (f"`{args['service']}` reports an upstream timeout; tracing its "
                f"dependencies to walk toward the real root cause.")
    if name == "get_status":
        return f"Checking `{args['service']}` status to confirm it is the faulting service."
    if name == "set_config":
        if ft == "pool_exhausted":
            return (f"Pool on `{root}` is exhausted. Raising pool_max to "
                    f"{POOL_MAX_MIN} before restarting so the larger pool takes effect.")
        return (f"Correcting the drifted config `{args['key']}` to its healthy value "
                f"{args['value']} on `{root}`; a restart is required to apply it.")
    if name == "restart":
        if ft == "crash":
            return f"`{root}` crashed and is not answering health checks; restarting it."
        return f"Config fix is staged on `{root}`; restarting so the corrected value loads."
    if name == "rollback":
        return (f"Error rate on `{root}` spiked right after its last deploy; rolling "
                f"back to the previous good release.")
    return "Proceeding with the next diagnostic step."


def render_trajectory(scenario, max_calls=15):
    """Replay the expert trajectory through a fresh sim, building a messages
    array with real tool outputs. Returns (messages, solved, steps)."""
    sim = IncidentSim(scenario, max_calls=max_calls)
    steps = expert_trajectory(scenario)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt()},
    ]
    total = len(steps)
    for i, (name, args) in enumerate(steps):
        tc_id = _next_id()
        thinking = _reason_for_step(scenario, name, args, i, total)
        messages.append({
            "role": "assistant",
            "content": "",
            "thinking": thinking,
            "tool_calls": [{
                "id": tc_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        })
        result = sim.dispatch(name, args)
        messages.append({
            "role": "tool", "tool_call_id": tc_id, "name": name,
            "content": result,
        })
    score = sim.score()
    # final assistant summary turn (no tool call = done)
    root = scenario["root_service"]
    ft = scenario["fault_type"]
    summary = (
        f"Root cause was a {ft} fault on `{root}` ({scenario['error_code']}). "
        f"Applied the correct remediation and verified every service is healthy "
        f"again in {score['steps']} tool calls.")
    messages.append({"role": "assistant", "content": summary, "thinking":
                     "check_all shows all services healthy and the root fault is "
                     "cleared; the incident is resolved."})
    return messages, score["solved"], score["steps"]


def paraphrase_user(base, rng, scenario):
    """Light surface variation on the opening user turn so 500+ instances are
    not byte-identical, without changing the task semantics."""
    openers = [
        "Incident: alerts firing across the fleet.",
        "PagerDuty just lit up -- multiple NimbusWorks services are alerting.",
        "Sev2 in progress: several services are unhealthy right now.",
        "On-call handoff: the fleet is throwing alerts across multiple services.",
        "Dashboard is red -- a bunch of services just went unhealthy.",
    ]
    closers = [
        "Call tools now.",
        "Start diagnosing with the tools.",
        "Work the incident using the tools now.",
        "Find the root cause and fix it.",
    ]
    op = rng.choice(openers)
    cl = rng.choice(closers)
    body = (" Services: " + ", ".join(sorted(SERVICES.keys()))
            + ".\nDiagnose the root cause and restore all services to healthy. "
            "Fix types: bad_config -> set_config then restart; bad_deploy -> "
            "rollback; pool_exhausted -> set_config(pool_max>=512) then restart; "
            "dependency outage -> restart the ROOT service (not the symptom). ")
    return op + body + cl


def all_scenarios():
    """Every valid (root, fault_type) combination -> full scenario dicts."""
    combos = []
    for ft in FAULT_TYPES:
        pool = POOL_SERVICES if ft == "pool_exhausted" else list(SERVICES.keys())
        for root in pool:
            combos.append(_make_scenario(root, ft))
    return combos


def gen_sft_trajectories(target=520):
    rng = random.Random(SEED)
    base_scenarios = all_scenarios()
    out = []
    verified = 0
    # Cycle through the full scenario space, adding surface variation, until we
    # reach the target count. Each instance is replayed + verified.
    i = 0
    while len(out) < target:
        sc = base_scenarios[i % len(base_scenarios)]
        i += 1
        messages, solved, steps = render_trajectory(sc)
        if not solved:
            continue  # never train on a non-solving trajectory
        verified += 1
        # apply user paraphrase variation for instances beyond the first pass
        if len(out) >= len(base_scenarios):
            messages[1]["content"] = paraphrase_user(messages[1]["content"], rng, sc)
        # reasoning effort varies to exercise the harmony template
        effort = rng.choice(["low", "medium", "high"])
        out.append({
            "scenario_id": sc["id"],
            "root_service": sc["root_service"],
            "fault_type": sc["fault_type"],
            "cascade_depth": sc["cascade_depth"],
            "reasoning_effort": effort,
            "messages": messages,
        })
    return out, verified


# ==========================================================================
# 3. GRPO SCENARIOS  -- specs for agentic RL with execution reward
# ==========================================================================
def gen_grpo_scenarios(target=300):
    rng = random.Random(SEED + 1)
    base = all_scenarios()
    out = []
    i = 0
    while len(out) < target:
        sc = base[i % len(base)]
        i += 1
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paraphrase_user("", rng, sc)
             if len(out) >= len(base) else build_user_prompt()},
        ]
        out.append({
            "scenario_id": sc["id"],
            "scenario": sc,               # full spec: sim can be reconstructed
            "prompt": prompt,
            "max_calls": 15,
            "reward": "execution_success:solved",
        })
    return out


# ==========================================================================
def main():
    print("Generating CPT corpus ...")
    cpt = gen_cpt_corpus()

    print("Generating SFT trajectories (replayed + verified) ...")
    sft, verified = gen_sft_trajectories(520)

    print("Generating GRPO scenarios ...")
    grpo = gen_grpo_scenarios(300)

    with open(OUT_CPT, "w") as f:
        json.dump(cpt, f, indent=1)
    with open(OUT_SFT, "w") as f:
        json.dump(sft, f, indent=1)
    with open(OUT_GRPO, "w") as f:
        json.dump(grpo, f, indent=1)

    # ---- report ----
    cpt_chars = sum(len(d["text"]) for d in cpt)
    sft_ft = Counter(x["fault_type"] for x in sft)
    sft_turns = sum(len(x["messages"]) for x in sft)
    grpo_ft = Counter(x["scenario"]["fault_type"] for x in grpo)

    def kb(path):
        return f"{os.path.getsize(path)/1024:.1f} KB"

    print("\n==== DATASET SUMMARY ====")
    print(f"cpt_corpus.json      : {len(cpt)} docs, {cpt_chars} chars, {kb(OUT_CPT)}")
    print(f"sft_trajectories.json: {len(sft)} trajectories "
          f"(all {verified} replayed & verified solved), "
          f"{sft_turns} total messages, {kb(OUT_SFT)}")
    print(f"  fault-type mix     : {dict(sft_ft)}")
    print(f"grpo_scenarios.json  : {len(grpo)} scenarios, {kb(OUT_GRPO)}")
    print(f"  fault-type mix     : {dict(grpo_ft)}")
    print(f"\nPaths:\n  {OUT_CPT}\n  {OUT_SFT}\n  {OUT_GRPO}")


if __name__ == "__main__":
    main()
