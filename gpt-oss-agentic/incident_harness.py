#!/usr/bin/env python3
"""Multi-turn agent loop for the Incident-Response simulator.

Backend-agnostic: `run_episode` takes a `chat(messages, tools) -> message`
callable, so any backend plugs in unchanged:
  - a llama.cpp / OpenAI-compatible endpoint serving a GGUF, or
  - local transformers / Unsloth `generate` for a fine-tuned adapter.

`chat` must return an assistant message dict. To act, it includes OpenAI-style
`tool_calls`:
  {"role":"assistant","content":"...","tool_calls":[
      {"id": "...", "type":"function",
       "function":{"name":"restart","arguments":"{\"service\":\"quillbase\"}"}}]}
Return an assistant message with NO tool_calls to finish. The harness executes
each call against the sim, feeds results back as role="tool" messages, loops to
the call budget, then scores.
"""
import json, uuid
from incident_sim import IncidentSim, TOOLS_SPEC, expert_trajectory, build_scenarios, SERVICES

SYSTEM_PROMPT = (
    "You are an on-call SRE for NimbusWorks. Multiple services are alerting. "
    "Exactly one service has a real ROOT-CAUSE fault; other alerting services "
    "are only cascading symptoms of an unhealthy dependency. Use the tools to "
    "diagnose the root cause, apply the CORRECT fix for the fault type, and "
    "verify every service is healthy. Do NOT blindly restart everything: a "
    "restart does not fix a bad config, a bad deploy, or an exhausted pool. "
    "You have a limited tool-call budget."
)


def build_user_prompt(sim):
    return (
        "Incident: alerts firing across the fleet. Services: "
        + ", ".join(sorted(SERVICES.keys()))
        + ".\nDiagnose the root cause and restore all services to healthy. "
        "Fix types: bad_config -> set_config then restart; bad_deploy -> "
        "rollback; pool_exhausted -> set_config(pool_max>=512) then restart; "
        "dependency outage -> restart the ROOT service (not the symptom). "
        "Call tools now."
    )


def _tool_msg(tc_id, name, content):
    return {"role": "tool", "tool_call_id": tc_id, "name": name, "content": content}


def run_episode(scenario, chat, max_calls=15, verbose=False):
    """Drive one incident to completion; returns the sim score dict + id."""
    sim = IncidentSim(scenario, max_calls=max_calls)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sim)},
    ]
    while sim.calls < max_calls:
        msg = chat(messages, TOOLS_SPEC)
        messages.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            break  # agent declared itself done
        for tc in tcs:
            if sim.calls >= max_calls:
                break
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {}
            result = sim.dispatch(name, args)
            if verbose:
                print(f"  -> {name}({args}) = {result}")
            messages.append(_tool_msg(tc.get("id", "0"), name, result))
    score = sim.score()
    score["id"] = scenario["id"]
    return score


# --------------------------------------------------------------------------
# Reference agents (no model needed) for smoke-testing the environment.
# --------------------------------------------------------------------------
def _asst_tool_call(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": uuid.uuid4().hex[:8], "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def expert_agent(scenario):
    """chat() that replays the gold trajectory one tool call per turn."""
    steps = list(expert_trajectory(scenario))
    idx = {"i": 0}

    def chat(messages, tools):
        if idx["i"] >= len(steps):
            return {"role": "assistant", "content": "All services healthy. Done."}
        name, args = steps[idx["i"]]
        idx["i"] += 1
        return _asst_tool_call(name, args)

    return chat


def restart_everything_agent():
    """Brute-force baseline: restart every service in turn, then stop."""
    svcs = sorted(SERVICES.keys())
    idx = {"i": 0}

    def chat(messages, tools):
        if idx["i"] >= len(svcs):
            return {"role": "assistant", "content": "Restarted everything. Done."}
        s = svcs[idx["i"]]
        idx["i"] += 1
        return _asst_tool_call("restart", {"service": s})

    return chat


if __name__ == "__main__":
    scs = build_scenarios(24)
    by_ft = {}
    for sc in scs:
        by_ft.setdefault(sc["fault_type"], sc)  # keep first of each type

    # 1) EXPERT agent must solve a config, a deploy, and a crash scenario.
    print("== expert agent (gold trajectory as the agent) ==")
    expert_pick = [by_ft["bad_config"], by_ft["bad_deploy"], by_ft["crash"]]
    for sc in expert_pick:
        r = run_episode(sc, expert_agent(sc))
        print(f"  {sc['id']:<28} solved={r['solved']} steps={r['steps']} "
              f"root_cause={r['correct_root_cause']} redundant={r['redundant_calls']}")

    # 2) BRUTE-FORCE restart-everything must FAIL on config/deploy/pool faults.
    print("== restart-everything agent (anti-brute-force proof) ==")
    brute_pick = [by_ft["bad_config"], by_ft["bad_deploy"], by_ft["pool_exhausted"], by_ft["crash"]]
    for sc in brute_pick:
        r = run_episode(sc, restart_everything_agent())
        print(f"  {sc['id']:<28} solved={r['solved']} steps={r['steps']} "
              f"root_cause={r['correct_root_cause']} redundant={r['redundant_calls']}")

    # 3) Full expert sweep across all 24 for a saturation reference.
    solved = sum(run_episode(sc, expert_agent(sc))["solved"] for sc in scs)
    print(f"== expert sweep: {solved}/{len(scs)} solved ==")

    # Example scenario detail.
    ex = by_ft["bad_config"]
    print("\n== example scenario ==")
    print(json.dumps({k: ex[k] for k in ("id", "root_service", "fault_type",
          "fix_key", "fix_value", "cascade", "cascade_depth")}, indent=2))
    print("expert steps:", [(n, a) for n, a in expert_trajectory(ex)])
