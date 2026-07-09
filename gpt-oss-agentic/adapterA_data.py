#!/usr/bin/env python3
"""ADAPTER A - "tool-call reliability" SFT data builder.

Produces harmony SFT data for RELIABLE tool-calling, from the incident simulator
plus the existing 520 harmony trajectories. Two ingredients, both rendered by the
OFFICIAL harmony chat template at train time (identical to the vLLM inference
path), so the model only ever sees correct commentary-channel tool calls:

  (i)  POSITIVE trajectories - the correct expert tool-call turns (check_all ->
       diagnose -> fix -> verify), which the harmony template renders as
       `<|start|>assistant to=functions.NAME<|channel|>commentary json
       <|message|>{...}<|call|>`. These teach well-formed args + valid tool
       names + correct multi-call sequencing.

  (ii) HARD NEGATIVES turned into POSITIVES - for exactly the situations where a
       naive model fails, we supply the CORRECT target so the assistant-only loss
       trains the fix:
         * wrong-tool / malformed-args: SINGLE-DECISION examples that isolate the
           "given the diagnosis, emit the RIGHT mutating tool with well-formed
           JSON args" decision (set_config vs rollback vs restart; numeric AND
           string arg values).
         * calls-when-no-tool-needed: ABSTENTION examples where the fleet is
           already healthy and the correct target is a FINAL message with NO tool
           call (suppresses spurious tool calls).
         * hallucinated tool name: never emit a name outside the 8 real tools;
           every real tool appears in the positives (name distribution pinned).

Split: 10% of the UNIQUE (fault_type:service) scenario ids are held out entirely
(no trajectory from a held-out id is trained). Their initial system+user prompts
form the FREE-RUNNING held-out eval set for the valid-tool-call-rate metric.

Outputs (this dir):
  adapterA_train.json     - list[{scenario_id, kind, messages}]  (train)
  adapterA_heldout.json   - {"held_ids":[...], "eval_prompts":[{scenario_id,
                             fault_type, messages}], "heldout_examples":[...]}
"""
import json, os, sys, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from incident_sim import (  # noqa: E402
    IncidentSim, build_scenarios, expert_trajectory, SERVICES,
    CONFIG_FIX, POOL_MAX_MIN, FAULT_CODE,
)
from incident_harness import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

SEED = 3407
HELDOUT_FRAC = 0.10
SFT_PATH = os.path.join(HERE, "sft_trajectories.json")
TRAIN_OUT = os.path.join(HERE, "adapterA_train.json")
HELD_OUT = os.path.join(HERE, "adapterA_heldout.json")

VALID_TOOLS = {"get_status", "get_logs", "get_dependencies", "check_all",
               "restart", "scale", "rollback", "set_config"}


# --------------------------------------------------------------------------
# message-construction helpers (schema identical to sft_trajectories.json)
# --------------------------------------------------------------------------
def _asst_call(name, args, thinking=""):
    return {"role": "assistant", "content": "", "thinking": thinking,
            "tool_calls": [{"id": f"call_{name}", "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args)}}]}


def _tool_result(name, content):
    return {"role": "tool", "tool_call_id": f"call_{name}", "name": name,
            "content": content}


def _asst_final(text):
    return {"role": "assistant", "content": text}


def _base_msgs(sim):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(sim)}]


def _correct_fix_call(scenario):
    """The single CORRECT mutating tool call for the fault (tool-selection target)."""
    root, ft = scenario["root_service"], scenario["fault_type"]
    if ft == "bad_config":
        return _asst_call("set_config",
                          {"service": root, "key": scenario["fix_key"],
                           "value": scenario["fix_value"]},
                          "Root fault is bad_config; correct the drifted key, do NOT just restart.")
    if ft == "pool_exhausted":
        return _asst_call("set_config",
                          {"service": root, "key": "pool_max", "value": POOL_MAX_MIN},
                          "Pool exhausted; raise pool_max>=512 before restarting.")
    if ft == "bad_deploy":
        return _asst_call("rollback", {"service": root},
                          "Error-rate spiked post-deploy; roll back, a restart relaunches the bad code.")
    return _asst_call("restart", {"service": root},
                      "Process is crash-looping; a restart is the correct fix for a crash.")


# --------------------------------------------------------------------------
# (ii) HARD-NEGATIVE-derived POSITIVE examples
# --------------------------------------------------------------------------
def build_single_decision(scenario):
    """Isolate the tool-SELECTION + arg-formatting decision. Context pre-fills the
    two diagnostic reads (check_all, get_logs root); TARGET = the correct mutating
    call. Counters wrong-tool and malformed-args failures."""
    sim = IncidentSim(scenario)
    root = scenario["root_service"]
    msgs = _base_msgs(sim)
    msgs.append(_asst_call("check_all", {}, "Survey the fleet first."))
    msgs.append(_tool_result("check_all", sim.check_all()))
    msgs.append(_asst_call("get_logs", {"service": root},
                           f"{root} looks like the root; read its logs to confirm the fault type."))
    msgs.append(_tool_result("get_logs", sim.get_logs(root)))
    msgs.append(_correct_fix_call(scenario))      # <-- the trained decision
    return {"scenario_id": scenario["id"], "kind": "single_decision", "messages": msgs}


def build_abstention(scenario):
    """calls-when-no-tool-needed -> correct = STOP. Context shows a check_all with
    every service healthy; TARGET = a FINAL message, NO tool call."""
    sim = IncidentSim(scenario)
    # force a fully-healthy world snapshot for the verify turn
    healthy = json.dumps({s: "healthy" for s in SERVICES})
    msgs = _base_msgs(sim)
    msgs.append(_asst_call("check_all", {}, "Verify current fleet health before acting."))
    msgs.append(_tool_result("check_all", healthy))
    msgs.append(_asst_final(
        "All ten services report healthy and no fault codes are present, so there is "
        "nothing to remediate. No tool call is needed; the fleet is already stable."))
    return {"scenario_id": scenario["id"], "kind": "abstention", "messages": msgs}


# --------------------------------------------------------------------------
# build + split
# --------------------------------------------------------------------------
def main():
    rng = random.Random(SEED)
    positives = json.load(open(SFT_PATH))     # 520 harmony trajectories (backbone)
    for t in positives:
        t.setdefault("kind", "positive_trajectory")

    all_ids = sorted({t["scenario_id"] for t in positives})
    rng.shuffle(all_ids)
    n_held = max(1, round(len(all_ids) * HELDOUT_FRAC))
    held_ids = set(sorted(all_ids[:n_held]))
    train_ids = [i for i in all_ids if i not in held_ids]

    # scenario objects keyed by id (for hard-negatives + held-out prompts)
    scen_by_id = {sc["id"]: sc for sc in build_scenarios(len(SERVICES) * 4)}
    # ensure every trajectory id has a scenario object (fault_type:service)
    for sid in all_ids:
        if sid not in scen_by_id:
            ft, root = sid.split(":", 1)
            from incident_sim import _make_scenario
            scen_by_id[sid] = _make_scenario(root, ft)

    # ---- TRAIN set -------------------------------------------------------
    train = [t for t in positives if t["scenario_id"] in train_ids]
    # hard-negative-derived positives, only from TRAIN ids
    hard = []
    for sid in train_ids:
        sc = scen_by_id[sid]
        hard.append(build_single_decision(sc))
        hard.append(build_abstention(sc))
    train.extend(hard)
    rng.shuffle(train)

    # ---- HELD-OUT set ----------------------------------------------------
    # Free-running eval prompts, ALL from held-out ids (no leakage). Two kinds:
    #   * initial system+user  -> first action should be a valid tool call
    #   * mid-episode truncations of held-out trajectories, cut right before an
    #     assistant tool-call turn -> the next action should be a valid tool call
    # Every prompt is a state where a valid commentary-channel tool call is the
    # correct output, so valid-tool-call-rate is well defined.
    heldout_examples = [t for t in positives if t["scenario_id"] in held_ids]
    initial_prompts = []
    for sid in sorted(held_ids):
        sc = scen_by_id[sid]
        sim = IncidentSim(sc)
        initial_prompts.append({"scenario_id": sid, "fault_type": sc["fault_type"],
                                "root_service": sc["root_service"], "kind": "initial",
                                "messages": _base_msgs(sim)})
    mid_prompts, seen_ctx = [], set()
    for t in heldout_examples:
        msgs = t["messages"]
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls") and i > 0:
                ctx = msgs[:i]                       # expects a tool call next
                key = json.dumps(ctx, sort_keys=True)
                if key in seen_ctx:
                    continue
                seen_ctx.add(key)
                mid_prompts.append({"scenario_id": t["scenario_id"],
                                    "fault_type": t["fault_type"],
                                    "root_service": t.get("root_service"),
                                    "kind": "mid_episode", "messages": ctx})
    rng.shuffle(mid_prompts)
    eval_prompts = initial_prompts + mid_prompts[:48 - len(initial_prompts)]

    # ---- persist ---------------------------------------------------------
    json.dump(train, open(TRAIN_OUT, "w"))
    json.dump({"held_ids": sorted(held_ids),
               "eval_prompts": eval_prompts,
               "heldout_examples": heldout_examples},
              open(HELD_OUT, "w"))

    # ---- report ----------------------------------------------------------
    kinds = Counter(t["kind"] for t in train)
    print("=== ADAPTER A DATA ===")
    print(f"unique scenario ids      : {len(all_ids)}  "
          f"(train {len(train_ids)} / held {len(held_ids)})")
    print(f"held-out ids             : {sorted(held_ids)}")
    print(f"TRAIN examples           : {len(train)}")
    for k, v in sorted(kinds.items()):
        print(f"    {k:<22}: {v}")
    print(f"HELD-OUT eval prompts    : {len(eval_prompts)}  "
          f"(free-running valid-tool-call metric)")
    print(f"HELD-OUT trajectories    : {len(heldout_examples)} (excluded from train)")
    print(f"train -> {TRAIN_OUT}")
    print(f"held  -> {HELD_OUT}")


if __name__ == "__main__":
    main()
