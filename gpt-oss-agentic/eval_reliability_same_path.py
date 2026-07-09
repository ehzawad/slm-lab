#!/usr/bin/env python3
"""BEFORE/AFTER reliability evaluator on the SAME vLLM path that produced the
non-floored base baseline (eval_same_path.py).

Reuses eval_same_path.py verbatim for:
  - server lifecycle (start/wait/stop via SIGINT, GPU 1 / A6000)
  - the OpenAI chat() payload (temp1.0 top_p1.0 top_k0 min_p0, reasoning medium)
  - the harmony tool-name/arg sanitization (_clean_tool_name/_clean_tool_args)

It adds the NOVEL reliability instrumentation the base run lacked, computed on
the RAW server tool_calls (before repair) against the sim's TOOLS_SPEC:
  executable-call rate = { valid-JSON %, schema-valid %, dispatched % }
  invalid-call rate     = 1 - schema-valid %
  repaired %            = raw call needed harmony-token sanitization to survive

The instrumentation is injected by wrapping eval_same_path._normalize (which
receives the raw server message), so the actual agent path is byte-identical to
the base baseline: same sanitization stays ON. This keeps base vs adapter
apples-to-apples on one path.
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_same_path as esp  # noqa: E402
from incident_harness import run_episode  # noqa: E402
from incident_sim import build_scenarios, TOOLS_SPEC  # noqa: E402

TOOL_SCHEMAS = {t["function"]["name"]: t["function"].get("parameters", {}) for t in TOOLS_SPEC}
TOOL_NAMES = set(TOOL_SCHEMAS.keys())


def _valid_json_obj(s):
    if not isinstance(s, str):
        return isinstance(s, dict)
    try:
        return isinstance(json.loads(s), dict)
    except (json.JSONDecodeError, ValueError):
        return False


def _schema_valid(name, args_json):
    """Cleaned name is a known tool AND cleaned args satisfy the JSON schema
    (required keys present, no unknown keys, enum + primitive types honored)."""
    if name not in TOOL_NAMES:
        return False
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else args_json
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(args, dict):
        return False
    params = TOOL_SCHEMAS[name]
    props = params.get("properties", {})
    required = params.get("required", [])
    for r in required:
        if r not in args:
            return False
    for k, v in args.items():
        if k not in props:
            return False
        spec = props[k]
        enum = spec.get("enum")
        if enum is not None and v not in enum:
            return False
        t = spec.get("type")
        if t == "string" and not isinstance(v, str):
            return False
        if t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            return False
    return True


def _make_recording_normalize(stats):
    orig = esp._normalize

    def recording_normalize(msg):
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {}) or {}
            raw_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            cname = esp._clean_tool_name(raw_name)
            cargs = esp._clean_tool_args(raw_args)
            stats["total"] += 1
            stats["valid_json"] += int(_valid_json_obj(raw_args))
            sv = _schema_valid(cname, cargs)
            stats["schema_valid"] += int(sv)
            # dispatched: what run_episode actually hands to sim.dispatch --
            # a recovered known tool name with parseable args (non-empty name
            # survives _normalize; unknown-tool names sim rejects).
            disp = (cname in TOOL_NAMES) and _valid_json_obj(cargs)
            stats["dispatched"] += int(disp)
            if not sv:
                stats["invalid"] += 1
            raw_clean_name = (raw_name or "").strip()
            if cname != raw_clean_name or (isinstance(raw_args, str) and cargs != raw_args.strip()):
                stats["repaired"] += 1
        return orig(msg)

    return recording_normalize


def evaluate(label, model_path, port, adapter_path=None, temp=1.0,
             n_scenarios=24, max_calls=15):
    stats = {"total": 0, "valid_json": 0, "schema_valid": 0,
             "dispatched": 0, "invalid": 0, "repaired": 0}
    esp._normalize = _make_recording_normalize(stats)  # inject instrumentation

    scenarios = build_scenarios(n_scenarios)
    model_name = esp.ADAPTER_NAME if adapter_path else esp.SERVED_NAME
    proc, log, log_path = esp.start_server(model_path, port, adapter_path=adapter_path)
    per = []
    try:
        esp.wait_ready(proc, port, log_path)
        chat = esp.make_chat(port, model_name, temp=temp)
        t0 = time.time()
        for i, sc in enumerate(scenarios, 1):
            r = run_episode(sc, chat, max_calls=max_calls)
            per.append(r)
            print(f"  [{i:>2}/{len(scenarios)}] {sc['id']:<28} "
                  f"solved={int(r['solved'])} steps={r['steps']:>2} "
                  f"rc={int(r['correct_root_cause'])} redun={r['redundant_calls']}",
                  flush=True)
        elapsed = time.time() - t0
    finally:
        esp.stop_server(proc, log)

    n = len(per)
    solved = sum(x["solved"] for x in per)
    rc = sum(x["correct_root_cause"] for x in per)
    total_calls = sum(x["steps"] for x in per)
    total_redundant = sum(x["redundant_calls"] for x in per)
    tot = max(stats["total"], 1)

    summary = {
        "label": label,
        "path": "vllm-openai-tool-parser",
        "model": model_path,
        "adapter": adapter_path,
        "temp": temp,
        "reasoning_effort": "medium",
        "n_scenarios": n,
        "max_calls": max_calls,
        "solved": solved,
        "solved_rate": round(solved / n, 4),
        "correct_root_cause": rc,
        "root_cause_rate": round(rc / n, 4),
        "avg_steps": round(total_calls / n, 3),
        "redundant_calls_total": total_redundant,
        "redundant_call_rate": round((total_redundant / total_calls) if total_calls else 0.0, 4),
        # ---- novel reliability metrics (raw model tool_calls vs schema) ----
        "reliability": {
            "raw_tool_calls": stats["total"],
            "valid_json": stats["valid_json"],
            "valid_json_rate": round(stats["valid_json"] / tot, 4),
            "schema_valid": stats["schema_valid"],
            "schema_valid_rate": round(stats["schema_valid"] / tot, 4),
            "dispatched": stats["dispatched"],
            "dispatched_rate": round(stats["dispatched"] / tot, 4),
            "executable_call_rate": round(stats["schema_valid"] / tot, 4),
            "invalid_calls": stats["invalid"],
            "invalid_call_rate": round(stats["invalid"] / tot, 4),
            "repaired_by_sanitizer": stats["repaired"],
            "repaired_rate": round(stats["repaired"] / tot, 4),
        },
        "elapsed_s": round(elapsed, 1),
        "gpu": f"GPU{esp.GPU_ID}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_scenario": [
            {"id": x["id"], "solved": bool(x["solved"]), "steps": x["steps"],
             "correct_root_cause": bool(x["correct_root_cause"]),
             "redundant_calls": x["redundant_calls"]}
            for x in per
        ],
    }

    rel = summary["reliability"]
    print(f"\n== {label} ==")
    print(f"  solved            : {solved}/{n}  ({summary['solved_rate']*100:.1f}%)")
    print(f"  correct root-cause: {rc}/{n}  ({summary['root_cause_rate']*100:.1f}%)")
    print(f"  avg steps         : {summary['avg_steps']}")
    print(f"  raw tool calls    : {rel['raw_tool_calls']}")
    print(f"  valid-JSON        : {rel['valid_json_rate']*100:.1f}%")
    print(f"  schema-valid (exec): {rel['schema_valid_rate']*100:.1f}%")
    print(f"  dispatched        : {rel['dispatched_rate']*100:.1f}%")
    print(f"  invalid-call rate : {rel['invalid_call_rate']*100:.1f}%")
    print(f"  repaired by saniti: {rel['repaired_rate']*100:.1f}%")
    print(f"  elapsed           : {summary['elapsed_s']}s")

    esp._append_score(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=esp.DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=18490)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-calls", type=int, default=15)
    args = ap.parse_args()
    evaluate(args.label, args.model, args.port, adapter_path=args.adapter,
             temp=args.temp, n_scenarios=args.n, max_calls=args.max_calls)


if __name__ == "__main__":
    main()
