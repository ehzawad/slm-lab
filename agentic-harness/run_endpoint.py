#!/usr/bin/env python3
"""Run the agentic 8-task suite against an EXISTING OpenAI-compatible endpoint
(e.g. a vLLM server) instead of spawning llama-server. Reuses the harness's MCP
server, tools, tasks, and scoring. Usage:
  python run_endpoint.py <label> <base_url> <served_model>
e.g. python run_endpoint.py "Qwythos-9B (vLLM)" http://127.0.0.1:18420/v1 qwythos"""
import sys, os, json, re, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urllib.request
import agent_bench as A

LABEL, BASE_URL, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]

def chat(messages, tools):
    body = {"model": MODEL, "messages": messages, "tools": tools,
            "tool_choice": "auto", "temperature": 0.0, "max_tokens": 4096}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]

A.chat = chat  # redirect the harness's model calls to the endpoint

def main():
    rp = f"{A.HERE}/agentic_results.json"
    results = json.load(open(rp)) if os.path.exists(rp) else []
    results = [x for x in results if x["model"] != LABEL]
    mcp = A.MCP(); local = A.Local()
    tools = A.to_openai_tools(mcp.tools, A.LOCAL_TOOLS)
    tasks = A.build_tasks()
    print(f"\n=== {LABEL} (endpoint {BASE_URL}) ===", flush=True)
    agg = {"success": 0, "tool_calls": 0, "invalid": 0, "errors": 0, "recovered": 0}
    per = []
    for task in tasks:
        ok, stats, final = A.run_task(task, tools, local, mcp)
        agg["success"] += ok
        for k in ("tool_calls", "invalid", "errors"): agg[k] += stats[k]
        if task["id"] in ("cannot_cancel_shipped", "error_recover") and ok: agg["recovered"] += 1
        per.append({"task": task["id"], "ok": ok, **stats, "final": final[-120:]})
        print(f"  {task['id']:22s} {'PASS' if ok else 'FAIL'} "
              f"(iters={stats['iters']} calls={stats['tool_calls']} "
              f"err={stats['errors']} xmlfb={stats['xml_fallback']})", flush=True)
    mcp.close()
    agg["n_tasks"] = len(tasks)
    xfb = sum(t["xml_fallback"] for t in per)
    print(f"  => {LABEL}: {agg['success']}/{len(tasks)} tasks, {agg['tool_calls']} calls, "
          f"{agg['invalid']} invalid, {agg['errors']} tool-errors, {xfb} xml-fallbacks", flush=True)
    results.append({"model": LABEL, "agg": agg, "per_task": per})
    json.dump(results, open(rp, "w"), indent=2)

if __name__ == "__main__":
    main()
