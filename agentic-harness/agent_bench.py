#!/usr/bin/env python3
"""Multi-turn agentic harness for sub-10B GGUF models on the A5000.

Unlike the single-turn tool-call probe, this EXECUTES tools, feeds results back,
loops until the agent finishes, and scores END-TO-END task success (including
final MCP database state). Tools come from two sources:
  - a real (minimal) MCP server over JSON-RPC stdio (stateful order DB), and
  - local tools (calculator, kv memory, mock search).
Covers: sequential dependency, parallel compare, state/memory across turns,
tool-error recovery, and abstention. Metrics: task success, avg tool iterations,
invalid calls, and recovery rate. Resumable per model.
"""
import subprocess, time, json, re, sys, os, urllib.request, signal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("SLM_LAB_ROOT", os.path.dirname(HERE))  # holds models/ + llama.cpp/
BIN = f"{REPO}/llama.cpp/build/bin"
PORT = 18413
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

MODELS = [
    ("Qwen3.5-4B Q4_K_M",  f"{REPO}/models/q4b/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.5-9B Q4_K_M",  f"{REPO}/models/q9b/Qwen3.5-9B-Q4_K_M.gguf"),
    ("Qwen3.5-9B Q8_0",    f"{REPO}/models/q9b/Qwen3.5-9B-Q8_0.gguf"),
    ("gpt-oss-20b Q4_K_M", f"{REPO}/models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf"),
]

# ---------------- MCP client (spawns mcp_server.py, JSON-RPC over stdio) -------
class MCP:
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, f"{HERE}/mcp_server.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0
        self._rpc("initialize", {"protocolVersion": "2024-11-05",
            "capabilities": {}, "clientInfo": {"name": "agent-bench", "version": "0.1"}})
        self._notify("notifications/initialized")
        self.tools = self._rpc("tools/list", {})["tools"]
    def _rpc(self, method, params):
        self._id += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self._id,
            "method": method, "params": params}) + "\n"); self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("mcp server closed")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg: raise RuntimeError(msg["error"])
                return msg["result"]
    def _notify(self, method):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.p.stdin.flush()
    def call(self, name, args):
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        return r["content"][0]["text"]
    def reset(self): self._rpc("admin/reset", {})
    def state(self): return self._rpc("admin/state", {})["orders"]
    def close(self):
        try: self.p.terminate(); self.p.wait(timeout=5)
        except Exception: self.p.kill()

# ---------------- Local tools -------------------------------------------------
SEARCH_CORPUS = {
    "capital of france": "Paris",
    "population of paris": "2100000",
    "capital of japan": "Tokyo",
}
class Local:
    def __init__(self): self.kv = {}
    def calculator(self, expression):
        if not re.fullmatch(r"[0-9+\-*/(). %]+", expression or ""):
            return {"error": "invalid expression"}
        try: return {"result": eval(expression, {"__builtins__": {}}, {})}
        except Exception as e: return {"error": str(e)}
    def kv_set(self, key, value): self.kv[key] = value; return {"ok": True}
    def kv_get(self, key): return {"value": self.kv.get(key)}
    def search(self, query):
        q = (query or "").lower().strip().rstrip("?")
        for k, v in SEARCH_CORPUS.items():
            if k in q: return {"result": v}
        return {"result": None, "note": "no match"}

LOCAL_TOOLS = [
    {"name": "calculator", "description": "Evaluate an arithmetic expression.",
     "inputSchema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "kv_set", "description": "Store a value in memory under a key.",
     "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}},
    {"name": "kv_get", "description": "Retrieve a value from memory by key.",
     "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
    {"name": "search", "description": "Search a small knowledge base for a fact.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]

def to_openai_tools(mcp_tools, local_tools):
    out = []
    for t in mcp_tools + local_tools:
        out.append({"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}})
    return out

# ---------------- llama-server ------------------------------------------------
def start_server(path):
    p = subprocess.Popen([f"{BIN}/llama-server", "-m", path, "-ngl", "99",
        "--port", str(PORT), "-c", "8192", "--jinja", "--host", "127.0.0.1"],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return p
        except Exception:
            if p.poll() is not None: raise RuntimeError("server died")
            time.sleep(1)
    raise RuntimeError("server timeout")

def chat(messages, tools):
    body = {"messages": messages, "tools": tools, "tool_choice": "auto",
            "temperature": 0.0, "max_tokens": 4096}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]

# ---------------- Tasks -------------------------------------------------------
MCP_NAMES = {"get_order", "cancel_order"}

def build_tasks():
    return [
        {"id": "cancel_unshipped", "reset": True,
         "turns": ["Cancel order A1002 if it has not shipped yet, then tell me the refund amount in the form 'REFUND: <number>'."],
         "check": lambda t, w: bool(re.search(r"REFUND:\s*80", t)) and w.state()["A1002"]["status"] == "cancelled"},
        {"id": "cannot_cancel_shipped", "reset": True,
         "turns": ["Try to cancel order A1001. If it cannot be cancelled, reply exactly 'CANNOT CANCEL'."],
         "check": lambda t, w: "CANNOT CANCEL" in t.upper() and w.state()["A1001"]["status"] == "shipped"},
        {"id": "multihop_search_calc", "reset": True,
         "turns": ["Find the population of the capital of France, multiply it by 2, and give the answer as 'RESULT: <number>'."],
         "check": lambda t, w: bool(re.search(r"RESULT:\s*4200000", t.replace(",", "")))},
        {"id": "state_memory", "reset": True,
         "turns": ["Remember that my budget is 500 dollars.",
                   "I want to buy items costing 120, 90, and 200 dollars. Am I within budget? Reply exactly 'WITHIN' or 'OVER'."],
         "check": lambda t, w: "WITHIN" in t.upper() and "OVER" not in t.upper()},
        {"id": "parallel_compare", "reset": True,
         "turns": ["Which is cheaper, order A1001 or A1003? Reply as 'CHEAPER: <order_id>'."],
         "check": lambda t, w: bool(re.search(r"CHEAPER:\s*A1003", t))},
        {"id": "error_recover", "reset": True,
         "turns": ["Get the status of order A9999. If it does not exist, get order A1003 instead and report its status as 'STATUS: <status>'."],
         "check": lambda t, w: bool(re.search(r"STATUS:\s*processing", t, re.I))},
        {"id": "abstain_no_tool", "reset": True,
         "turns": ["What is the capital of Japan? Answer as 'ANSWER: <city>'."],
         "check": lambda t, w: "TOKYO" in t.upper()},
        {"id": "calc_direct", "reset": True,
         "turns": ["What is 17*23+5? Reply as 'CALC: <number>'."],
         "check": lambda t, w: bool(re.search(r"CALC:\s*396", t))},
    ]

def parse_xml_tool_calls(text):
    """Recover Qwen3.5 native tool calls that llama.cpp's --jinja parser leaves as
    raw text on multi-turn: <tool_call><function=NAME><parameter=K>V</parameter>...
    Returns list of (name, args_dict). This is a SERVING-STACK workaround, not a
    model fix — it measures true agentic capability despite the parser gap."""
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        fm = re.search(r"<function=([^>\s]+)", block)
        if not fm:
            continue
        args = {}
        for pm in re.finditer(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", block, re.S):
            args[pm.group(1).strip()] = pm.group(2).strip()
        calls.append((fm.group(1).strip(), args))
    return calls

SYSTEM = ("You are a helpful agent with access to tools. Use tools when needed, "
          "one step at a time. After you have the information, give the final answer "
          "in the exact format requested. Do not call a tool if you already can answer.")

def run_task(task, tools, local, mcp):
    if task.get("reset"): mcp.reset(); local.kv.clear()
    messages = [{"role": "system", "content": SYSTEM}]
    stats = {"tool_calls": 0, "invalid": 0, "errors": 0, "iters": 0, "xml_fallback": 0}
    final = ""
    for user_turn in task["turns"]:
        messages.append({"role": "user", "content": user_turn})
        for _ in range(6):  # tool-loop budget per turn
            stats["iters"] += 1
            msg = chat(messages, tools)
            tcs = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if not tcs:  # fallback: recover Qwen XML tool calls llama.cpp didn't parse
                xml = parse_xml_tool_calls(content) or parse_xml_tool_calls(reasoning)
                if xml:
                    stats["xml_fallback"] += 1
                    tcs = [{"id": f"xml{i}", "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)}}
                           for i, (n, a) in enumerate(xml)]
            if not tcs:
                messages.append({"role": "assistant", "content": content})
                final = content.strip() or reasoning  # thinking-model fallback
                break
            messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
            for tc in tcs:
                stats["tool_calls"] += 1
                fn = tc["function"]["name"]
                try: args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception: args = {}; stats["invalid"] += 1
                if fn in MCP_NAMES:
                    result = mcp.call(fn, args)
                elif fn == "calculator": result = json.dumps(local.calculator(**args))
                elif fn == "kv_set":    result = json.dumps(local.kv_set(**args))
                elif fn == "kv_get":    result = json.dumps(local.kv_get(**args))
                elif fn == "search":    result = json.dumps(local.search(**args))
                else: result = json.dumps({"error": f"unknown tool {fn}"}); stats["invalid"] += 1
                if '"error"' in result: stats["errors"] += 1
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result})
    ok = False
    try: ok = bool(task["check"](final, mcp))
    except Exception: ok = False
    return ok, stats, final

def main():
    rp = f"{HERE}/agentic_results.json"
    results = json.load(open(rp)) if os.path.exists(rp) else []
    done = {r["model"] for r in results}
    tasks = build_tasks()
    for name, path in MODELS:
        if name in done: print(f"[done] {name}", flush=True); continue
        if not os.path.exists(path): print(f"[skip] {name} missing", flush=True); continue
        print(f"\n=== {name} ===", flush=True)
        mcp = MCP(); local = Local()
        tools = to_openai_tools(mcp.tools, LOCAL_TOOLS)
        srv = start_server(path)
        per, agg = [], {"success": 0, "tool_calls": 0, "invalid": 0, "errors": 0, "recovered": 0}
        try:
            for task in tasks:
                ok, stats, final = run_task(task, tools, local, mcp)
                agg["success"] += ok
                for k in ("tool_calls", "invalid", "errors"): agg[k] += stats[k]
                if task["id"] in ("cannot_cancel_shipped", "error_recover") and ok:
                    agg["recovered"] += 1
                per.append({"task": task["id"], "ok": ok, **stats,
                            "final": final[-120:]})
                print(f"  {task['id']:22s} {'PASS' if ok else 'FAIL'} "
                      f"(iters={stats['iters']} calls={stats['tool_calls']} err={stats['errors']})", flush=True)
        finally:
            srv.send_signal(signal.SIGINT)
            try: srv.wait(timeout=30)
            except Exception: srv.kill()
            mcp.close()
        agg["n_tasks"] = len(tasks)
        print(f"  => {name}: {agg['success']}/{len(tasks)} tasks, "
              f"{agg['tool_calls']} calls, {agg['invalid']} invalid, {agg['errors']} tool-errors", flush=True)
        results.append({"model": name, "agg": agg, "per_task": per})
        json.dump(results, open(rp, "w"), indent=2)
        time.sleep(3)
    print("\n===== AGENTIC DONE =====", flush=True)

if __name__ == "__main__":
    main()
