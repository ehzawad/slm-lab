#!/usr/bin/env python3
"""Head-to-head GGUF benchmark: Qwen3.5-4B vs 9B across quant levels.
Metrics: speed (llama-bench) + VRAM + reasoning accuracy + tool-call validity."""
import subprocess, time, json, re, sys, os, urllib.request, signal

# Repo root holds models/ and llama.cpp/ ; override with SLM_LAB_ROOT if they live elsewhere.
WS = os.environ.get("SLM_LAB_ROOT", os.path.dirname(os.path.abspath(__file__)))
BIN = f"{WS}/llama.cpp/build/bin"
sys.path.insert(0, WS)
from probes import REASONING, TOOLS, TOOLCALLS

PORT = 18412  # high port: 8081 is taken by another service on this shared box
# IMPORTANT: llama.cpp's default CUDA order is NOT nvidia-smi order (it lists the
# A6000 as device 0). Force PCI_BUS_ID order so device 0 == A5000 == `nvidia-smi -i 0`.
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}  # A5000 only

MODELS = [
    # --- Quant study: within Qwen3.5 (Q8 vs Q4 vs imatrix) ---
    ("Qwen3.5-4B Q8_0",     "models/q4b/Qwen3.5-4B-Q8_0.gguf"),
    ("Qwen3.5-4B Q4_K_M",   "models/q4b/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.5-4B UD-Q4KXL", "models/q4b/Qwen3.5-4B-UD-Q4_K_XL.gguf"),
    ("Qwen3.5-9B Q8_0",     "models/q9b/Qwen3.5-9B-Q8_0.gguf"),
    ("Qwen3.5-9B Q4_K_M",   "models/q9b/Qwen3.5-9B-Q4_K_M.gguf"),
    ("Qwen3.5-9B UD-Q4KXL", "models/q9b/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
    # --- Cross-family (Q8_0 + Q4_K_M, common on-device points) ---
    ("Gemma-3n-E2B Q8_0",   "models/gemma_e2b/gemma-3n-E2B-it-Q8_0.gguf"),
    ("Gemma-3n-E2B Q4_K_M", "models/gemma_e2b/gemma-3n-E2B-it-Q4_K_M.gguf"),
    ("Gemma-3n-E4B Q8_0",   "models/gemma_e4b/gemma-3n-E4B-it-Q8_0.gguf"),
    ("Gemma-3n-E4B Q4_K_M", "models/gemma_e4b/gemma-3n-E4B-it-Q4_K_M.gguf"),
    ("DeepSeek-R1-7B Q8_0", "models/deepseek7b/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"),
    ("DeepSeek-R1-7B Q4_K_M","models/deepseek7b/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
    ("gpt-oss-20b Q4_K_M",  "models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf"),  # Q8 ~21GB, too tight
]

def gpu_mem_used():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"])
    return int(out.decode().strip())

def run_speed(path):
    """llama-bench: prompt-processing (pp) and token-gen (tg) throughput."""
    try:
        out = subprocess.check_output(
            [f"{BIN}/llama-bench", "-m", path, "-p", "512", "-n", "128", "-ngl", "99", "-r", "3", "-o", "json"],
            env=ENV, stderr=subprocess.DEVNULL, timeout=600).decode()
        data = json.loads(out)
        pp = next((r["avg_ts"] for r in data if r.get("n_prompt", 0) > 0), None)
        tg = next((r["avg_ts"] for r in data if r.get("n_gen", 0) > 0), None)
        return round(pp, 1) if pp else None, round(tg, 1) if tg else None
    except Exception as e:
        print(f"    speed err: {e}", flush=True); return None, None

def start_server(path):
    p = subprocess.Popen(
        [f"{BIN}/llama-server", "-m", path, "-ngl", "99", "--port", str(PORT),
         "-c", "8192", "--jinja", "--host", "127.0.0.1"],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server died on startup")
            time.sleep(1)
    raise RuntimeError("server health timeout")

def chat(messages, tools=None, max_tokens=2048):
    body = {"messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools; body["tool_choice"] = "auto"
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def parse_int(text):
    m = re.findall(r"-?\d[\d,]*", text.replace(",", ""))
    return int(m[-1]) if m else None

def eval_reasoning():
    ok = 0
    for q, ans in REASONING:
        try:
            # 4096 tokens: DeepSeek-R1 / Qwen3.5-thinking / gpt-oss emit long CoT before the answer
            r = chat([{"role": "user", "content": q +
                       "\nThink briefly, then end with a line 'Answer: <integer>'."}],
                     max_tokens=4096)
            msg = r["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            # Prefer the final answer in content; fall back to reasoning (thinking models
            # that hit the token cap leave the answer only in reasoning_content).
            got = None
            for src in (content, reasoning):
                m = re.search(r"[Aa]nswer:\s*(-?\d[\d,]*)", src)
                if m:
                    got = int(m.group(1).replace(",", "")); break
            if got is None:
                got = parse_int(content) if content.strip() else parse_int(reasoning)
            ok += (got == ans)
        except Exception as e:
            print(f"    reason err: {e}", flush=True)
    return ok, len(REASONING)

def eval_tools():
    valid = correct = 0
    for prompt, exp_tool, arg_subs in TOOLCALLS:
        try:
            r = chat([{"role": "user", "content": prompt}], tools=TOOLS, max_tokens=512)
            msg = r["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs:
                continue
            valid += 1  # server returned a well-formed tool_call
            tc = tcs[0]["function"]
            name = tc["name"]
            args = (tc.get("arguments") or "").lower()
            if name == exp_tool and all(s.lower() in args for s in arg_subs):
                correct += 1
        except Exception as e:
            print(f"    tool err: {e}", flush=True)
    return valid, correct, len(TOOLCALLS)

def main():
    # Resume: load any results already computed so foreground chunks can continue.
    results = []
    rp = f"{WS}/results.json"
    if os.path.exists(rp):
        try: results = json.load(open(rp))
        except Exception: results = []
    done = {r["model"] for r in results}
    for name, rel in MODELS:
        if name in done:
            print(f"[done] {name}", flush=True); continue
        path = os.path.join(WS, rel)
        if not os.path.exists(path):
            print(f"[skip] {name}: file missing", flush=True); continue
        size_gb = round(os.path.getsize(path) / 1e9, 2)
        print(f"\n=== {name}  ({size_gb} GB) ===", flush=True)
        pp, tg = run_speed(path)
        print(f"  speed: pp={pp} t/s  tg={tg} t/s", flush=True)
        base = gpu_mem_used()
        srv = start_server(path)
        time.sleep(2)
        vram = gpu_mem_used()
        try:
            r_ok, r_n = eval_reasoning()
            t_valid, t_correct, t_n = eval_tools()
        finally:
            srv.send_signal(signal.SIGINT);
            try: srv.wait(timeout=30)
            except Exception: srv.kill()
        print(f"  vram_loaded: {vram} MiB (base {base})", flush=True)
        print(f"  reasoning: {r_ok}/{r_n}   tools valid: {t_valid}/{t_n} correct: {t_correct}/{t_n}", flush=True)
        results.append({"model": name, "size_gb": size_gb, "pp_ts": pp, "tg_ts": tg,
                        "vram_mib": vram, "reason_ok": r_ok, "reason_n": r_n,
                        "tool_valid": t_valid, "tool_correct": t_correct, "tool_n": t_n})
        with open(f"{WS}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(3)  # let VRAM free
    print("\n===== DONE =====", flush=True)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
