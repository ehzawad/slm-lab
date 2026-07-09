#!/usr/bin/env python3
"""SAME-PATH incident evaluator for gpt-oss-20b via vLLM's OpenAI server.

This is the GATE evaluator and the reusable serving path the LoRA adapter will
later be evaluated on. It serves the *unmerged* MXFP4 gpt-oss-20b (optionally
with a LoRA adapter) through vLLM's OpenAI-compatible endpoint with the native
`openai` tool-call parser + auto tool choice, so the model emits real
OpenAI-style tool_calls on the harmony `commentary` channel. That structured
tool-calling is what prevents the flooring the prior unconstrained transformers
run hit.

Usage:
  # base gpt-oss-20b (the GATE baseline)
  python eval_same_path.py --label "gpt-oss-20b base (vllm same-path)"

  # same path, with a LoRA adapter (later)
  python eval_same_path.py --adapter /path/to/adapter \
      --label "gpt-oss-20b + adapter (vllm same-path)"

Sampling per research recipe: temp 1.0, top_p 1.0, top_k 0, min_p 0,
reasoning_effort medium. Server torn down via SIGINT (never pkill).
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from incident_harness import run_episode  # noqa: E402
from incident_sim import build_scenarios  # noqa: E402

SCORES_PATH = os.path.join(HERE, "incident_scores.json")
VENV_VLLM_PY = os.path.join(HERE, "..", ".venv-vllm", "bin", "python")
DEFAULT_MODEL = os.path.join(HERE, "..", "models", "gptoss20b-hf")
GPU_ID = 1  # A6000
SERVED_NAME = "gpt-oss-20b"
ADAPTER_NAME = "incident"


# --------------------------------------------------------------------------
# vLLM OpenAI server lifecycle
# --------------------------------------------------------------------------
def start_server(model_path, port, adapter_path=None, gpu_mem=0.85,
                 max_model_len=16384, max_lora_rank=16):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    log_path = os.path.join(HERE, f"vllm_server_{port}.log")
    log = open(log_path, "w")
    cmd = [
        VENV_VLLM_PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1", "--port", str(port),
        "--tool-call-parser", "openai",
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(max_model_len),
    ]
    if adapter_path:
        cmd += [
            "--enable-lora",
            "--max-lora-rank", str(max_lora_rank),
            "--lora-modules", f"{ADAPTER_NAME}={adapter_path}",
        ]
    print(f"[server] launching: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={GPU_ID}) -> {log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, log_path


def wait_ready(proc, port, log_path, timeout=1200):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early (code {proc.returncode}):\n{_tail(log_path, 60)}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    print(f"[server] ready after {time.time()-t0:.1f}s")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"server not ready within {timeout}s:\n{_tail(log_path, 60)}")


def stop_server(proc, log):
    if proc.poll() is None:
        print("[server] sending SIGINT")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            print("[server] SIGINT timed out; SIGTERM")
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("[server] SIGTERM timed out; SIGKILL")
                proc.kill()
                proc.wait()
    try:
        log.close()
    except Exception:
        pass


def _tail(path, n):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return "(no log)"


# --------------------------------------------------------------------------
# OpenAI-compatible chat() backend
# --------------------------------------------------------------------------
def make_chat(port, model_name, temp=1.0, max_tokens=2048, retries=3,
              reasoning_effort="medium"):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def chat(messages, tools):
        payload = {
            "model": model_name,
            "messages": _sanitize(messages),
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temp,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            **({"seed": SEED} if SEED is not None else {}),
            "max_tokens": max_tokens,
            "stream": False,
            "reasoning_effort": reasoning_effort,
            "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
        }
        data = json.dumps(payload).encode()
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    body = json.loads(r.read().decode())
                return _normalize(body["choices"][0]["message"])
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode()[:600]}"
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)
            time.sleep(1.5 * (attempt + 1))
        return {"role": "assistant", "content": f"[backend error: {last_err}]"}

    return chat


def _sanitize(messages):
    out = []
    for m in messages:
        mm = dict(m)
        if mm.get("role") == "assistant" and mm.get("content") is None:
            mm["content"] = ""
        out.append(mm)
    return out


def _clean_tool_name(name):
    """Strip harmony control tokens that vLLM's parser sometimes leaks into the
    tool name (e.g. 'check_all<|channel|>commentary', 'functions.restart')."""
    if not isinstance(name, str):
        return ""
    name = name.split("<|")[0]              # drop trailing channel/control tokens
    if "to=functions." in name:
        name = name.split("to=functions.")[-1]
    name = name.replace("functions.", "")
    return name.strip().strip(" .")


def _clean_tool_args(raw):
    """Recover JSON args from parser corruption. Handles trailing harmony
    tokens and the '{"": "{...}"}' single-empty-key wrapping we observe when the
    parser mis-splits the commentary channel."""
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    raw = raw.split("<|")[0].strip()
    if not raw:
        return "{}"
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return "{}"
    # unwrap {"": "<json>"} / {"": {...}} corruption
    if isinstance(obj, dict) and set(obj.keys()) == {""}:
        inner = obj[""]
        if isinstance(inner, str):
            try:
                obj = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                obj = {}
        elif isinstance(inner, dict):
            obj = inner
        else:
            obj = {}
    return json.dumps(obj if isinstance(obj, dict) else {})


def _normalize(msg):
    out = {"role": "assistant", "content": msg.get("content") or ""}
    tcs = msg.get("tool_calls") or []
    norm = []
    for tc in tcs:
        fn = tc.get("function", {}) or {}
        name = _clean_tool_name(fn.get("name", ""))
        if not name:
            continue  # drop unrecoverable calls rather than poison history
        args = _clean_tool_args(fn.get("arguments", "{}"))
        norm.append({
            "id": tc.get("id") or "call_%d" % len(norm),
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    if norm:
        out["tool_calls"] = norm
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(label, model_path, port, adapter_path=None, temp=1.0,
             n_scenarios=24, max_calls=15, verbose=False):
    scenarios = build_scenarios(n_scenarios)
    model_name = ADAPTER_NAME if adapter_path else SERVED_NAME
    proc, log, log_path = start_server(model_path, port, adapter_path=adapter_path)
    per = []
    try:
        wait_ready(proc, port, log_path)
        chat = make_chat(port, model_name, temp=temp)
        t0 = time.time()
        for i, sc in enumerate(scenarios, 1):
            r = run_episode(sc, chat, max_calls=max_calls, verbose=verbose)
            per.append(r)
            print(f"  [{i:>2}/{len(scenarios)}] {sc['id']:<28} "
                  f"solved={int(r['solved'])} steps={r['steps']:>2} "
                  f"root_cause={int(r['correct_root_cause'])} "
                  f"redundant={r['redundant_calls']}", flush=True)
        elapsed = time.time() - t0
    finally:
        stop_server(proc, log)

    n = len(per)
    solved = sum(x["solved"] for x in per)
    rc = sum(x["correct_root_cause"] for x in per)
    avg_steps = sum(x["steps"] for x in per) / n
    total_calls = sum(x["steps"] for x in per)
    total_redundant = sum(x["redundant_calls"] for x in per)
    redundant_rate = (total_redundant / total_calls) if total_calls else 0.0

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
        "avg_steps": round(avg_steps, 3),
        "redundant_calls_total": total_redundant,
        "redundant_call_rate": round(redundant_rate, 4),
        "elapsed_s": round(elapsed, 1),
        "gpu": f"GPU{GPU_ID}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_scenario": [
            {"id": x["id"], "solved": bool(x["solved"]), "steps": x["steps"],
             "correct_root_cause": bool(x["correct_root_cause"]),
             "redundant_calls": x["redundant_calls"]}
            for x in per
        ],
    }

    print(f"\n== {label} ==")
    print(f"  solved            : {solved}/{n}  ({summary['solved_rate']*100:.1f}%)")
    print(f"  correct root-cause: {rc}/{n}  ({summary['root_cause_rate']*100:.1f}%)")
    print(f"  avg steps         : {summary['avg_steps']}")
    print(f"  redundant-call rate: {summary['redundant_call_rate']*100:.1f}%  "
          f"({total_redundant}/{total_calls})")
    print(f"  elapsed           : {summary['elapsed_s']}s")

    _append_score(summary)
    return summary


def _append_score(summary):
    data = []
    if os.path.exists(SCORES_PATH):
        try:
            with open(SCORES_PATH) as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        except (json.JSONDecodeError, OSError):
            data = []
    data.append(summary)
    with open(SCORES_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[scores] appended -> {SCORES_PATH}")


SEED = None

def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="gpt-oss-20b base (vllm same-path)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=18480)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-calls", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    SEED = args.seed
    evaluate(args.label, args.model, args.port, adapter_path=args.adapter,
             temp=args.temp, n_scenarios=args.n, max_calls=args.max_calls,
             verbose=args.verbose)


if __name__ == "__main__":
    main()
