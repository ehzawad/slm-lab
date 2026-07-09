#!/usr/bin/env python3
"""Baseline the incident-response env against a GGUF served by llama-server.

Given (label, gguf_path, port) this script:
  1. starts llama-server on GPU 1 (A6000) with --jinja (tool-calling grammar),
  2. wires an OpenAI-compatible /v1/chat/completions chat() into incident_harness,
  3. runs ALL build_scenarios(N) episodes,
  4. prints solved/total, avg steps, correct-root-cause rate, redundant-call rate,
  5. appends a result record to incident_scores.json,
  6. tears the server down via subprocess SIGINT (never pkill).

Usage:
  python eval_incident_gguf.py --label gpt-oss-20b \
      --gguf /path/to/gpt-oss-20b-Q4_K_M.gguf --port 18470 [--temp 0.6]

  # or run the full deep-dive baseline sweep (gpt-oss-20b, Qwen3.5-9B, Qwythos)
  python eval_incident_gguf.py --sweep
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from incident_harness import run_episode, SYSTEM_PROMPT  # noqa: E402
from incident_sim import build_scenarios, TOOLS_SPEC  # noqa: E402

LLAMA_SERVER = os.path.join(HERE, "..", "llama.cpp", "build", "bin", "llama-server")
SCORES_PATH = os.path.join(HERE, "incident_scores.json")
GPU_ID = 1  # A6000, per task spec (GPU-1 baseline agent)


# --------------------------------------------------------------------------
# llama-server lifecycle
# --------------------------------------------------------------------------
def start_server(gguf_path, port, temp, ctx=16384):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    log_path = os.path.join(HERE, f"server_{port}.log")
    log = open(log_path, "w")
    cmd = [
        LLAMA_SERVER, "-m", gguf_path,
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "999", "-c", str(ctx),
        "--jinja",
        "--temp", str(temp),
        "--parallel", "1",
    ]
    print(f"[server] launching: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={GPU_ID}) -> {log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, log_path


def wait_ready(proc, port, log_path, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = _tail(log_path, 40)
            raise RuntimeError(f"server exited early (code {proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    body = json.loads(r.read().decode())
                    if body.get("status") == "ok":
                        print(f"[server] ready after {time.time()-t0:.1f}s")
                        return
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready within {timeout}s")


def stop_server(proc, log):
    if proc.poll() is None:
        print("[server] sending SIGINT")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[server] SIGINT timed out; SIGTERM")
            proc.terminate()
            try:
                proc.wait(timeout=15)
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
def make_chat(port, temp, max_tokens=2048, retries=3):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def chat(messages, tools):
        payload = {
            "messages": _sanitize(messages),
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temp,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode()
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    body = json.loads(r.read().decode())
                msg = body["choices"][0]["message"]
                return _normalize(msg)
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode()[:500]}"
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)
            time.sleep(1.5 * (attempt + 1))
        # give up on this turn: return a no-tool-call message to end the episode
        return {"role": "assistant", "content": f"[backend error: {last_err}]"}

    return chat


def _sanitize(messages):
    """Ensure messages are JSON-serialisable OpenAI shapes.

    The harness appends the raw assistant message we returned (which may carry a
    null content) and tool messages. Normalise content=None to "" only where an
    empty string is required; keep tool_calls intact.
    """
    out = []
    for m in messages:
        mm = dict(m)
        if mm.get("role") == "assistant":
            if mm.get("content") is None:
                mm["content"] = ""
        out.append(mm)
    return out


def _normalize(msg):
    """Coerce a server message into the harness's expected assistant dict."""
    out = {"role": "assistant", "content": msg.get("content") or ""}
    tcs = msg.get("tool_calls") or []
    norm = []
    for tc in tcs:
        fn = tc.get("function", {}) or {}
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args)
        norm.append({
            "id": tc.get("id") or "call_%d" % len(norm),
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    if norm:
        out["tool_calls"] = norm
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(label, gguf_path, port, temp, n_scenarios=24, max_calls=15, verbose=False):
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(gguf_path)
    scenarios = build_scenarios(n_scenarios)
    proc, log, log_path = start_server(gguf_path, port, temp)
    per = []
    try:
        wait_ready(proc, port, log_path)
        chat = make_chat(port, temp)
        t0 = time.time()
        for i, sc in enumerate(scenarios, 1):
            r = run_episode(sc, chat, max_calls=max_calls, verbose=verbose)
            per.append(r)
            print(f"  [{i:>2}/{len(scenarios)}] {sc['id']:<28} "
                  f"solved={int(r['solved'])} steps={r['steps']:>2} "
                  f"root_cause={int(r['correct_root_cause'])} "
                  f"redundant={r['redundant_calls']}")
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
        "gguf": gguf_path,
        "temp": temp,
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
MODELS = os.path.join(HERE, "..", "models")
SWEEP = [
    ("gpt-oss-20b", os.path.join(MODELS, "gptoss20b", "gpt-oss-20b-Q4_K_M.gguf"), 1.0),
    ("Qwen3.5-9B",  os.path.join(MODELS, "q9b", "Qwen3.5-9B-Q4_K_M.gguf"), 0.7),
    ("Qwythos-9B",  os.path.join(MODELS, "qwythos", "qwythos-nomtp-Q4_K_M.gguf"), 0.6),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--gguf")
    ap.add_argument("--port", type=int, default=18470)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-calls", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="run gpt-oss-20b then Qwen3.5-9B then Qwythos on port 18470")
    args = ap.parse_args()

    if args.sweep:
        results = []
        for label, gguf, temp in SWEEP:
            print(f"\n########## {label} ##########")
            try:
                results.append(evaluate(label, gguf, args.port, temp,
                                        n_scenarios=args.n, max_calls=args.max_calls,
                                        verbose=args.verbose))
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] {label}: {e!r}")
                results.append({"label": label, "error": repr(e)})
        print("\n########## SWEEP SUMMARY ##########")
        for r in results:
            if "error" in r:
                print(f"  {r['label']:<14} ERROR {r['error']}")
            else:
                print(f"  {r['label']:<14} solved={r['solved']}/{r['n_scenarios']} "
                      f"root_cause={r['correct_root_cause']}/{r['n_scenarios']} "
                      f"avg_steps={r['avg_steps']} redundant={r['redundant_call_rate']}")
        return

    if not (args.label and args.gguf):
        ap.error("provide --label and --gguf, or use --sweep")
    evaluate(args.label, args.gguf, args.port, args.temp,
             n_scenarios=args.n, max_calls=args.max_calls, verbose=args.verbose)


if __name__ == "__main__":
    main()
