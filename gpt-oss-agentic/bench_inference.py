#!/usr/bin/env python3
"""INFERENCE-PERF benchmark for gpt-oss-20b served by llama-server on the A5000.

Measures, with the CORRECT gpt-oss sampling (temp 1.0 top_p 1.0 top_k 0 min_p 0):
  (1) llama.cpp decode tok/s + TTFT on agentic-style prompts, and the effect of
      --cache-type-k q8_0 (KV-cache quantization).
  (2) reasoning_effort low vs medium vs high: tokens generated + total latency.
  (3) prompt/prefix caching effect on a repeated system+tool-schema prefix.

GPU 0 (A5000) ONLY. Never touches GPU 1. Server lifecycle via subprocess SIGINT.

Usage:
  CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python bench_inference.py [--quick]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from incident_harness import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from incident_sim import build_scenarios, IncidentSim, TOOLS_SPEC  # noqa: E402

LLAMA_SERVER = os.path.join(HERE, "..", "llama.cpp", "build", "bin", "llama-server")
GGUF = os.path.join(HERE, "..", "models", "gptoss20b", "gpt-oss-20b-Q4_K_M.gguf")
GPU_ID = 0  # A5000 ONLY
PORT = 18492

# gpt-oss correct sampling (override any server defaults per-request)
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------
def start_server(extra_args, tag):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    log_path = os.path.join(HERE, f"bench_server_{tag}.log")
    log = open(log_path, "w")
    cmd = [
        LLAMA_SERVER, "-m", GGUF,
        "--host", "127.0.0.1", "--port", str(PORT),
        "-ngl", "99", "--jinja", "-fa", "on",
        "-ub", "2048", "-b", "2048", "--ctx-size", "8192",
        "--parallel", "1",
    ] + extra_args
    print(f"[server:{tag}] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, log_path


def wait_ready(proc, log_path, timeout=600):
    url = f"http://127.0.0.1:{PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early ({proc.returncode}):\n{_tail(log_path,40)}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200 and json.loads(r.read().decode()).get("status") == "ok":
                    print(f"[server] ready in {time.time()-t0:.1f}s")
                    return
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready within {timeout}s")


def stop_server(proc, log):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log.close()


def _tail(path, n):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


# --------------------------------------------------------------------------
# request helper — returns timing + token usage from server
# --------------------------------------------------------------------------
def chat_completion(messages, tools=None, reasoning_effort="medium",
                    max_tokens=2048, cache_prompt=True):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "cache_prompt": cache_prompt,
        "reasoning_effort": reasoning_effort,
        "stream": True,
        **SAMPLING,
    }
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    gen_tokens = 0
    timings = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            # count any streamed content/reasoning token chunk as first-token signal
            has_text = bool(delta.get("content")) or bool(delta.get("reasoning_content")) \
                or bool(delta.get("tool_calls"))
            if has_text and ttft is None:
                ttft = time.time() - t0
            if has_text:
                gen_tokens += 1
            if obj.get("timings"):
                timings = obj["timings"]
    total = time.time() - t0
    # prefer server-reported token counts + speeds when present
    n_pred = timings.get("predicted_n", gen_tokens)
    n_prompt = timings.get("prompt_n")
    decode_tps = timings.get("predicted_per_second")
    prefill_tps = timings.get("prompt_per_second")
    server_ttft = None
    if timings.get("prompt_ms") is not None:
        server_ttft = timings["prompt_ms"] / 1000.0
    return {
        "ttft_s": server_ttft if server_ttft is not None else ttft,
        "stream_ttft_s": ttft,
        "total_s": total,
        "gen_tokens": n_pred,
        "prompt_tokens": n_prompt,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
    }


# --------------------------------------------------------------------------
# prompt builders — realistic agentic (system + 8 tool schemas + user)
# --------------------------------------------------------------------------
def agentic_messages(seed=0):
    scen = build_scenarios(1)[0]
    sim = IncidentSim(scen)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sim)},
    ]


# reasoning-heavy, no-tools prompt to force sustained CoT so reasoning_effort
# actually changes generated-token count, and to get a clean decode tok/s.
REASON_PROMPT = (
    "A logic puzzle: five on-call engineers (Ana, Ben, Cy, Dee, Eli) each own "
    "exactly one of five microservices and were paged at five distinct minutes "
    "past the hour (1,2,3,4,5). Clues: the owner of 'ledger' was paged before "
    "Ana but after Ben; Cy was paged at an even minute; Dee owns 'cache' and "
    "was paged immediately after the 'ledger' owner; Eli was paged at minute 5 "
    "and does not own 'gate'; the 'gate' owner was paged at an odd minute "
    "before Cy. Work out who owns what and the exact page order, showing every "
    "deduction step. Then write a short postmortem paragraph."
)


def reason_messages():
    return [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": REASON_PROMPT},
    ]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "n/a"


# --------------------------------------------------------------------------
# benchmark passes
# --------------------------------------------------------------------------
def bench_pass(label, reasoning_efforts, n_prompts, max_tokens):
    """Run n_prompts agentic prompts at each reasoning effort, tool-calling on."""
    rows = []
    for eff in reasoning_efforts:
        results = []
        for i in range(n_prompts):
            msgs = agentic_messages(i)
            r = chat_completion(msgs, tools=TOOLS_SPEC, reasoning_effort=eff,
                                max_tokens=max_tokens, cache_prompt=True)
            results.append(r)
            print(f"  [{label}|{eff}] p{i}: ttft={fmt(r['ttft_s'],3)}s "
                  f"gen={r['gen_tokens']}tok decode={fmt(r['decode_tps'])}tps "
                  f"total={fmt(r['total_s'],2)}s")
        rows.append({
            "label": label, "reasoning_effort": eff,
            "ttft_s": mean([r["ttft_s"] for r in results]),
            "decode_tps": mean([r["decode_tps"] for r in results]),
            "prefill_tps": mean([r["prefill_tps"] for r in results]),
            "gen_tokens": mean([r["gen_tokens"] for r in results]),
            "total_s": mean([r["total_s"] for r in results]),
            "prompt_tokens": mean([r["prompt_tokens"] for r in results]),
        })
    return rows


def bench_reason_effort(reasoning_efforts, n_samples):
    """No-tools CoT prompt: show tokens generated + total latency vs effort."""
    rows = []
    for eff in reasoning_efforts:
        results = []
        for i in range(n_samples):
            r = chat_completion(reason_messages(), tools=None, reasoning_effort=eff,
                                max_tokens=4096, cache_prompt=True)
            results.append(r)
            print(f"  [reason|{eff}] s{i}: gen={r['gen_tokens']}tok "
                  f"decode={fmt(r['decode_tps'])}tps ttft={fmt(r['ttft_s'],3)}s "
                  f"total={fmt(r['total_s'],2)}s")
        rows.append({
            "reasoning_effort": eff,
            "gen_tokens": mean([r["gen_tokens"] for r in results]),
            "total_s": mean([r["total_s"] for r in results]),
            "ttft_s": mean([r["ttft_s"] for r in results]),
            "decode_tps": mean([r["decode_tps"] for r in results]),
        })
    return rows


def bench_decode_sustained():
    """Long single generation -> clean steady-state decode tok/s."""
    return chat_completion(reason_messages(), tools=None, reasoning_effort="high",
                           max_tokens=1024, cache_prompt=True)


def bench_prompt_cache():
    """Same long system+tool-schema prefix, cold vs warm. Measures TTFT drop."""
    msgs = agentic_messages(0)
    # cold: fresh prefix (cache_prompt False forces reprocessing)
    cold = chat_completion(msgs, tools=TOOLS_SPEC, reasoning_effort="low",
                           max_tokens=8, cache_prompt=False)
    # prime the cache
    chat_completion(msgs, tools=TOOLS_SPEC, reasoning_effort="low",
                    max_tokens=8, cache_prompt=True)
    # warm: identical prefix reused
    warm = chat_completion(msgs, tools=TOOLS_SPEC, reasoning_effort="low",
                           max_tokens=8, cache_prompt=True)
    return cold, warm


# --------------------------------------------------------------------------
def run_config(tag, extra_args, reasoning_efforts, n_prompts, max_tokens,
               do_cache_test):
    proc, log, log_path = start_server(extra_args, tag)
    out = {}
    try:
        wait_ready(proc, log_path)
        time.sleep(1)
        # warmup (compile kernels / graph)
        chat_completion(agentic_messages(0), tools=TOOLS_SPEC,
                        reasoning_effort="low", max_tokens=8)
        out["rows"] = bench_pass(tag, reasoning_efforts, n_prompts, max_tokens)
        out["sustained_decode"] = bench_decode_sustained()
        if do_cache_test:
            print("  -- reasoning-effort token/latency tradeoff (no tools) --")
            out["reason_effort"] = bench_reason_effort(reasoning_efforts,
                                                       n_prompts)
            cold, warm = bench_prompt_cache()
            out["cache"] = {"cold": cold, "warm": warm}
    finally:
        stop_server(proc, log)
        time.sleep(3)  # let VRAM free before next server
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer prompts / smaller max_tokens for a fast smoke run")
    args = ap.parse_args()

    n_prompts = 2 if args.quick else 3
    max_tokens = 1024 if args.quick else 2048
    efforts = ["low", "medium", "high"]

    report = {"gpu": "A5000 (GPU0)", "gguf": os.path.basename(GGUF),
              "sampling": SAMPLING, "configs": {}}

    # Config A: default f16 KV cache
    print("\n===== CONFIG A: f16 KV cache (-fa) =====")
    report["configs"]["f16_kv"] = run_config(
        "f16kv", [], efforts, n_prompts, max_tokens, do_cache_test=True)

    # Config B: q8_0 K cache (--cache-type-k q8_0)
    print("\n===== CONFIG B: q8_0 K cache =====")
    report["configs"]["q8_k"] = run_config(
        "q8k", ["--cache-type-k", "q8_0"], efforts, n_prompts, max_tokens,
        do_cache_test=False)

    out_path = os.path.join(HERE, "bench_inference_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\n[written] {out_path}")


def print_report(report):
    print("\n" + "=" * 78)
    print("INFERENCE-PERF REPORT — gpt-oss-20b Q4_K_M, A5000, temp1/top_p1/top_k0/min_p0")
    print("=" * 78)

    for cfg_name, cfg in report["configs"].items():
        print(f"\n### KV cache: {cfg_name}")
        print(f"{'effort':<8}{'TTFT(s)':>9}{'decode tok/s':>14}"
              f"{'gen tok':>9}{'total(s)':>10}{'prefill tok/s':>15}")
        for r in cfg["rows"]:
            print(f"{r['reasoning_effort']:<8}{fmt(r['ttft_s'],3):>9}"
                  f"{fmt(r['decode_tps']):>14}{fmt(r['gen_tokens'],0):>9}"
                  f"{fmt(r['total_s'],2):>10}{fmt(r['prefill_tps']):>15}")
        sd = cfg.get("sustained_decode")
        if sd:
            print(f"  sustained decode ({fmt(sd['gen_tokens'],0)} tok): "
                  f"{fmt(sd['decode_tps'])} tok/s")
        if "reason_effort" in cfg:
            print("  reasoning-effort tradeoff (no tools, CoT puzzle):")
            print(f"    {'effort':<8}{'gen tok':>9}{'total(s)':>10}{'decode tok/s':>14}")
            for r in cfg["reason_effort"]:
                print(f"    {r['reasoning_effort']:<8}{fmt(r['gen_tokens'],0):>9}"
                      f"{fmt(r['total_s'],2):>10}{fmt(r['decode_tps']):>14}")
        if "cache" in cfg:
            c, w = cfg["cache"]["cold"], cfg["cache"]["warm"]
            print(f"  prompt-cache prefix reuse: cold TTFT={fmt(c['ttft_s'],3)}s "
                  f"-> warm TTFT={fmt(w['ttft_s'],3)}s "
                  f"(prefix={c['prompt_tokens']} tok)")


if __name__ == "__main__":
    main()
