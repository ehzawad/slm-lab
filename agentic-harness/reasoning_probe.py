#!/usr/bin/env python3
"""GSM8K execution-style reasoning probe for a GGUF model via llama-server.

Usage:
  python reasoning_probe.py <label> <gguf_path> <port> <temp> [n]

Starts its own llama-server (inheriting CUDA_VISIBLE_DEVICES from the caller),
scores accuracy on the first n GSM8K test questions, appends a row to
reasoning_scores.json, and always kills the server in a finally block.
"""
import subprocess, time, json, re, sys, os, signal, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = "/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin"
SCORES = f"{HERE}/reasoning_scores.json"
ENV = {**os.environ,
       "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
       "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def start_server(path, port):
    p = subprocess.Popen(
        [f"{BIN}/llama-server", "-m", path, "-ngl", "99", "--port", str(port),
         "-c", "8192", "--jinja", "--host", "127.0.0.1"],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(180):
        if p.poll() is not None:
            err = p.stderr.read().decode(errors="replace")[-2000:]
            print(f"llama-server died during startup:\n{err}", file=sys.stderr)
            sys.exit(1)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status == 200:
                return p
        except Exception:
            pass
        time.sleep(1)
    err = b""
    try:
        p.send_signal(signal.SIGINT); time.sleep(2)
        err = p.stderr.read() or b""
    except Exception:
        pass
    print(f"llama-server did not become healthy in 180s:\n{err.decode(errors='replace')[-2000:]}",
          file=sys.stderr)
    sys.exit(1)


def parse_int(text):
    """Extract predicted integer: prefer an 'Answer:' line, else last integer."""
    if not text:
        return None
    m = re.findall(r"[Aa]nswer\s*:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        s = m[-1]
    else:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "")
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def gold_int(answer):
    s = answer.split("####")[-1].strip().replace(",", "")
    return int(round(float(s)))


def chat(port, q, temp):
    body = {
        "messages": [{"role": "user",
                      "content": q + "\nReason step by step, then end with a line "
                                     "'Answer: <integer>'."}],
        "temperature": temp, "top_p": 0.95, "max_tokens": 2048,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip():
        text = msg.get("reasoning_content") or ""
    return text


def main():
    label, path, port, temp = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 40

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")

    p = start_server(path, port)
    correct = 0
    try:
        for i, row in enumerate(ds):
            gold = gold_int(row["answer"])
            text = chat(port, row["question"], temp)
            pred = parse_int(text)
            ok = pred is not None and pred == gold
            correct += ok
            print(f"  [{i+1}/{n}] pred={pred} gold={gold} {'OK' if ok else 'X'}", flush=True)
    finally:
        try:
            p.send_signal(signal.SIGINT); p.wait(timeout=30)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    acc = round(100.0 * correct / n, 1)
    print(f"=== {label}: {correct}/{n} = {acc}% (temp={temp}) ===", flush=True)

    rows = json.load(open(SCORES)) if os.path.exists(SCORES) else []
    rows.append({"label": label, "n": n, "accuracy_pct": acc, "temp": temp})
    json.dump(rows, open(SCORES, "w"), indent=2)


if __name__ == "__main__":
    main()
