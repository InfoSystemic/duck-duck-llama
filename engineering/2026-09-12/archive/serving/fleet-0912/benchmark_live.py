#!/usr/bin/env python3
"""Record real-server identity, quality, and comparable decode measurements.

Raw prompts exactly match fleet-0911/bench2.sh. Quality uses the chat endpoint
separately; short arithmetic answers are never used as throughput benchmarks.
This program does not launch or terminate servers.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request

from ab_profiles import owned_listener

PROMPTS = {
    "prose": "Write a detailed technical explanation of how a modern out-of-order CPU core executes instructions, covering fetch, decode, rename, scheduling, execution and retirement.",
    "code": "Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.",
    "analysis": "Analyse the causes of the 1997 Asian financial crisis, covering capital account liberalisation, currency pegs, short-term external debt and the IMF response.",
}
QUALITY = [
    ("arithmetic", "What is 17 * 23? Give the answer briefly.", "391"),
    ("fact", "What is the capital of Australia? Give the answer briefly.", "canberra"),
    ("state", "Mira has 3 red marbles and 5 blue marbles. She gives away 2 blue marbles and receives 4 red marbles. How many marbles does she now have in total? Give the answer briefly.", "10"),
]


def request(base, path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def pid_identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    if fields[0] == "Z":
        raise RuntimeError(f"PID {pid} is a zombie")
    return fields[19]


def metadata(pid):
    proc = Path(f"/proc/{pid}")
    env = {}
    for entry in (proc / "environ").read_bytes().split(b"\0"):
        key, sep, value = entry.partition(b"=")
        name = key.decode(errors="replace")
        if sep and (name.startswith(("GGML_", "LLAMA_", "OMP_", "GOMP_", "KMP_")) or name == "LD_LIBRARY_PATH"):
            env[name] = value.decode(errors="replace")
    libs = sorted({line.split()[-1] for line in (proc / "maps").read_text().splitlines()
                   if "/libggml" in line or "/libllama" in line})
    paths = [os.readlink(proc / "exe"), *libs]
    return {
        "pid": pid, "start_ticks": pid_identity(pid),
        "argv": [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s],
        "tuning_environment": env,
        "cpu_affinity": sorted(os.sched_getaffinity(pid)),
        "files_sha256": {p: hashlib.file_digest(open(p, "rb"), "sha256").hexdigest() for p in paths},
    }


def nondegenerate(text):
    compact = "".join(text.split())
    return bool(compact) and any(c.isalnum() for c in compact) and not (
        len(compact) >= 40 and max(compact.count(c) for c in set(compact)) / len(compact) > 0.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18131)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.reps < 1 or args.tokens < 32:
        parser.error("positive reps and at least 32 benchmark tokens required")
    args.out.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    start = pid_identity(args.pid)
    deadline = time.monotonic() + args.wait_seconds
    next_log = 0
    while True:
        if pid_identity(args.pid) != start:
            raise RuntimeError("server PID was replaced")
        try:
            health = request(base, "/health", timeout=3)
            if health.get("status") == "ok":
                break
        except (urllib.error.URLError, TimeoutError):
            pass
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("observation deadline reached; server was left running")
        if now >= next_log:
            io = Path(f"/proc/{args.pid}/io").read_text().splitlines()
            read = next((s for s in io if s.startswith("read_bytes:")), "read_bytes: unknown")
            print(f"WAIT pid={args.pid} alive; {read}", flush=True)
            next_log = now + 30
        time.sleep(3)
    record = metadata(args.pid)
    if not owned_listener(args.pid, args.port):
        raise RuntimeError("HTTP listener is not owned by the requested server PID")
    if "ngram" in " ".join(record["argv"]) and args.reps > 1:
        raise RuntimeError("ngram comparisons require a fresh server per repetition")
    models = request(base, "/v1/models", timeout=10)
    record["models"] = models
    # Require the requested exact model in advertised aliases before measuring.
    ids = {str(item.get("id", "")) for item in models.get("data", [])}
    if args.model not in ids:
        raise RuntimeError(f"requested model {args.model!r} absent from advertised {ids}")
    (args.out / "runtime.json").write_text(json.dumps(record, indent=2))
    print(f"READY {args.model} pid={args.pid}", flush=True)
    summary = {"model": args.model, "quality": [], "rows": [], "passed": True,
               "metric": "server decode tokens/s; raw 0911 prompts, no prompt cache",
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def save():
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    def check_pid():
        if pid_identity(args.pid) != start or not owned_listener(args.pid, args.port):
            raise RuntimeError("server changed during benchmark")

    for name, prompt, expected in QUALITY:
        check_pid()
        payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 384, "temperature": 0, "seed": 42,
                   "cache_prompt": False, "reasoning_effort": "low"}
        response = request(base, "/v1/chat/completions", payload, args.timeout)
        check_pid()
        (args.out / f"quality-{name}.json").write_text(json.dumps({"request": payload, "response": response}, indent=2))
        choice = response.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content") or ""
        answer = content.casefold()
        exact_token = re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", answer)
        negated = re.search(r"\b(?:not|isn't)\s+(?:\*\*)?" + re.escape(expected) + r"\b", answer)
        ok = bool(nondegenerate(content) and exact_token and not negated and choice.get("finish_reason") == "stop")
        summary["quality"].append({"name": name, "passed": ok, "content": content})
        summary["passed"] &= ok
        save()
        print(f"QUALITY {name}: {ok} {content[:140]!r}", flush=True)
    # Inference with broken numerics is not a performance result.
    if not summary["passed"]:
        return 1
    for rep in range(args.reps):
        for name, prompt in PROMPTS.items():
            check_pid()
            payload = {"prompt": prompt, "n_predict": args.tokens, "temperature": 0,
                       "seed": 42, "cache_prompt": False, "stream": False}
            t0 = time.monotonic()
            response = request(base, "/completion", payload, args.timeout)
            check_pid()
            wall = time.monotonic() - t0
            (args.out / f"{name}-{rep}.json").write_text(json.dumps({"request": payload, "response": response}, indent=2))
            timing = response.get("timings") or {}
            rate = timing.get("predicted_per_second")
            content = response.get("content") or ""
            ok = (isinstance(rate, (float, int)) and math.isfinite(rate) and rate > 0
                  and nondegenerate(content) and timing.get("predicted_n") == args.tokens
                  and timing.get("cache_n", 0) == 0)
            row = {"prompt": name, "rep": rep, "decode_tps": rate, "timings": timing,
                   "wall_seconds": wall, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                   "passed": ok}
            summary["rows"].append(row)
            summary["passed"] &= ok
            save()
            print(f"BENCH {name}#{rep}: {rate} tok/s; n={timing.get('predicted_n')} wall={wall:.2f}s", flush=True)
    rates = [r["decode_tps"] for r in summary["rows"] if r["passed"]]
    summary["mean_tps"] = statistics.mean(rates) if rates else None
    summary["median_tps"] = statistics.median(rates) if rates else None
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print(json.dumps({k: summary[k] for k in ("model", "passed", "mean_tps", "median_tps")}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
