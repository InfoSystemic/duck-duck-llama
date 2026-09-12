#!/usr/bin/env python3
"""Controlled decode-throughput harness for llama-server.

Reports emitted-token decode rate from the server's own timings block, so the
number is independent of client-side latency. Speculative arms additionally
report draft acceptance.
"""
import argparse, hashlib, json, re, statistics, sys, time, urllib.request, urllib.error

WORKLOADS = {
    "prose": "Write a clear 250-word explanation of how a B-tree index speeds up "
             "database lookups compared to a full table scan. Plain prose, no lists.",
    "code":  "Write a Python function `merge_intervals(intervals)` that merges "
             "overlapping closed intervals and returns them sorted. Include a "
             "docstring and three doctest examples.",
    "recall": "List the first 40 prime numbers, then explain in one paragraph why "
              "the sieve of Eratosthenes is more efficient than trial division.",
    "structured": "Output the integers from 1 through 60 in ascending order, "
                  "separated by a single space. Output only the integers.",
    "novel": "In exactly six complete sentences, explain how NUMA locality affects "
             "CPU inference throughput for a large language model. Use plain prose "
             "and do not use a list.",
}

METRIC_NAMES = {
    "draft": "llamacpp:spec_decode_num_draft_tokens_total",
    "accepted": "llamacpp:spec_decode_num_accepted_tokens_total",
}


def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def read_spec_metrics(host, port, timeout):
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode()
    except Exception:
        return None
    values = {}
    for key, metric in METRIC_NAMES.items():
        match = re.search(rf"^{re.escape(metric)}\s+([0-9.eE+-]+)$", body, re.MULTILINE)
        if match:
            values[key] = float(match.group(1))
    return values if len(values) == len(METRIC_NAMES) else None


def extract_decode_tps(data):
    """Decode tok/s from the server's own timings, llama or DeepSeek native."""
    if not isinstance(data, dict):
        return None
    tm = data.get("timings") or {}
    tps = tm.get("predicted_per_second")
    if isinstance(tps, (int, float)) and tps > 0:
        return float(tps)
    decode_s = tm.get("decode_seconds")
    if isinstance(decode_s, (int, float)) and decode_s > 0:
        n = tm.get("decode_tokens")
        if not isinstance(n, (int, float)):
            usage = data.get("usage") or {}
            n = (usage.get("completion_tokens") or 0) - 1
        if isinstance(n, (int, float)) and n > 0:
            return float(n) / float(decode_s)
    return None


def is_degenerate_completion(text):
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    compact = stripped.replace(" ", "").replace("\n", "").replace("\t", "")
    if not compact:
        return True
    if set(compact) <= {"/", "-", ".", ",", ";", ":"}:
        return True
    if compact[:24] == "/" * min(24, len(compact)) and compact.count("/") / max(len(compact), 1) > 0.8:
        return True
    return False


def validate_output(workload, content):
    if workload == "structured":
        expected = list(range(1, 61))
        try:
            actual = [int(token) for token in content.split()]
        except ValueError:
            return False, "non-integer output"
        return actual == expected, f"{len(actual)} integers"
    if workload == "novel":
        sentences = re.findall(r'[.!?]["\']?(?=\s|$)', content.strip())
        return len(sentences) == 6, f"{len(sentences)} sentences"
    return None, None


def run(args):
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    rows = []
    failures = 0
    for name in args.workloads.split(","):
        prompt = WORKLOADS[name]
        for rep in range(args.reps):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "seed": 42,
                "cache_prompt": False,
            }
            if args.reasoning_effort:
                payload["reasoning_effort"] = args.reasoning_effort
            if args.spec_n_max is not None:
                payload["speculative"] = {"n_max": args.spec_n_max}
                if args.spec_p_min is not None:
                    payload["speculative"]["p_min"] = args.spec_p_min
            metrics_before = read_spec_metrics(args.host, args.port, min(args.timeout, 10))
            t0 = time.time()
            try:
                d = post(url, payload, args.timeout)
            except urllib.error.HTTPError as e:
                print(f"  {name}#{rep} HTTP {e.code}: {e.read()[:300].decode()}", file=sys.stderr)
                failures += 1
                continue
            except Exception as e:
                print(f"  {name}#{rep} FAILED: {e}", file=sys.stderr)
                failures += 1
                continue
            wall = time.time() - t0
            tm = d.get("timings", {}) or {}
            ch = d.get("choices", [{}])[0].get("message", {}) or {}
            content = ch.get("content") or ""
            reasoning = ch.get("reasoning_content") or ""
            metrics_after = read_spec_metrics(args.host, args.port, min(args.timeout, 10))
            draft_n = accepted_n = None
            if metrics_before and metrics_after:
                draft_n = int(metrics_after["draft"] - metrics_before["draft"])
                accepted_n = int(metrics_after["accepted"] - metrics_before["accepted"])
            draft_acc = tm.get("draft_acceptance_rate")
            if draft_acc is None and draft_n:
                draft_acc = accepted_n / draft_n
            quality_ok, quality_detail = validate_output(name, content)
            decode_tps = extract_decode_tps(d)
            if decode_tps is None or is_degenerate_completion(content):
                failures += 1
            if quality_ok is False:
                failures += 1
            row = {
                "workload": name, "rep": rep,
                "decode_tps": decode_tps,
                "prompt_tps": tm.get("prompt_per_second"),
                "n_predict": tm.get("predicted_n"),
                "draft_n": draft_n,
                "accepted_n": accepted_n,
                "draft_acc": draft_acc,
                "wall_s": round(wall, 2),
                "content_chars": len(content),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "reasoning_chars": len(reasoning),
                "quality_ok": quality_ok,
                "quality_detail": quality_detail,
                "finish": d.get("choices", [{}])[0].get("finish_reason"),
            }
            rows.append(row)
            rate_text = (f"{decode_tps:.3f} tok/s"
                         if isinstance(decode_tps, (int, float)) else "NO TIMING")
            print(f"  {name}#{rep}: {rate_text}  "
                  f"n={row['n_predict']}  acc={row['draft_acc']}  "
                  f"chars={row['content_chars']}  quality={row['quality_ok']}  "
                  f"{row['finish']}", flush=True)

    vals = [r["decode_tps"] for r in rows if r["decode_tps"]]
    summary = {}
    if vals:
        summary = {
            "n": len(vals),
            "mean_tps": round(statistics.mean(vals), 3),
            "median_tps": round(statistics.median(vals), 3),
            "min_tps": round(min(vals), 3),
            "max_tps": round(max(vals), 3),
            "stdev": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
        }
        accs = [r["draft_acc"] for r in rows if r.get("draft_acc")]
        if accs:
            summary["mean_draft_acc"] = round(statistics.mean(accs), 4)
        print(f"\nSUMMARY {args.label}: mean {summary['mean_tps']} tok/s  "
              f"median {summary['median_tps']}  range {summary['min_tps']}-{summary['max_tps']}"
              + (f"  acc {summary['mean_draft_acc']}" if accs else ""))
    else:
        print(f"\nSUMMARY {args.label}: NO SUCCESSFUL RUNS", file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "summary": summary, "rows": rows}, f, indent=2)
    return 0 if vals and failures == 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="default")
    p.add_argument("--label", default="run")
    p.add_argument("--workloads", default="prose,code")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--spec-n-max", type=int, default=None)
    p.add_argument("--spec-p-min", type=float, default=None)
    p.add_argument("--out", default=None)
    sys.exit(run(p.parse_args()))
