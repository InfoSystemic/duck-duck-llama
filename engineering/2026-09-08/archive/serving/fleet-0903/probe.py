#!/usr/bin/env python3
"""Deterministic correctness probe: ask for 17*23 and report what came back.

Kept as a file rather than an inline -c so the quoting cannot break the caller.
Exit code 0 when the answer contains 391, 1 otherwise, so a benchmark run can
refuse to trust throughput from a model that is producing nonsense.
"""
import json
import sys
import urllib.request

port = sys.argv[1]
model = sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 700

body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "What is 17 * 23? Reply with only the number."}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "seed": 42,
    "cache_prompt": False,
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % port,
    data=body, headers={"Content-Type": "application/json"})
try:
    raw = urllib.request.urlopen(req, timeout=900).read().decode()
except Exception as exc:  # HTTPError carries the server's JSON error body
    raw = getattr(exc, "read", lambda: str(exc).encode())().decode(errors="replace")

try:
    d = json.loads(raw)
    choice = d["choices"][0]
    msg = choice["message"]
    text = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or ""
    if not text:
        text = "[no content; finish=%s; %d reasoning chars: %s]" % (
            choice.get("finish_reason"), len(reasoning), reasoning[-80:].replace("\n", " "))
    tm = d.get("timings") or {}
    print("answer: %s" % text[:100].replace("\n", " "))
    if tm:
        print("        %.2f tok/s over %d tokens, draft %s/%s accepted" % (
            tm.get("predicted_per_second", 0), tm.get("predicted_n", 0),
            tm.get("draft_n_accepted", "-"), tm.get("draft_n", "-")))
    sys.exit(0 if "391" in text else 1)
except Exception:
    print("non-JSON / error body: %s" % raw[:300].replace("\n", " "))
    sys.exit(1)
