#!/usr/bin/env python3
"""toolcall-probe.py [port] [n] -- sample the same tool-calling request n times at the server's default sampling
(Xiaomi's recommended temperature 1.0 / top_p 0.95) and count calls whose arguments are not exactly right.
Before the 09-22 chat-parser fix: 3 of 10 malformed ("\\n\\nParis\\n\\n", "Paris\\n</invoke>", "</invoke>")."""
import json, sys, urllib.request
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                          "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris right now, in Celsius? Use the tool."}]
bad = 0
for seed in range(N):
    body = {"messages": msgs, "tools": tools, "seed": seed, "max_tokens": 500}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    calls = r["choices"][0]["message"].get("tool_calls") or []
    args = calls[0]["function"].get("arguments") if calls else None
    try:
        a = json.loads(args) if args else None
        ok = calls and calls[0]["function"]["name"] == "get_weather" and a.get("city") == "Paris" and a.get("unit", "C") == "C"
    except Exception:
        ok = False
    bad += not ok
    if not ok:
        print(f"seed {seed}: MALFORMED finish={r['choices'][0]['finish_reason']} args={args!r}", flush=True)
print(f"tool calls malformed: {bad} of {N}")
