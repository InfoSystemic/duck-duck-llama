#!/usr/bin/env python3
# Smoke test for the MiMo-V2.6-Pro-RL llama-server: correctness first (math with thinking, no-think mode,
# tool call, code, vision via mmproj), then raw decode/prefill speed. Stdlib only.
#   smoke-mimo.py PORT TAG [--speed-only]    -> results/<TAG>.json + a summary on stdout

import base64
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

PORT, TAG = int(sys.argv[1]), sys.argv[2]
SPEED_ONLY = "--speed-only" in sys.argv
BASE = f"http://127.0.0.1:{PORT}"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def post(path, body, timeout=1200):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def chat(messages, **kw):
    body = {"model": "mimo-v2.6-pro", "messages": messages, "temperature": 1.0, "top_p": 0.95, "max_tokens": 2048}
    body.update(kw)
    resp, wall = post("/v1/chat/completions", body)
    msg = resp["choices"][0]["message"]
    return msg, resp.get("timings", {}), wall, resp["choices"][0].get("finish_reason")


def png_red_square():
    w = h = 256
    rows = []
    for y in range(h):
        row = bytearray(b"\xff\xff\xff" * w)
        if 64 <= y < 192:
            row[64 * 3:192 * 3] = b"\xdd\x11\x11" * 128
        rows.append(b"\x00" + bytes(row))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


results = {"tag": TAG, "port": PORT, "t": time.strftime("%Y-%m-%d %H:%M:%S")}


def record(name, ok, **info):
    results[name] = dict(ok=ok, **info)
    short = {k: v for k, v in info.items() if k in ("answer", "tok_s", "pp_tok_s", "wall_s", "n", "tool", "args", "finish")}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(short)[:400]}", flush=True)


def run(name, fn):
    try:
        fn()
    except Exception as e:  # keep going: a smoke test reports every check
        record(name, False, error=f"{type(e).__name__}: {e}")


def t_models():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    record("models", "mimo-v2.6-pro" in ids, ids=ids)


def t_math_thinking():
    msg, tm, wall, fin = chat([{"role": "user", "content": "What is 84 * 3 / 2? Reply with just the number."}])
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    record("math_thinking", "126" in content, answer=content.strip()[:200], reasoning_chars=len(reasoning),
           reasoning_head=reasoning[:300], n=tm.get("predicted_n"), tok_s=tm.get("predicted_per_second"), wall_s=round(wall, 1), finish=fin)


def t_no_think():
    msg, tm, wall, fin = chat([{"role": "user", "content": "Name the three primary colors of light, comma separated."}],
                              chat_template_kwargs={"enable_thinking": False}, max_tokens=200)
    content = (msg.get("content") or "").lower()
    record("no_think", all(c in content for c in ("red", "green", "blue")), answer=content.strip()[:200],
           reasoning_chars=len(msg.get("reasoning_content") or ""), tok_s=tm.get("predicted_per_second"), wall_s=round(wall, 1), finish=fin)


def t_tool_call():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]
    msg, tm, wall, fin = chat([{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}], tools=tools)
    calls = msg.get("tool_calls") or []
    name = calls[0]["function"]["name"] if calls else None
    args = calls[0]["function"].get("arguments") if calls else None
    ok = name == "get_weather" and args is not None and "paris" in str(args).lower()
    record("tool_call", ok, tool=name, args=args, content=(msg.get("content") or "")[:200], finish=fin, wall_s=round(wall, 1))


def t_code():
    msg, tm, wall, fin = chat([{"role": "user", "content": "Write a Python function is_palindrome(s) that ignores case and "
                                "non-alphanumeric characters. Code only, with a one-line docstring."}],
                              chat_template_kwargs={"enable_thinking": False}, max_tokens=400)
    content = msg.get("content") or ""
    record("code", "def is_palindrome" in content, answer=content.strip()[:600], tok_s=tm.get("predicted_per_second"), wall_s=round(wall, 1))


def t_vision():
    img = "data:image/png;base64," + base64.b64encode(png_red_square()).decode()
    msg, tm, wall, fin = chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img}},
        {"type": "text", "text": "What color is the square in this image? Answer with one word."}]}],
        chat_template_kwargs={"enable_thinking": False}, max_tokens=50)
    content = (msg.get("content") or "").lower()
    record("vision", "red" in content, answer=content.strip()[:200], wall_s=round(wall, 1))


def t_speed():
    # raw completion, greedy: ~1.9K-token prompt, 256 generated tokens
    text = Path(__file__).resolve().parent.joinpath("fill_experts.py").read_text()[:7000]
    prompt = f"Here is a Python script:\n\n{text}\n\nSummarize what this script does in detail:\n"
    resp, wall = post("/completion", {"prompt": prompt, "n_predict": 256, "temperature": 0, "cache_prompt": False})
    tm = resp.get("timings", {})
    record("speed", tm.get("predicted_n", 0) > 0, n=tm.get("predicted_n"), tok_s=tm.get("predicted_per_second"),
           pp_n=tm.get("prompt_n"), pp_tok_s=tm.get("prompt_per_second"), wall_s=round(wall, 1),
           draft_n=tm.get("draft_n"), draft_accepted=tm.get("draft_n_accepted"), head=resp.get("content", "")[:300])


tests = [("speed", t_speed)] if SPEED_ONLY else [
    ("models", t_models), ("math_thinking", t_math_thinking), ("no_think", t_no_think), ("tool_call", t_tool_call),
    ("code", t_code), ("vision", t_vision), ("speed", t_speed)]
for name, fn in tests:
    run(name, fn)
(OUT / f"{TAG}.json").write_text(json.dumps(results, indent=2))
n_ok = sum(1 for k, v in results.items() if isinstance(v, dict) and v.get("ok"))
n_all = sum(1 for v in results.values() if isinstance(v, dict))
print(f"SMOKE {TAG}: {n_ok}/{n_all} passed -> {OUT / (TAG + '.json')}", flush=True)
