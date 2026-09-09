#!/usr/bin/env python3
"""Fail-fast deterministic correctness checks for a running llama-server."""

import argparse
import json
import sys
import urllib.error
import urllib.request


PROBES = (
    ("math", "What is 17 * 23?", lambda text: "391" in text),
    ("fact", "What is the capital of Australia?", lambda text: "canberra" in text.lower()),
    (
        "logic",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?",
        lambda text: "yes" in text.lower() and "lazz" in text.lower(),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("port_pos", nargs="?", type=int)
    parser.add_argument("model_pos", nargs="?")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", dest="port_opt", type=int)
    parser.add_argument("--model", dest="model_opt")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    args.port = args.port_opt or args.port_pos or 8080
    args.model = args.model_opt or args.model_pos or "default"
    return args


def run():
    args = parse_args()
    failures = 0
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    for name, prompt, validator in PROBES:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.0,
            "seed": 42,
            "reasoning_effort": "low",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read().decode())
            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip().replace("\n", " ")
            reasoning = (message.get("reasoning_content") or "").strip().replace("\n", " ")
            text = content or reasoning
            ok = bool(content) and validator(content)
            rate = (data.get("timings") or {}).get("predicted_per_second")
            rate_text = f" {rate:.3f} tok/s" if isinstance(rate, (int, float)) else ""
            print(f"  {name:6}: {'PASS' if ok else 'FAIL'}{rate_text} {text[:110]!r}")
            failures += not ok
        except urllib.error.HTTPError as error:
            body = error.read(300).decode(errors="replace")
            print(f"  {name:6}: ERROR HTTP {error.code} {body!r}")
            failures += 1
        except Exception as error:
            print(f"  {name:6}: ERROR {error}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
