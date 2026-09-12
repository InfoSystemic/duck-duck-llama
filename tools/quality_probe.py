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


def is_degenerate_completion(text):
    """True for empty, slash-only, or otherwise non-content completions."""
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
    if compact[:24] == "/" * min(24, len(compact)) and compact.count("/") / len(compact) > 0.8:
        return True
    return False


def build_probe_payload(model, prompt, max_tokens=600):
    """Temperature-0 factual probe used on the real serving entry."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "reasoning_effort": "low",
    }


def evaluate_probe_response(data, validator):
    """Score a /v1/chat/completions JSON body from the real server."""
    if not isinstance(data, dict):
        return False, "", None, "not an object"
    choices = data.get("choices") or []
    if not choices:
        return False, "", None, "no choices"
    message = (choices[0] or {}).get("message") or {}
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    text = content or reasoning
    timings = data.get("timings") or {}
    rate = timings.get("predicted_per_second")
    if is_degenerate_completion(content if content else text):
        return False, text, rate, "degenerate"
    ok = bool(content) and bool(validator(content))
    return ok, text, rate, "ok" if ok else "validator_miss"


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
        payload = build_probe_payload(args.model, prompt)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read().decode())
            ok, text, rate, detail = evaluate_probe_response(data, validator)
            text = text.replace("\n", " ")
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
