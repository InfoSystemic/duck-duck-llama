#!/usr/bin/env python3
"""Drive the real serving entry of each named model; assert decode tok/s and non-garbage.

Uses the existing quality_probe / decode_bench helpers against live
/v1/chat/completions (or /completion). Decode rate comes from the server's
timings block, compared to this-host-baselines.json — not a hardcoded tok/s
in the test, not a reimplementation of the server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOOLS = Path("/home/kwebb/InfoSystemic/AI-Server/engines/llama-llama-duck/tools")
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from decode_bench import extract_decode_tps, is_degenerate_completion  # noqa: E402
from quality_probe import (  # noqa: E402
    PROBES,
    build_probe_payload,
    evaluate_probe_response,
)

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINES = HERE / "this-host-baselines.json"


def load_baselines(path=None):
    path = Path(path) if path else DEFAULT_BASELINES
    return json.loads(path.read_text())


def meets_baseline(measured, baseline, spread=0.05):
    """Pass if measured is at least baseline, or within documented run-to-run spread."""
    if measured is None or baseline is None:
        return False
    measured = float(measured)
    baseline = float(baseline)
    if measured >= baseline:
        return True
    return measured >= baseline * (1.0 - spread)


def chat_url(host, port):
    return f"http://{host}:{port}/v1/chat/completions"


def post_json(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode())


def completion_text(data):
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content") or ""
        if content:
            return content
        if data.get("content"):
            return data["content"]
        return message.get("reasoning_content") or ""
    return data.get("content") or ""


def probe_once(host, port, model, prompt, timeout, max_tokens=128):
    payload = build_probe_payload(model, prompt, max_tokens=max_tokens)
    url = chat_url(host, port)
    status, data = post_json(url, payload, timeout)
    text = completion_text(data)
    tps = extract_decode_tps(data)
    return {
        "url": url,
        "http_status": status,
        "request": payload,
        "response": data,
        "text": text,
        "decode_tps": tps,
        "degenerate": is_degenerate_completion(text),
    }


def run_model(name, spec, host, timeout, reps, out_dir, spread):
    port = spec["port"]
    model = spec["model"]
    prompt = spec["prompt"]
    expect = spec.get("expect_substr")
    results = []
    failures = []
    for i in range(1, reps + 1):
        t0 = time.time()
        try:
            row = probe_once(host, port, model, prompt, timeout)
        except urllib.error.HTTPError as error:
            body = error.read(400).decode(errors="replace")
            row = {
                "http_status": error.code,
                "error": body,
                "decode_tps": None,
                "degenerate": True,
                "text": "",
            }
            failures.append(f"{name}#{i} HTTP {error.code}")
        except Exception as error:
            row = {
                "http_status": None,
                "error": repr(error),
                "decode_tps": None,
                "degenerate": True,
                "text": "",
            }
            failures.append(f"{name}#{i} {error}")
        row["wall_s"] = time.time() - t0
        row["rep"] = i
        if out_dir:
            dest = Path(out_dir) / f"{name}-probe-{i}.json"
            dest.write_text(json.dumps(row, indent=2, default=str))
        text = row.get("text") or ""
        if row.get("http_status") != 200:
            failures.append(f"{name}#{i} bad status {row.get('http_status')}")
        elif row.get("degenerate"):
            failures.append(f"{name}#{i} degenerate {text[:80]!r}")
        elif expect and expect.lower() not in text.lower() and expect not in text:
            # DeepSeek greeting and GLM math use this; still require non-degenerate
            # content even if the substr check is the quality gate.
            failures.append(f"{name}#{i} missing {expect!r} in {text[:80]!r}")
        results.append(row)
        print(
            f"  {name}#{i}: status={row.get('http_status')} "
            f"tps={row.get('decode_tps')} degenerate={row.get('degenerate')} "
            f"{text[:90]!r}",
            flush=True,
        )
    rates = [r["decode_tps"] for r in results if isinstance(r.get("decode_tps"), (int, float))]
    baseline = spec.get("baseline_tps")
    rate_ok = bool(rates) and all(meets_baseline(r, baseline, spread) for r in rates)
    if rates and not rate_ok:
        failures.append(
            f"{name} decode {rates} below baseline {baseline} (spread {spread})"
        )
    if len(rates) < reps:
        failures.append(f"{name} missing timings ({len(rates)}/{reps})")
    return {
        "name": name,
        "rates": rates,
        "baseline_tps": baseline,
        "rate_ok": rate_ok,
        "failures": failures,
        "passed": not failures,
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    parser.add_argument("--models", default="")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    baselines = load_baselines(args.baselines)
    spread = float(baselines.get("spread", 0.05))
    wanted = [m.strip() for m in args.models.split(",") if m.strip()] or list(
        baselines["models"]
    )
    summary = {"passed": True, "models": {}, "spread": spread}
    for name in wanted:
        spec = baselines["models"][name]
        report = run_model(name, spec, args.host, args.timeout, args.reps, args.out_dir, spread)
        summary["models"][name] = {k: report[k] for k in ("rates", "baseline_tps", "rate_ok", "failures", "passed")}
        if not report["passed"]:
            summary["passed"] = False
        print(
            f"SUMMARY {name}: passed={report['passed']} rates={report['rates']} "
            f"baseline={report['baseline_tps']} failures={report['failures']}",
            flush=True,
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
