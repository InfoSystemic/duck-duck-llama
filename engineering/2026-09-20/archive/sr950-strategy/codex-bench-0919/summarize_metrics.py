#!/usr/bin/env python3
"""Report isolated-request server work from before/after Prometheus snapshots.

Decode rate includes reasoning and tool-call tokens. It excludes prefill, queueing,
and client work; report wall time separately. Only valid with no other requests.
"""
import argparse
import json
from pathlib import Path


def parse(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.rsplit(None, 1)
            values[key] = float(value)
    return values


def summarize(before, after):
    names = {
        "prompt_tokens": "prompt_tokens_total",
        "cached_prompt_tokens": "prompt_tokens_cached_total",
        "prompt_seconds": "prompt_seconds_total",
        "generated_tokens": "tokens_predicted_total",
        "decode_seconds": "tokens_predicted_seconds_total",
        "draft_tokens": "spec_decode_num_draft_tokens_total",
        "accepted_draft_tokens": "spec_decode_num_accepted_tokens_total",
        "draft_verification_steps": "spec_decode_num_drafts_total",
    }
    result = {}
    for name, metric in names.items():
        key = "llamacpp:" + metric
        delta = after[key] - before[key]
        if delta < 0:
            raise ValueError(f"Counter reset for {key}; snapshots span a restart")
        result[name] = delta
    for name, numerator, denominator in [
        ("decode_tokens_per_second", "generated_tokens", "decode_seconds"),
        ("prefill_tokens_per_second", "prompt_tokens", "prompt_seconds"),
        ("draft_acceptance", "accepted_draft_tokens", "draft_tokens"),
        ("draft_tokens_per_verification", "draft_tokens", "draft_verification_steps"),
    ]:
        result[name] = result[numerator] / result[denominator] if result[denominator] else None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    print(json.dumps(summarize(parse(args.before), parse(args.after)), indent=2))
