#!/usr/bin/env python3
"""Summarize armed graph phase measurements by token and graph shape."""
import argparse
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument("case_dir", type=Path)
a = p.parse_args()
groups = {}
records = []
for line in (a.case_dir / "server.log").read_text(errors="replace").splitlines():
    match = re.search(r"\b(GRAPH_PHASE|META_PHASE) (.*) ms", line)
    if not match:
        continue
    kind, payload = match.groups()
    values = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", payload)}
    integers = ("tokens", "nodes", "reused") if kind == "GRAPH_PHASE" else ("nodes", "rebuild")
    shape = {k: int(values[k]) for k in integers}
    key = kind + " " + " ".join(f"{k}={v}" for k, v in shape.items())
    groups.setdefault(key, []).append({k: v for k, v in values.items() if k not in integers})
    records.append({"kind": kind, **values})

summary = {}
for key, rows in groups.items():
    summary[key] = {"count": len(rows), "timings_ms": {
        k: {"mean": statistics.mean(r[k] for r in rows),
            "median": statistics.median(r[k] for r in rows),
            "min": min(r[k] for r in rows), "max": max(r[k] for r in rows)}
        for k in rows[0]}}
output = {"note": "Armed diagnostic wall times; use unprofiled requests for throughput.",
          "summary": summary, "records": records}
(a.case_dir / "phase-profile-summary.json").write_text(json.dumps(output, indent=2))
print(json.dumps(summary, indent=2))
