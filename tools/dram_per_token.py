#!/usr/bin/env python3
"""Measure ACTUAL DRAM bytes per generated token from the uncore memory-controller counters.

`active_bytes.py` computes what a forward pass SHOULD stream from the tensor table. This measures what the machine
actually moved, which is the only way to settle three things a static model cannot:

  - whether decode is bandwidth-bound or overhead-bound (compare achieved tok/s to bytes/token x bandwidth);
  - how much traffic is waste (KV re-reads, cross-socket all-reduce copies, page-cache churn);
  - for a speculative MoE, how large the expert UNION really is, since adjacent tokens route similarly and the
    independent-routing model in active_bytes.py is only an upper bound. Run once speculative and once with
    speculation off; the two bytes/token figures bracket the union.

Needs Intel uncore IMC PMUs (server parts; consumer chips expose a different `uncore_imc` with the same idea) and
either root or perf_event_paranoid <= 0. Check the PMU cpumask first: on a 4-socket box each `uncore_imc_N` carries
one CPU per socket (e.g. `0,16,32,48`), which means perf aggregates all sockets and the totals are machine-wide.

Usage: dram_per_token.py PORT [n_predict] [label] [--controllers N] [--bandwidth GBPS]
"""
import subprocess, sys, json, time, urllib.request, argparse

ap = argparse.ArgumentParser()
ap.add_argument("port"); ap.add_argument("n", nargs="?", type=int, default=192)
ap.add_argument("label", nargs="?", default="run")
ap.add_argument("--controllers", type=int, default=6, help="number of uncore_imc_N PMUs to sum")
ap.add_argument("--bandwidth", type=float, default=381.0, help="measured aggregate GB/s for the extraction figure")
ap.add_argument("--prompt", default="Write a detailed technical explanation of how a modern out-of-order CPU core "
                                    "executes instructions, covering fetch, decode, rename, scheduling, execution "
                                    "and retirement.")
a = ap.parse_args()
BW = a.bandwidth * 1e9
EV = ",".join(f"uncore_imc_{i}/cas_count_{d}/" for i in range(a.controllers) for d in ("read", "write"))

def counters(cmd=None, secs=None):
    args = ["sudo", "-n", "perf", "stat", "-a", "-x", ",", "-e", EV, "--"] + (cmd or ["sleep", str(secs)])
    p = subprocess.run(args, capture_output=True, text=True)
    out = {}
    for line in p.stderr.splitlines():
        f = line.split(",")
        if len(f) > 3 and f[0].replace(".", "").isdigit():
            out[f[2]] = float(f[0]) * 1024 * 1024      # perf scales cas_count_* to MiB
    if not out:
        raise SystemExit("no counter values; check sudo, perf_event_paranoid and that uncore_imc PMUs exist:\n"
                         "  ls /sys/bus/event_source/devices/uncore_imc*")
    return out

def generate(n):
    body = json.dumps({"prompt": a.prompt, "n_predict": n, "temperature": 0,
                       "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["timings"]

rd = lambda c: sum(v for k, v in c.items() if "read" in k)
wr = lambda c: sum(v for k, v in c.items() if "write" in k)

print(f"=== {a.label}: idle baseline (4 s)", flush=True)
idle = counters(secs=4)
idle_r, idle_w = rd(idle) / 4, wr(idle) / 4
print(f"  idle read {idle_r/1e9:6.2f} GB/s   write {idle_w/1e9:6.2f} GB/s")

generate(16)                                  # warm the slot so cycle 1 is not a graph build
t0 = time.time()
c = counters([sys.executable, "-c",
              "import sys,json,urllib.request;"
              "b=json.dumps({'prompt':sys.argv[1],'n_predict':int(sys.argv[2]),'temperature':0,"
              "'cache_prompt':False,'stream':False}).encode();"
              "r=urllib.request.Request('http://127.0.0.1:'+sys.argv[3]+'/completion',data=b,"
              "headers={'Content-Type':'application/json'});"
              "json.load(urllib.request.urlopen(r,timeout=900))", a.prompt, str(a.n), a.port])
wall = time.time() - t0
t = generate(a.n)                             # re-ask for the decode/prefill split
ntok, dec = t["predicted_n"], t["predicted_ms"] / 1000.0
dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
net = rd(c) - idle_r * wall

print(f"\n  wall {wall:.1f} s | decode {dec:.1f} s | {ntok} tok | {ntok/dec:.2f} tok/s"
      f"{f' | draft acc {100*da/dn:.0f}%' if dn else ''}")
print(f"  DRAM read  {rd(c)/1e9:8.1f} GB, minus idle {net/1e9:8.1f} GB = {net/wall/1e9:6.1f} GB/s "
      f"({100*net/wall/BW:.0f}% of {a.bandwidth:.0f})")
print(f"  DRAM write {wr(c)/1e9:8.1f} GB = {wr(c)/wall/1e9:6.1f} GB/s")
print(f"  ==> {net/ntok/1e9:.2f} GB read per generated token")
print(f"  ==> those bytes alone would allow {BW/(net/ntok):.0f} tok/s; achieved {ntok/dec:.2f} "
      f"= {100*(ntok/dec)/(BW/(net/ntok)):.0f}% extraction")
