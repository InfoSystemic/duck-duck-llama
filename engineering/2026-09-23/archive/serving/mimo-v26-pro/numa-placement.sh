#!/usr/bin/env bash
# numa-placement.sh [pid] -- is a tensor-parallel server's weight memory actually spread across the NUMA nodes?
#
# The failure this catches: a model whose pages all sit on one node while the other sockets read them across UPI. Decode is
# bandwidth-bound, so that caps throughput at one node's share of the memory system no matter how many threads run. On this
# host a single node is ~95 GB/s of the 381.6 GB/s aggregate, so the symptom is a rate about a quarter of what the model's
# bytes-per-token allow. Intel PCM shows it directly as one socket's READ column high and the others near zero with large UPI
# traffic; this reads the same fact out of /proc without needing root.
#
# Reports anonymous pages per node (the node-bound weight buffers the meta backend allocates) and the per-node share. A
# healthy four-way tensor split is roughly 25% on each node.
set -euo pipefail
PID=${1:-$(pgrep -f "build/bin/llama-server" | head -1)}
[[ -n "${PID:-}" && -d /proc/$PID ]] || { echo "no llama-server pid found"; exit 1; }

printf 'pid %s  %s\n' "$PID" "$(tr '\0' ' ' < /proc/$PID/cmdline | grep -o '\-\-model [^ ]*' | head -1)"
printf 'rss %.1f GiB\n\n' "$(awk '/^VmRSS/ {print $2/1048576}' /proc/$PID/status)"

python3 - "$PID" <<'PY'
import re, sys
from collections import defaultdict
pid = sys.argv[1]
per = defaultdict(int)
huge = defaultdict(int)
try:
    for line in open(f'/proc/{pid}/numa_maps'):
        for k, v in re.findall(r'\bN(\d+)=(\d+)\b', line):
            per[int(k)] += int(v)
        for k, v in re.findall(r'\bhuge N(\d+)=(\d+)\b', line):
            huge[int(k)] += int(v)
except FileNotFoundError:
    sys.exit('numa_maps unreadable (process gone?)')
tot = sum(per.values())
if not tot:
    sys.exit('no NUMA-attributed pages yet')
print(f'{"node":>5} {"GiB":>9} {"share":>7}')
for n in sorted(per):
    gib = per[n] * 4096 / 2**30
    print(f'{n:>5} {gib:>9.1f} {100*per[n]/tot:>6.1f}%')
print(f'{"total":>5} {tot*4096/2**30:>9.1f}')
shares = [per[n] / tot for n in sorted(per)]
worst, best = max(shares), min(shares)
spread = worst - best
even = 1.0 / len(shares)
# a load still in progress fills nodes in order and looks skewed; say so rather than reporting a verdict
# a 503 "Loading model" makes urlopen raise, so read the body off the exception too
loading = False
try:
    import urllib.request, urllib.error
    try:
        body = urllib.request.urlopen('http://127.0.0.1:18190/health', timeout=2).read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
    loading = 'Loading model' in body
except Exception:
    pass
if loading:
    print('\nthe server is still loading; nodes fill in order, so re-run this after /health returns ok')
elif len(shares) > 1 and worst > 0.6:
    print(f'\nSKEWED: one node holds {100*worst:.0f}% of the pages. A tensor-parallel server should be near '
          f'{100/len(shares):.0f}% per node; the other sockets are reading this across UPI.')
    print('Check: --device lists one CPU-NUMA device per socket, --split-mode tensor, --tensor-split has one weight per '
          'device, and GGML_CPU_NUMA_DEVICES=1 is set. For a SINGLE-socket run the opposite applies: numactl --membind with '
          '--load-mode none, because mmap pages come from the page cache and ignore membind.')
elif len(shares) > 1 and spread > 0.15:
    print(f'\nUNEVEN: shares span {100*spread:.0f} points around an even {100*even:.0f}%. Not necessarily fatal -- mirrored '
          f'tensors and the KV cache are not split -- but it caps the aggregate bandwidth below the sum of the nodes.')
else:
    print(f'\nbalanced: shares span {100*spread:.0f} points around an even {100*even:.0f}%; placement is not the limiter')
PY
