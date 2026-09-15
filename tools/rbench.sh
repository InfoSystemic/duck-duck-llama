#!/bin/bash
# rbench.sh <port> <tag> <reps> [n_predict] -- repeat bench3 and report the DISTRIBUTION, not one sample.
# Written 09-14 after four hard stalls and an 8-slot outlier that produced a wrong conclusion: a single run of a config on
# this box carries ~13% spread on prefill and can stall outright. Any claim about a config should quote median and spread.
set -u
cd "$(dirname "$0")"; P=${1:?port}; TAG=${2:?tag}; R=${3:-5}; N=${4:-192}; D=results; TS=$(date +%Y%m%d-%H%M%S)
for i in $(seq 1 $R); do ./bench3.sh $P "$TAG-r$i-$TS" $N > $D/rbench-$TAG-r$i-$TS.txt 2>&1; done
python3 - "$TAG" $D/rbench-$TAG-r*-$TS.txt <<'PY'
import sys, re, statistics
tag = sys.argv[1]; per = {}
for f in sys.argv[2:]:
    for l in open(f):
        m = re.search(r'p(\d): (\d+) tok\s+([\d.]+) tok/s\s+pp ([\d.]+)', l)
        if m: per.setdefault(m.group(1), []).append((float(m.group(3)), float(m.group(4))))
print(f"  {tag}: {len(sys.argv)-2} repeats")
for p in sorted(per):
    tg = [x[0] for x in per[p]]; pp = [x[1] for x in per[p]]
    med = statistics.median(tg); spread = 100*(max(tg)-min(tg))/statistics.mean(tg)
    flag = "  <-- UNSTABLE" if spread > 10 else ""
    print(f"    p{p}: decode median {med:6.2f}  min {min(tg):6.2f}  max {max(tg):6.2f}  spread {spread:5.1f}%"
          f" | prefill median {statistics.median(pp):6.1f}{flag}")
allv = [x[0] for v in per.values() for x in v]
print(f"    overall: median {statistics.median(allv):.2f} tok/s, min {min(allv):.2f}, "
      f"spread {100*(max(allv)-min(allv))/statistics.mean(allv):.1f}%")
PY
