#!/bin/bash
# opprofile.sh <tag>  -- arm the in-engine op profiler for a few graphs and summarise
ARM=/dev/shm/bwprobe/opprof.arm
LOG=$(ls -t ~/InfoSystemic/AI-Server/serving/fleet-0911/results/server-*.log | head -1)
MARK=$(wc -l < "$LOG")
touch $ARM
curl -s -m 150 http://127.0.0.1:18131/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Count from one to twenty in words.","n_predict":24,"temperature":0,"cache_prompt":false}' >/dev/null
rm -f $ARM
tail -n +$((MARK+1)) "$LOG" | grep 'CPU_OP_PROFILE' > /dev/shm/bwprobe/opprof-$1.txt
echo "captured $(wc -l < /dev/shm/bwprobe/opprof-$1.txt) profile lines"
python3 - /dev/shm/bwprobe/opprof-$1.txt <<'PY'
import sys,re,collections
lines=open(sys.argv[1]).read().split('\n')
graphs=[l for l in lines if ' nodes=' in l]
print(f"graphs profiled: {len(graphs)}")
for g in graphs[:6]:
    m=re.search(r'index=(\d+) cpu=(\d+).*nodes=(\d+) total=([\d.]+) ms',g)
    if m: print(f"  graph idx={m.group(1)} cpu={m.group(2)} nodes={m.group(3)} in-op total={m.group(4)} ms")
ops=collections.Counter(); tim=collections.Counter()
for l in lines:
    m=re.search(r"op=(\w+) time=([\d.]+) ms",l)
    if m: ops[m.group(1)]+=1; tim[m.group(1)]+=float(m.group(2))
if tim:
    tot=sum(tim.values())
    print(f"  measured op time total {tot:.1f} ms across {sum(ops.values())} timed ops (>=5us only)")
    for op,t in tim.most_common(12):
        print(f"    {op:22s} {t:8.1f} ms {100*t/tot:5.1f}%  n={ops[op]}")
PY
