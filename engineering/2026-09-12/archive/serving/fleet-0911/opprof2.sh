#!/bin/bash
# Launch with the in-engine op profiler armed, capture 4 graphs, summarise node counts.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
$D/launch.sh "GGML_CPU_OP_PROFILE=* GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/bwprobe/opprof.arm GGML_CPU_OP_PROFILE_COUNT=4 GGML_CPU_OP_PROFILE_SKIP=2" > $D/results/launch-opprof.log 2>&1 || { echo LAUNCH_FAIL; tail -4 $D/results/launch-opprof.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
LOG=$(ls -t $D/results/server-*.log | head -1); MARK=$(wc -l < "$LOG")
touch /dev/shm/bwprobe/opprof.arm
curl -s -m 200 http://127.0.0.1:18131/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Count from one to thirty in words.","n_predict":32,"temperature":0,"cache_prompt":false}' >/dev/null
rm -f /dev/shm/bwprobe/opprof.arm
tail -n +$((MARK+1)) "$LOG" | grep 'CPU_OP_PROFILE' > $D/results/opprof.txt
echo "captured $(wc -l < $D/results/opprof.txt) lines"
python3 - $D/results/opprof.txt <<'PY'
import sys,re,collections
lines=open(sys.argv[1]).read().split('\n')
g=[l for l in lines if ' nodes=' in l]
print(f"graphs: {len(g)}")
for x in g[:8]:
    m=re.search(r'index=(\d+) cpu=(\d+).*nodes=(\d+) total=([\d.]+) ms',x)
    if m: print(f"   idx={m.group(1)} cpu={m.group(2)} nodes={m.group(3)} in-op total={m.group(4)} ms")
ops=collections.Counter(); tim=collections.Counter()
for l in lines:
    m=re.search(r"op=(\w+) time=([\d.]+) ms",l)
    if m: ops[m.group(1)]+=1; tim[m.group(1)]+=float(m.group(2))
if tim:
    tot=sum(tim.values())
    print(f"timed ops (>=5us): {sum(ops.values())}  total {tot:.1f} ms")
    for op,t in tim.most_common(14): print(f"   {op:22s} {t:8.2f} ms {100*t/tot:5.1f}%  n={ops[op]}")
PY
