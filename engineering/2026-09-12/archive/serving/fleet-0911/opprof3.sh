#!/bin/bash
# Op profile + UNCONDITIONAL production restore. Box is quiet (peer down), which is the
# condition the earlier attempt lacked -- it was OOM-killed with a 119 GB second model resident.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
echo "=== per-node free before ==="; numactl -H | grep free
$D/launch.sh "GGML_CPU_OP_PROFILE=* GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/bwprobe/opprof.arm GGML_CPU_OP_PROFILE_COUNT=4 GGML_CPU_OP_PROFILE_SKIP=2" > $D/results/launch-opprof3.log 2>&1 || { echo LAUNCH_FAIL; tail -4 $D/results/launch-opprof3.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
LOG=$(ls -t $D/results/server-*.log | head -1); MARK=$(wc -l < "$LOG")
touch /dev/shm/bwprobe/opprof.arm
curl -s -m 200 http://127.0.0.1:18131/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Count from one to thirty in words.","n_predict":32,"temperature":0,"cache_prompt":false}' >/dev/null
rm -f /dev/shm/bwprobe/opprof.arm
tail -n +$((MARK+1)) "$LOG" | grep 'CPU_OP_PROFILE' > $D/results/opprof.txt
echo "captured $(wc -l < $D/results/opprof.txt) profile lines"
python3 - $D/results/opprof.txt <<'PY'
import sys,re,collections
lines=open(sys.argv[1]).read().split('\n')
g=[l for l in lines if ' nodes=' in l]
print(f"graphs profiled: {len(g)}")
for x in g[:6]:
    m=re.search(r'index=(\d+) cpu=(\d+).*nodes=(\d+) total=([\d.]+) ms',x)
    if m: print(f"   idx={m.group(1)} cpu={m.group(2)} NODES={m.group(3)} in-op total={m.group(4)} ms")
ops=collections.Counter(); tim=collections.Counter()
for l in lines:
    m=re.search(r"op=(\w+) time=([\d.]+) ms",l)
    if m: ops[m.group(1)]+=1; tim[m.group(1)]+=float(m.group(2))
if tim:
    tot=sum(tim.values())
    print(f"\ntimed ops (>=5us only): {sum(ops.values())}  total {tot:.1f} ms")
    for op,t in tim.most_common(16):
        print(f"   {op:24s} {t:8.2f} ms {100*t/tot:5.1f}%  n={ops[op]:5d}  avg {1000*t/ops[op]:6.1f} us")
PY
echo "=== OPPROF DONE ==="
