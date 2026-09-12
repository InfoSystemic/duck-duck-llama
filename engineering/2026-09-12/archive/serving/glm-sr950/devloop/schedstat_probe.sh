#!/usr/bin/env bash
# Sample run_delay (ns spent waiting on a runqueue) across all llama-server threads while the v3 bench runs.
until grep -q "\[v4-fast\] up" /dev/shm/glm-dev/chain14.log 2>/dev/null; do sleep 10; done
pid=$(pgrep -f 'llama-server.*--port 1809[1]' | head -1)
[ -z "$pid" ] && { echo "no server"; exit 1; }
sample() { local rd=0 rt=0; for f in /proc/$pid/task/*/schedstat; do read a b c < $f 2>/dev/null || continue; rt=$((rt + a)); rd=$((rd + b)); done; echo "$(date +%s) run_ns=$rt wait_ns=$rd"; }
echo "pid $pid threads $(ls /proc/$pid/task | wc -l)"
s0=$(sample); echo "start $s0"
sleep 600
s1=$(sample); echo "end   $s1"
python3 - "$s0" "$s1" <<'PY'
import sys
def parse(s):
    p=s.split(); return int(p[0]), int(p[1].split('=')[1]), int(p[2].split('=')[1])
t0,r0,w0=parse(sys.argv[1]); t1,r1,w1=parse(sys.argv[2])
el=t1-t0; run=(r1-r0)/1e9; wait=(w1-w0)/1e9
print(f"elapsed {el}s  cpu-run {run:.0f}s ({run/el:.1f} cores)  runqueue-wait {wait:.1f}s ({wait/el:.2f} core-equivalents, {100*wait/max(run,1):.1f}% of run time)")
PY
