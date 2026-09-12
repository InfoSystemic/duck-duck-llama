#!/usr/bin/env bash
# Per-role run-queue wait for the llama-server on 18091 over N seconds (default 60).
# Roles are inferred from each thread's CPU affinity: 1 CPU = pinned worker (node k), spare-core set = main/http, other = unpinned.
N=${1:-60}
pid=$(pgrep -f 'llama-server.*--port 1809[1]' | head -1); [ -z "$pid" ] && { echo "no server"; exit 1; }
snap() { for t in /proc/$pid/task/*; do tid=$(basename $t); read a b c < $t/schedstat 2>/dev/null || continue; aff=$(taskset -pc $tid 2>/dev/null | awk '{print $NF}'); echo "$tid $aff $a $b"; done; }
snap > /tmp/ss0.txt; sleep $N; snap > /tmp/ss1.txt
python3 - <<'PY'
import collections
def load(f):
    d={}
    for l in open(f):
        p=l.split(); d[p[0]]=(p[1], int(p[2]), int(p[3]))
    return d
a=load('/tmp/ss0.txt'); b=load('/tmp/ss1.txt')
role_run=collections.Counter(); role_wait=collections.Counter(); role_n=collections.Counter()
def role(aff):
    if aff in ('15,31,47,63','15,31,47,63,79,95,111,127'): return 'main/http(spare cores)'
    if '-' in aff or ',' in aff: return 'unpinned(%s)'%aff if len(aff)<12 else 'unpinned(range)'
    c=int(aff); return 'worker node%d%s' % ((c%64)//16, ' cpu%d(dispatch)'%c if c%16==0 else '')
for tid,(aff,r1,w1) in b.items():
    if tid not in a: continue
    _,r0,w0=a[tid]; k=role(aff); role_run[k]+=r1-r0; role_wait[k]+=w1-w0; role_n[k]+=1
tot_run=sum(role_run.values()); tot_wait=sum(role_wait.values())
print(f"{'role':32s} {'threads':>7s} {'run s':>8s} {'wait s':>8s} {'wait%':>6s}")
for k in sorted(role_run, key=lambda k:-role_wait[k]):
    print(f"{k:32s} {role_n[k]:7d} {role_run[k]/1e9:8.1f} {role_wait[k]/1e9:8.2f} {100*role_wait[k]/max(role_run[k],1):6.2f}")
print(f"TOTAL run {tot_run/1e9:.0f}s wait {tot_wait/1e9:.1f}s ({100*tot_wait/max(tot_run,1):.2f}%)")
PY
