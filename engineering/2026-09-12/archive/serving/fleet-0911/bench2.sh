#!/bin/bash
# bench.sh <port> <tag> [n_predict]  -- 3 fixed prompts, tok/s + IMC GB/s per prompt
PORT=${1:-18131}; TAG=${2:-run}; N=${3:-192}
D=~/InfoSystemic/AI-Server/serving/fleet-0911/results; mkdir -p $D
P1="Write a detailed technical explanation of how a modern out-of-order CPU core executes instructions, covering fetch, decode, rename, scheduling, execution and retirement."
P2="Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments."
P3="Analyse the causes of the 1997 Asian financial crisis, covering capital account liberalisation, currency pegs, short-term external debt and the IMF response."
TOT_TOK=0; TOT_T=0; SUM=""
for i in 1 2 3; do
  eval "PR=\$P$i"
  DUR=$(python3 -c "print(max(6,int($N/7)+5))")
  OUT=$D/bw-$TAG-$i.csv
  /dev/shm/bwprobe/bw.sh 10 $DUR $OUT & SP=$!
  sleep 0.7
  R=$(curl -s -m 240 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$PR"),\"n_predict\":$N,\"temperature\":0,\"seed\":42,\"cache_prompt\":false,\"stream\":false}")
  wait $SP
  echo "$R" > $D/resp-$TAG-$i.json
  LINE=$(python3 - "$D/resp-$TAG-$i.json" "$OUT" <<'PY'
import json,sys,csv,collections,statistics
try: d=json.load(open(sys.argv[1]))
except Exception: print("0\t0\t0\t0\t0"); raise SystemExit
t=d.get('timings',{})
tps=t.get('predicted_per_second',0); n=t.get('predicted_n',0)
dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
b=collections.defaultdict(float)
for line in open(sys.argv[2]):
    line=line.strip()
    if not line or line.startswith('#'): continue
    r=next(csv.reader([line]))
    if len(r)<4: continue
    try: tt=float(r[0]); v=float(r[1])
    except ValueError: continue
    b[round(tt,4)]+=v*{'MiB':2**20,'KiB':1024,'GiB':2**30,'':1}.get(r[2].strip(),1)
ts=sorted(b); dt=statistics.median([ts[i+1]-ts[i] for i in range(len(ts)-1)])
gb=[b[x]/dt/1e9 for x in ts][1:-1]
# robust: LONGEST CONTIGUOUS run above threshold (small gaps ok). A first-to-last window is
# corrupted by isolated spikes in the idle tail.
thr=50.0; gap=max(1,int(0.25/dt)); best=(0,-1); i=0; n=len(gb)
while i<n:
    if gb[i]<=thr: i+=1; continue
    j=i; last=i
    while j<n:
        if gb[j]>thr: last=j
        elif j-last>gap: break
        j+=1
    if last-i > best[1]-best[0]: best=(i,last)
    i=max(j,i+1)
w=gb[best[0]:best[1]+1] or [0]
print(f"{n}\t{tps:.2f}\t{statistics.mean(w):.1f}\t{statistics.median(w):.1f}\t{100*da/dn if dn else 0:.0f}")
PY
)
  read NN TPS MGB MDGB ACC <<< "$LINE"
  [ -z "${TPS:-}" ] && { NN=0; TPS=0; MGB=0; MDGB=0; ACC=0; }
  printf "  p%d: %4s tok  %6s tok/s  %6s GB/s mean  %6s median  draft %s%%\n" $i $NN $TPS $MGB $MDGB $ACC
  TOT_TOK=$(python3 -c "print($TOT_TOK+$NN)"); SUM="$SUM $TPS:$MGB"
done
python3 - "$TAG" $SUM <<'PY'
import sys, statistics
tag=sys.argv[1]; pairs=[p.split(':') for p in sys.argv[2:]]
tps=[float(a) for a,_ in pairs]; gb=[float(b) for _,b in pairs]
print(f"  == {tag}: mean {statistics.mean(tps):.2f} tok/s   {statistics.mean(gb):.1f} GB/s ({100*statistics.mean(gb)/380:.1f}% of 380)")
PY
