#!/bin/bash
# benchpar.sh <port> <tag> <concurrency> [n_predict] -- N concurrent streams, aggregate tok/s + IMC GB/s
PORT=${1:-18131}; TAG=${2:-par}; C=${3:-4}; N=${4:-192}
D=~/InfoSystemic/AI-Server/serving/fleet-0911/results; mkdir -p $D
OUT=$D/bw-$TAG.csv
DUR=$(python3 -c "print(max(8,int($N/6)+6))")
/dev/shm/bwprobe/bw.sh 10 $DUR $OUT & SP=$!
sleep 0.7
PROMPTS=(
 "Write a detailed technical explanation of how a modern out-of-order CPU core executes instructions."
 "Implement a thread-safe LRU cache in C++ with O(1) get and put, with full commented code."
 "Analyse the causes of the 1997 Asian financial crisis and the IMF response."
 "Explain how DDR4 memory controllers schedule reads and writes across bank groups."
 "Describe the design of a distributed consensus protocol, covering leader election and log replication."
 "Explain the mathematics of singular value decomposition and its use in low-rank approximation."
 "Write a complete tutorial on writing a bytecode virtual machine in C."
 "Discuss the history and design trade-offs of copy-on-write filesystems."
)
T0=$(date +%s.%N)
for i in $(seq 0 $((C-1))); do
  P="${PROMPTS[$((i % 8))]} (variant $i)"
  curl -s -m 300 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$P"),\"n_predict\":$N,\"temperature\":0,\"seed\":42,\"cache_prompt\":false,\"stream\":false}" \
    > $D/par-$TAG-$i.json &
done
wait $(jobs -p | grep -v "^$SP$" 2>/dev/null) 2>/dev/null || true
T1=$(date +%s.%N)
wait $SP 2>/dev/null
python3 - "$D" "$TAG" "$C" "$T0" "$T1" "$OUT" <<'PY'
import json,sys,glob,csv,collections,statistics
D,TAG,C,T0,T1,BW=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),sys.argv[6]
tot=0; per=[]
for i in range(C):
    try:
        d=json.load(open(f"{D}/par-{TAG}-{i}.json")); t=d.get('timings',{})
        tot+=t.get('predicted_n',0); per.append(t.get('predicted_per_second',0))
    except Exception: pass
wall=T1-T0
b=collections.defaultdict(float)
for line in open(BW):
    line=line.strip()
    if not line or line.startswith('#'): continue
    r=next(csv.reader([line]))
    if len(r)<4: continue
    try: tt=float(r[0]); v=float(r[1])
    except ValueError: continue
    b[round(tt,4)]+=v*{'MiB':2**20,'KiB':1024,'GiB':2**30,'':1}.get(r[2].strip(),1)
ts=sorted(b); dt=statistics.median([ts[i+1]-ts[i] for i in range(len(ts)-1)])
gb=[b[x]/dt/1e9 for x in ts][1:-1]
idx=[i for i,g in enumerate(gb) if g>50]
w=gb[idx[0]:idx[-1]+1] if idx else [0]
m=statistics.mean(w)
print(f"  == {TAG} C={C}: aggregate {tot/wall:.2f} tok/s ({tot} tok / {wall:.1f}s)  per-stream {statistics.mean(per) if per else 0:.2f}")
print(f"     {m:.1f} GB/s mean ({100*m/380:.1f}% of 380)  median {statistics.median(w):.1f}")
PY
