#!/bin/bash
# optrace.sh <port> <log> <tag> [prompt-kind] -- arm the CPU op profiler during a warm decode, extract the new trace lines
PORT=$1; LOG=$2; TAG=$3; KIND=${4:-mid}; ARM=${ARM:-/dev/shm/f18-optrace.arm}
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
python3 - "$PORT" "$KIND" "$ARM" <<'PY'
import json,sys,urllib.request,threading,time,os
from pathlib import Path
port,kind,arm=sys.argv[1:4]
src=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/README.md').read_text()
p={'short':'Implement a thread-safe LRU cache in C++ with O(1) get and put.','mid':src[:12000]+'\n\nSummarise the key findings above in five bullet points.\n','long':src[:40000]+'\n\nSummarise the key findings above in five bullet points.\n'}[kind]
def req(n):
    body=dict(prompt=p,n_predict=n,temperature=0,seed=42,cache_prompt=True,stream=False)
    return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=3600))
req(4)  # warm the prompt cache
def armer():
    time.sleep(0.6); open(arm,'w').close()
t=threading.Thread(target=armer); t.start()
r=req(64); t.join(); time.sleep(0.5)
try: os.unlink(arm)
except FileNotFoundError: pass
print('tg %.2f tok/s'%r['timings']['predicted_per_second'])
PY
grep 'CPU_OP_PROFILE' "$LOG" > $W/results/optrace-$TAG.log
grep -c 'CPU_OP_PROFILE index' $W/results/optrace-$TAG.log
