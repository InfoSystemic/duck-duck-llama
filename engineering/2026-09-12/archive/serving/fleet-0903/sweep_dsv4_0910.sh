#!/usr/bin/env bash
# sweep_dsv4_0910.sh - unattended config sweep for DeepSeek-V4-Flash on the tuned engine.
# Each arm: relaunch, correctness-check, then 512-token prose+code timing.
set -uo pipefail
BASE=/home/kwebb/InfoSystemic/AI-Server/serving/fleet-0903
DRAFT=/models/deepseek-v4-tuning/DSV4-Flash-DSpark-draft-DS3-IQ3S-protected.gguf
OUT=$BASE/results/dsv4-sweep-0910.tsv
: > "$OUT"
run_arm() {   # name  threads  envfile
  local name="$1" thr="$2" envf="$3"
  echo "=== ARM $name (threads=$thr env=$(basename $envf))" >&2
  THREADS_OVERRIDE=$thr ENVFILE_OVERRIDE=$envf "$BASE/launch_dsv4_arm_0910.sh" "$DRAFT" >&2 2>&1
  if ! curl -sf -m 5 http://127.0.0.1:18132/health >/dev/null 2>&1; then
    printf '%s\t%s\tFAILED_TO_START\n' "$name" "$thr" >> "$OUT"; return
  fi
  python3 - "$name" "$thr" "$OUT" <<'PY'
import json,sys,urllib.request
name,thr,out=sys.argv[1],sys.argv[2],sys.argv[3]
def ask(p,mx):
    b=dict(model='dsv4-flash',messages=[dict(role='user',content=p)],temperature=0,seed=42,
           max_tokens=mx,cache_prompt=False,stream=False)
    r=urllib.request.Request('http://127.0.0.1:18132/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    d=json.load(urllib.request.urlopen(r,timeout=1800)); m=d['choices'][0]['message']
    return (m.get('reasoning_content') or '')+(m.get('content') or ''), d.get('timings',{})
c,_=ask('What is 17 * 23? Reply with only the number.',96)
ok='391' in c
res={}
for k,p in (('prose','Explain how a refrigerator moves heat. Give a detailed explanation in plain English.'),
            ('code','Write a Python function that merges two sorted lists. Include an explanation of its time complexity.')):
    _,tm=ask(p,512); res[k]=tm.get('predicted_per_second',0)
with open(out,'a') as f:
    f.write(f"{name}\t{thr}\t{'PASS' if ok else 'FAIL'}\t{res['prose']:.2f}\t{res['code']:.2f}\n")
print(f"{name} thr={thr} {'PASS' if ok else 'FAIL'} prose={res['prose']:.2f} code={res['code']:.2f}",flush=True)
PY
}
run_arm "moe-fusion"  15 "$BASE/results/goal-0910-restore/env-knobs-noprec.txt"
run_arm "threads-12"  12 "$BASE/results/goal-0910-restore/env.txt"
run_arm "threads-16"  16 "$BASE/results/goal-0910-restore/env.txt"
echo "=== SWEEP DONE ==="; cat "$OUT"
