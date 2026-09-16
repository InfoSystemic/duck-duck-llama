#!/bin/bash
# Prefill a long context ONCE and save the slot to disk, so later experiments restore it in seconds.
#
# Why: a 250K prefill runs about two hours on this box, and every experiment at that length pays it again. The
# server supports /slots/{id}?action=save|restore when started with --slot-save-path (common/arg.cpp:3643,
# server-context.cpp:4746-4766), so the KV and indexer state can be written once and reloaded.
#
# Caveat that matters: a saved slot is tied to the cache geometry it was made with. A different --kv-type, context
# size or model needs its own snapshot. Snapshots are named accordingly so they cannot be mixed up.
#
# Usage: ctx-snapshot.sh <n_ctx_tokens> <tag> [extra launcher args...]
set -u
cd "$(dirname "$0")"; D=results; TS=$(date +%Y%m%d-%H%M%S)
N=${1:?tokens}; TAG=${2:?tag}; shift 2
SNAP=/models/qwen-slots/   # TRAILING SLASH REQUIRED: the server builds the path as
                          # params.slot_save_path + filename, a plain concatenation with no separator
                          # (server-context.cpp:5271). Without it the file lands as /models/qwen-slots<name>.
exec > >(tee -a $D/ctx-snapshot-$TAG-$TS.log) 2>&1
echo "=== snapshot $TAG: prefilling $N tokens, then saving the slot $(date -Is)"
./stop-by-port.sh 18083
for i in $(seq 1 90); do ss -ltn | grep -q ':18083 ' || break; sleep 2; done
nohup python3 launch-qwen-native.py --port 18083 --slot-save-path $SNAP "$@" > $D/ctx-snapshot-$TAG-$TS.server.log 2>&1 &
for i in $(seq 1 500); do curl -s -m 2 http://127.0.0.1:18083/health | grep -q '"ok"' && break; sleep 5; done
curl -s -m 2 http://127.0.0.1:18083/health | grep -q '"ok"' || { echo "server never came up"; exit 1; }
./quiet.sh >/dev/null 2>&1 || true
python3 - "$N" "$TAG" <<'PY'
import json,sys,time,urllib.request,random
n=int(sys.argv[1]); tag=sys.argv[2]
W=("the of and to in a is that it for was as with be by on not he this are or his from at which but have an they one "
   "you had we all her she there their when who will more no if out so said what up its about into than them can only").split()
r=random.Random(1234)
p=" ".join(r.choice(W) for _ in range(int(n*0.95)))
open(f"/dev/shm/ctx-snapshot-{tag}.txt","w").write(p)
b=json.dumps({"prompt":p,"n_predict":1,"temperature":0,"cache_prompt":True,"stream":False}).encode()
t0=time.time()
d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18083/completion",data=b,
    headers={"Content-Type":"application/json"}),timeout=20000))
print(f"  prefilled {d['timings']['prompt_n']:,} tokens in {time.time()-t0:.0f}s "
      f"({d['timings']['prompt_per_second']:.1f} tok/s)", flush=True)
PY
# measure decode at this context BEFORE saving, so the expensive prefill yields the number as well as the artifact
echo "  measuring decode at the prefilled context"
python3 - "$TAG" <<'PY2'
import json,sys,urllib.request
tag=sys.argv[1]
p=open(f"/dev/shm/ctx-snapshot-{tag}.txt").read()
b=json.dumps({"prompt":p,"n_predict":96,"temperature":0,"cache_prompt":True,"stream":False,"ignore_eos":True}).encode()
d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18083/completion",data=b,
    headers={"Content-Type":"application/json"}),timeout=20000))
t=d["timings"]; dn,da=t.get("draft_n",0),t.get("draft_n_accepted",0)
print(f"    ctx {t['prompt_n']:,} | decode {t['predicted_per_second']:.2f} tok/s | "
      f"{1000/t['predicted_per_second']:.1f} ms/token | draft acc {100*da/dn if dn else 0:.0f}% | "
      f"tok/cycle {t['predicted_n']/(dn/4.0) if dn else 0:.2f}", flush=True)
PY2
echo "  saving slot 0 -> $SNAP$TAG.bin"
curl -s -m 1800 -X POST "http://127.0.0.1:18083/slots/0?action=save" -H 'Content-Type: application/json' \
     -d "{\"filename\":\"$TAG.bin\"}" | head -c 400; echo
ls -la $SNAP$TAG.bin 2>/dev/null | awk '{printf "  saved %.1f GB\n", $5/1e9}'
if [ -s "$SNAP$TAG.bin" ]; then
  echo "  VERIFIED on disk: $(stat -c %s "$SNAP$TAG.bin") bytes"
else
  echo "  !! NO FILE at $SNAP$TAG.bin -- the save did not land; do not rely on this snapshot"
fi
./stop-by-port.sh 18083
echo "=== done $(date -Is). Restore with: curl -X POST :18083/slots/0?action=restore -d '{\"filename\":\"$TAG.bin\"}'"
