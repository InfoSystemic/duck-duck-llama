#!/usr/bin/env bash
# Launch the 8-layer dev model with a given binary + env overrides, benchmark raw decode, kill.
# Usage: ENV... devrun.sh <label> <llama-server-binary> [extra llama-server args...]
set -u
label=$1; bin=$2; shift 2
port=18095
OUT=/dev/shm/glm-dev/results; mkdir -p $OUT
DB=~/InfoSystemic/AI-Server/engines/llama-llama-duck/tools/decode_bench.py
ENVF=/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.glm53-q4.env
# production env, then overrides from the caller's environment take precedence
set -a; source $ENVF; set +a
for v in $(env | grep -E '^(GGML_|GLM_|OMP_|GOMP_|LLAMA_)' | cut -d= -f1); do :; done
export GLM_MODEL=${DEV_MODEL:-/dev/shm/glm-dev/glm53-q4kxl-8L.gguf}
export GLM_SPEC_TYPE=none
export GLM_CONTEXT_SIZE=${GLM_CONTEXT_SIZE_DEV:-8192}
pkill -f "llama-server.*--port $port" 2>/dev/null; sleep 1
log=$OUT/dev-$label.server.log
setsid nohup "$bin" --host 127.0.0.1 --port $port --model "$GLM_MODEL" --alias glm-dev \
  --load-mode mmap --fit off --ctx-size $GLM_CONTEXT_SIZE --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --batch-size 512 --ubatch-size 256 --parallel 1 --cache-prompt --cache-reuse 256 --jinja \
  --chat-template-file "$GLM_CHAT_TEMPLATE_FILE" --reasoning-format deepseek --reasoning-preserve \
  --predict 4096 --temp 1.0 --top-p 0.95 --top-k 20 --timeout 3600 --threads-http 4 --no-webui --metrics --log-timestamps --verbosity 2 \
  --threads "$GLM_THREADS" --threads-batch "$GLM_THREADS" \
  --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split "${DEV_TENSOR_SPLIT:-1,1,1,1}" "$@" \
  > "$log" 2>&1 < /dev/null &
spid=$!
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1; then break; fi
  if ! kill -0 $spid 2>/dev/null; then echo "SERVER DIED"; tail -20 "$log"; exit 1; fi
  sleep 2
done
curl -sf http://127.0.0.1:$port/health >/dev/null || { echo "TIMEOUT waiting for server"; tail -5 "$log"; pkill -f "llama-server.*--port $port"; exit 1; }
echo "[$label] up after ~$((i*2))s; RssAnon=$(grep RssAnon /proc/$spid/status | awk '{print int($2/1048576)"G"}')"
# warmup
curl -s http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm-dev","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"cache_prompt":false}' >/dev/null
python3 $DB --port $port --model glm-dev --label $label --workloads prose,code,novel --reps ${DEV_REPS:-2} --max-tokens ${DEV_TOKENS:-96} --reasoning-effort low --spec-n-max 0 --out $OUT/dev-$label.json 2>&1 | grep -E "SUMMARY|tok/s"
# fingerprint of output text for correctness comparison (temp 0)
python3 - $port $label <<'PY'
import sys,json,urllib.request,hashlib
port=sys.argv[1]
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps({"model":"glm-dev","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":24,"temperature":0,"cache_prompt":False,"reasoning_effort":"low"}).encode(),headers={"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(req,timeout=600)); c=d["choices"][0]["message"].get("content") or ""; r=d["choices"][0]["message"].get("reasoning_content") or ""
print("fingerprint:", hashlib.sha256((r+"|"+c).encode()).hexdigest()[:16], repr((r+c)[:60]))
# numeric probe: top-5 probabilities of the first generated token for a fixed prompt
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=json.dumps({"prompt":"The capital of France is","n_predict":1,"temperature":0,"n_probs":10,"cache_prompt":False}).encode(),headers={"Content-Type":"application/json"})
raw=urllib.request.urlopen(req,timeout=600).read(); d=json.loads(raw)
open(f"/dev/shm/glm-dev/results/probe-{sys.argv[2] if len(sys.argv)>2 else 'x'}.json","wb").write(raw)
cp=d.get("completion_probabilities") or []
if cp and isinstance(cp[0],dict):
    tp=cp[0].get("top_logprobs") or cp[0].get("top_probs") or []
    print("top5:", [(t.get("token"), round(t.get("logprob", t.get("prob",0)),4)) for t in tp])
else:
    print("probe-keys:", list(d.keys())[:12], str(d)[:300])
PY
pkill -f "llama-server.*--port $port"; sleep 2
echo "[$label] done"
