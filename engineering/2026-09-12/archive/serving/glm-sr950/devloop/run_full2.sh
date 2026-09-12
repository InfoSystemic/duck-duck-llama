#!/usr/bin/env bash
# Launch the FULL model with a given binary + env overrides on port 18091, benchmark, leave running.
# Usage: ENV... run_full.sh <label> <binary>
set -u
label=$1; bin=$2
port=18091
OUT=/dev/shm/glm-dev/results; mkdir -p $OUT
DB=~/InfoSystemic/AI-Server/engines/llama-llama-duck/tools/decode_bench.py
ENVF=${FULL_ENVF:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.glm53-q4.env}
set -a; source $ENVF; set +a
export GLM_BINARY="$bin"
# whole-token per-op profile (CPU side; does not disable the fused reduce). Armed by touching the file.
export GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=/tmp/glm-op-full.arm GGML_CPU_OP_PROFILE_COUNT=${FULL_PROFILE_COUNT:-16}
rm -f /tmp/glm-op-full.arm
pkill -f "llama-server.*--port $port" 2>/dev/null; sleep 3
log=$OUT/full-$label.server.log
cd ~/InfoSystemic/AI-Server/serving/glm-sr950
GLM_SR950_CONFIG=$ENVF setsid nohup ${FULL_TASKSET:+taskset -c $FULL_TASKSET} ./launch-glm-sr950.sh $port > "$log" 2>&1 < /dev/null &
sleep 5; spid=$(pgrep -f "llama-server.*--port $port" | head -1)
echo "[$label] launched pid $spid $(date +%T)"
for i in $(seq 1 400); do
  if curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1; then break; fi
  if ! pgrep -f "llama-server.*--port $port" >/dev/null; then echo "SERVER DIED"; tail -30 "$log"; exit 1; fi
  sleep 5
done
curl -sf http://127.0.0.1:$port/health >/dev/null || { echo "TIMEOUT"; exit 1; }
echo "[$label] up $(date +%T) after ~$((i*5))s; RssAnon=$(grep RssAnon /proc/$(pgrep -f "llama-server.*--port $port" | head -1)/status | awk '{print int($2/1048576)"G"}')"
curl -s http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm-sr950","messages":[{"role":"user","content":"What is 17 * 23? Answer with the number only."}],"max_tokens":40,"temperature":0,"cache_prompt":false,"speculative.n_max":0,"reasoning_effort":"low"}' | python3 -c 'import sys,json; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("sanity:", repr((m.get("reasoning_content") or "")[:80]), "|", repr((m.get("content") or "")[:80]), "| tok/s", round(d.get("timings",{}).get("predicted_per_second",0),2))'
echo "== profiled raw token =="
python3 - "http://127.0.0.1:$port/v1/chat/completions" <<'PY'
import sys, json, urllib.request
url = sys.argv[1]
payload = {"model":"glm-sr950","messages":[{"role":"user","content":"Write a compact technical explanation of why NUMA-local memory placement matters for CPU language-model decoding. Use several paragraphs."}],
           "max_tokens":24,"temperature":0,"seed":1,"cache_prompt":False,"stream":True,"speculative.n_max":0,"reasoning_effort":"low"}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
n = 0; armed = False
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            n += 1
            if n == 4 and not armed:
                open("/tmp/glm-op-full.arm","w").close(); armed = True
print("profiled request events", n, "armed", armed)
PY
sleep 2; rm -f /tmp/glm-op-full.arm
python3 $DB --port $port --model glm-sr950 --label ${label}-raw --workloads prose,code,structured,novel --reps 2 --max-tokens 160 --reasoning-effort low --spec-n-max 0 --out $OUT/full-${label}-raw.json 2>&1 | grep -E "SUMMARY|tok/s"
python3 $DB --port $port --model glm-sr950 --label ${label}-n2p0 --workloads prose,code,structured,novel --reps 2 --max-tokens 160 --reasoning-effort low --spec-n-max 2 --spec-p-min 0 --out $OUT/full-${label}-n2p0.json 2>&1 | grep -E "SUMMARY|tok/s"
echo "[$label] bench done $(date +%T)"
