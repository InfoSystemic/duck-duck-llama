#!/usr/bin/env bash
# Per-op profile of the 8-layer GLM-5.3-Flash on one socket. The point is to
# find out WHERE the ~70% of missing bandwidth actually goes before writing any
# kernel: the GLM-5.3 Full effort only succeeded because it profiled first and
# found the loss was scheduling and mirroring, not arithmetic.
set -u
BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server
MODEL=${MODEL:-/dev/shm/glm53f-8L.gguf}
PORT=18097
OUT=/dev/shm/glm-dev/prof-8L.txt
NODE=${NODE:-2}

for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
sleep 2

( setsid env NVIDIA_TF32_OVERRIDE=0 \
    GGML_CPU_OP_PROFILE=1 GGML_CPU_OP_PROFILE_SKIP=${SKIP:-3} GGML_CPU_OP_PROFILE_COUNT=${COUNT:-24} \
    GGML_CPU_REPACK_LOAD_THREADS=16 \
    "$@" \
    numactl --cpunodebind=$NODE --membind=$NODE -- \
    "$BIN" --host 127.0.0.1 --port $PORT --model "$MODEL" --alias trunc \
      --gpu-layers 0 --load-mode none --threads ${THREADS:-32} --threads-batch ${THREADS:-32} \
      --ctx-size 4096 --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 \
      --jinja --reasoning-format deepseek --predict 4096 --temp 1.0 \
      --timeout 600 --no-webui --metrics --log-timestamps > "$OUT" 2>&1 < /dev/null & )

t0=$(date +%s)
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  pgrep -f "llama-server.*port $PORT" >/dev/null || { echo "DIED"; grep -iE 'error|assert' "$OUT" | tail -5; exit 1; }
  sleep 5
done
echo "healthy after $(( $(date +%s)-t0 ))s"

curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"trunc","messages":[{"role":"user","content":"Write one paragraph about memory bandwidth."}],"max_tokens":48,"temperature":0}' \
  | python3 -c "
import sys,json
d=json.load(sys.stdin)
t=d.get('timings',{})
print('decode %.2f tok/s over %s tokens' % (t.get('predicted_per_second',0), t.get('predicted_n',0)))" 2>/dev/null

sleep 3
for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill "$p" 2>/dev/null; done
sleep 5
echo "--- profile written to $OUT ---"
