#!/usr/bin/env bash
# Sweep the small-op dispatch knobs on the 8-layer truncated models. The profile
# showed both Flash models spend 10-11% of a token in UNARY ops on ~10K elements
# (0.19 ms each = 0.2 GB/s) and another 8-10% in GET_ROWS on the KDA state at
# 5.8 GB/s. Those are not kernel speeds, they are 32-thread dispatch overhead on
# ops too small to be worth splitting. GGML_CPU_SINGLE_TASK_MAX_ELEMENTS runs an
# op on one thread below a size threshold; it was set for GLM-5.3 Full and never
# tried on either Flash model.
set -u
which=$1
if [ "$which" = glm ]; then
  BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server
  MODEL=/dev/shm/glm53f-8L.gguf; PORT=18097; NODE=2; EXTRA="--flash-attn on"
else
  BIN=/dev/shm/q4e-fast/build-fast/bin/llama-server
  MODEL=/dev/shm/q4e-8L.gguf;   PORT=18098; NODE=3; EXTRA="--flash-attn on"
fi

run() { # $1 label, rest env
  local label=$1; shift
  for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
  sleep 2
  ( setsid env NVIDIA_TF32_OVERRIDE=0 GGML_CPU_REPACK_LOAD_THREADS=16 LD_LIBRARY_PATH="$(dirname $BIN)" "$@" \
      numactl --cpunodebind=$NODE --membind=$NODE -- \
      "$BIN" --host 127.0.0.1 --port $PORT --model "$MODEL" --alias trunc \
        --gpu-layers 0 --load-mode none --threads ${THREADS:-32} --threads-batch ${THREADS:-32} \
        --ctx-size 4096 $EXTRA --batch-size 512 --ubatch-size 256 --parallel 1 \
        --predict 4096 --temp 1.0 --timeout 600 --no-webui --metrics --log-timestamps \
        > /dev/shm/glm-dev/knob-$which-$label.log 2>&1 < /dev/null & )
  until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    pgrep -f "llama-server.*port $PORT" >/dev/null || { echo "$label: DIED"; return 1; }
    sleep 5
  done
  # /completion avoids the chat template entirely
  local r
  r=$(curl -s -m 300 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"The memory bandwidth of a four socket server","n_predict":64,"temperature":0,"cache_prompt":false}')
  printf '%-30s ' "$label"
  # The truncated model emits gibberish, which the server's output-format parser
  # rejects with a 500 even though generation itself succeeded. Read the decode
  # rate out of the server log rather than the HTTP response.
  sleep 1
  # the log can contain non-UTF8 bytes from the model's output, so force text
  tr -d '\000' < /dev/shm/glm-dev/knob-$which-$label.log | grep -a 'eval time =' | grep -av 'prompt eval' \
    | tail -1 | awk '{printf "%s tok/s\n", $(NF-3)}' | grep . || echo "NO TIMING"
}

echo "=== $which 8-layer knob sweep $(date -u +%FT%TZ) ==="
run baseline
run single-task-32k   GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=32768
run single-task-128k  GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=131072
run st32k-moe15       GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=32768 GGML_CPU_MOE_SINGLE_TOKEN_THREADS=15
run st32k-t16         GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=32768
for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
echo "=== done $(date -u +%FT%TZ) ==="
