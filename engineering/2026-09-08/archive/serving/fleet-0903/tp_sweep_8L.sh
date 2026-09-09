#!/usr/bin/env bash
# Tensor-parallel sweep of GLM-5.3-Flash on the 8-layer truncated model, now that
# the GLM fork's meta-backend scheduler (spin dispatch, fused/merged all-reduce)
# and its split-state rules are merged in. The rule that matters here is
# "mul_mat with a k-split weight against a mirrored input -> PARTIAL", which is
# exactly what blocked splitting attn_output while attention stays mirrored.
set -u
BIN=/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server
MODEL=/dev/shm/glm53f-8L.gguf
PORT=18097

run() { # $1 label, rest env
  local label=$1; shift
  for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  ( setsid env NVIDIA_TF32_OVERRIDE=0 GGML_CPU_REPACK_LOAD_THREADS=16 \
      GGML_CPU_NUMA_DEVICES=1 GGML_CPU_NUMA_THREADS=15 GGML_CPU_NUMA_POLL=100 \
      GGML_CPU_NUMA_DIRECT_ALLREDUCE=1 "$@" \
      "$BIN" --host 127.0.0.1 --port $PORT --model "$MODEL" --alias trunc \
        --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
        --split-mode tensor --tensor-split 1,1,1,1 --fit off --load-mode mmap \
        --threads 15 --threads-batch 15 \
        --ctx-size 4096 --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 \
        --predict 4096 --temp 1.0 --timeout 600 --no-webui --metrics --log-timestamps \
        > /dev/shm/glm-dev/tp8-$label.log 2>&1 < /dev/null & )
  local t0=$(date +%s)
  until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! pgrep -f "llama-server.*port $PORT" >/dev/null; then
      printf '%-24s DIED  %s\n' "$label" "$(tr -d '\000' < /dev/shm/glm-dev/tp8-$label.log | grep -aoE 'GGML_ASSERT[^\n]{0,90}|unsupported [^\n]{0,90}' | tail -1)"
      return 1
    fi
    [ $(( $(date +%s)-t0 )) -gt 300 ] && { echo "$label: TIMEOUT"; return 1; }
    sleep 5
  done
  curl -s -m 300 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"The memory bandwidth of a four socket server","n_predict":64,"temperature":0,"cache_prompt":false}' >/dev/null 2>&1
  sleep 1
  printf '%-24s load=%3ds  ' "$label" $(( $(date +%s)-t0 ))
  tr -d '\000' < /dev/shm/glm-dev/tp8-$label.log | grep -a 'eval time =' | grep -av 'prompt eval' \
    | tail -1 | awk '{printf "%s tok/s\n", $(NF-3)}' | grep . || echo "NO TIMING"
}

echo "=== GLM-5.3-Flash 8L tensor-parallel sweep $(date -u +%FT%TZ) ==="
run tp-mirrored-attn   GGML_GLM5N_ATTN_OUT_TP=0
run tp-attnout         GGML_GLM5N_ATTN_OUT_TP=1
run tp-attnout-fused   GGML_GLM5N_ATTN_OUT_TP=1 GGML_CPU_NUMA_FUSED_REDUCE=1 GGML_CPU_NUMA_DISPATCH_SPIN_US=20000 GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US=300
run tp-attnout-merged  GGML_GLM5N_ATTN_OUT_TP=1 GGML_CPU_NUMA_FUSED_REDUCE=1 GGML_CPU_NUMA_MERGE_REDUCE=1 GGML_CPU_NUMA_DISPATCH_SPIN_US=20000 GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US=300
run tp-shexp           GGML_GLM5N_ATTN_OUT_TP=1 GGML_GLM5N_SHEXP_TP=1 GGML_CPU_NUMA_FUSED_REDUCE=1 GGML_CPU_NUMA_MERGE_REDUCE=1 GGML_CPU_NUMA_DISPATCH_SPIN_US=20000
run tp-attn-headsplit  GGML_GLM5N_ATTN_TP=1 GGML_CPU_NUMA_FUSED_REDUCE=1 GGML_CPU_NUMA_MERGE_REDUCE=1 GGML_CPU_NUMA_DISPATCH_SPIN_US=20000
for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
echo "=== done $(date -u +%FT%TZ) ==="
