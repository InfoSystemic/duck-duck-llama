#!/bin/bash
# Op-level profile of PREFILL on the 4-layer proxy (same mimo2 graph and kernels as the full model).
# The fork's profiler is armed by a file, so it records only the graphs of the probe prompt, not load/warmup.
set -u
cd "$(dirname "$0")/.."
PORT=18198
OUT=/tmp/mimo-vis/prof
BUILD=${BUILD:-build-fagqa}
GQA=${GQA:-1}
N_LINES=${N_LINES:-110}      # ~2.2K tokens
mkdir -p $OUT
ARM=$OUT/arm
rm -f $ARM
stop_port() {
  local pid
  pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for i in $(seq 1 120); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 1; done
  [ -n "$pid" ] && for i in $(seq 1 180); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
}
stop_port
LOG=$OUT/server-$BUILD-gqa$GQA.log
GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=$ARM GGML_CPU_OP_PROFILE_COUNT=${COUNT:-16} \
  GGML_CPU_FA_GQA=$GQA BUILD=$BUILD SPEC=none PORT=$PORT CTX=16384 \
  MODEL=/models/mimo-v26-pro/proxy/mimo-v26-proxy-4L.gguf \
  nohup ./launch-mimo-tp.sh > $LOG 2>&1 &
for i in $(seq 1 600); do curl -sf -m 2 localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
echo "healthy after ${i}s"
# warm request (unprofiled) so first-touch effects are out of the way
python3 tools/longprobe.py $PORT $OUT/warm.json 40 >/dev/null
touch $ARM
python3 tools/longprobe.py $PORT $OUT/probe.json $N_LINES
rm -f $ARM
sleep 2
grep -c 'CPU_OP_PROFILE index=' $LOG
stop_port
echo "=== done"
