#!/bin/bash
# A/B of the GGML_CPU_FA_GQA flash-attention path on the 4-layer proxy (same mimo2 graph, loads in ~1 min).
#   variants: prod = build-dflashn (production binary); off = build-fagqa GQA=0 (must equal prod); on = build-fagqa GQA=1
set -u
cd "$(dirname "$0")/.."
PORT=18198
OUT=${OUT:-/tmp/mimo-vis/ab}
mkdir -p $OUT
stop_port() {
  local pid
  pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for i in $(seq 1 120); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 1; done
  [ -n "$pid" ] && for i in $(seq 1 180); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
}
run_variant() {
  local name=$1 build=$2 gqa=$3
  echo "=== $name ($build, GGML_CPU_FA_GQA=$gqa) $(date +%T)"
  stop_port
  GGML_CPU_FA_GQA=$gqa BUILD=$build SPEC=dflash NMAX=7 PMIN=0.0 PORT=$PORT CTX=16384 \
    MODEL=/models/mimo-v26-pro/proxy/mimo-v26-proxy-4L.gguf \
    DFLASH_MODEL=/models/mimo-v26-pro/gguf/MiMo-DFlash-proxytest.gguf \
    nohup ./launch-mimo-tp.sh > $OUT/server-$name.log 2>&1 &
  for i in $(seq 1 600); do
    curl -sf -m 2 localhost:$PORT/health >/dev/null 2>&1 && break
    kill -0 $! 2>/dev/null || { echo "server died"; tail -20 $OUT/server-$name.log; return 1; }
    sleep 1
  done
  echo "healthy after ${i}s"
  python3 golden.py $PORT $OUT/golden-$name.json 48
  python3 tools/longprobe.py $PORT $OUT/long-$name.json 400
  kill -0 $! 2>/dev/null && echo "server alive after requests" || echo "SERVER DIED"
  stop_port
}
NEW=${NEW:-build-fagqa}
run_variant prod build-dflashn 0
run_variant off  $NEW 0
run_variant l1   $NEW 1
[ "${WITH_L2:-0}" = 1 ] && run_variant l2 $NEW 2
echo "=== compare"
echo "prod vs off (must be identical):"; python3 golden.py --compare $OUT/golden-prod.json $OUT/golden-off.json; python3 golden.py --compare $OUT/long-prod.json $OUT/long-off.json
echo "off vs l1:"; python3 golden.py --compare $OUT/golden-off.json $OUT/golden-l1.json; python3 golden.py --compare $OUT/long-off.json $OUT/long-l1.json
if [ "${WITH_L2:-0}" = 1 ]; then
  echo "l1 vs l2 (prefill kernel only):"; python3 golden.py --compare $OUT/golden-l1.json $OUT/golden-l2.json; python3 golden.py --compare $OUT/long-l1.json $OUT/long-l2.json
fi
echo "=== done $(date +%T)"
