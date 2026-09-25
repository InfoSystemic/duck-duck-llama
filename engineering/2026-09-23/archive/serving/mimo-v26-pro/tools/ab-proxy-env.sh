#!/bin/bash
# ab-proxy-env.sh NAME1 "ENV1" NAME2 "ENV2" ... -- run the proxy golden + long probe once per variant (env strings are
# `export`ed before launch, e.g. "BUILD=build-faexp GGML_CPU_FA_GQA=2 GGML_CPU_X16_GEMM=1"), then compare consecutive pairs.
set -u
cd "$(dirname "$0")/.."
PORT=18198
OUT=${OUT:-/tmp/mimo-vis/abenv}
SPEC_V=${SPEC_V:-dflash}
mkdir -p $OUT
stop_port() {
  local pid
  pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for i in $(seq 1 120); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 1; done
  [ -n "$pid" ] && for i in $(seq 1 180); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
}
names=()
while [ $# -ge 2 ]; do
  name=$1; envs=$2; shift 2; names+=("$name")
  echo "=== $name ($envs) $(date +%T)"
  stop_port
  ( eval "export $envs"; SPEC=$SPEC_V NMAX=7 PMIN=0.0 PORT=$PORT CTX=16384 \
      MODEL=/models/mimo-v26-pro/proxy/mimo-v26-proxy-4L.gguf \
      DFLASH_MODEL=/models/mimo-v26-pro/gguf/MiMo-DFlash-proxytest.gguf \
      nohup ./launch-mimo-tp.sh > $OUT/server-$name.log 2>&1 & )
  for i in $(seq 1 600); do curl -sf -m 2 localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  echo "healthy after ${i}s"
  python3 golden.py $PORT $OUT/golden-$name.json 48 | cut -c1-60
  python3 tools/longprobe.py $PORT $OUT/long-$name.json 400
  python3 tools/longprobe.py $PORT $OUT/long2-$name.json 400
  ss -ltnH "sport = :$PORT" | grep -q . && echo "server alive" || echo "SERVER DIED"
  stop_port
done
echo "=== compare"
for ((i = 1; i < ${#names[@]}; i++)); do
  a=${names[$((i-1))]}; b=${names[$i]}
  echo "$a vs $b:"; python3 golden.py --compare $OUT/golden-$a.json $OUT/golden-$b.json; python3 golden.py --compare $OUT/long-$a.json $OUT/long-$b.json
done
echo "=== done $(date +%T)"
