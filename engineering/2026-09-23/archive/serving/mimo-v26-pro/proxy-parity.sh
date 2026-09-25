#!/bin/bash
# Engine parity on the truncated proxy: upstream (reference) vs fork vs fork+4-way TP.
# Same greedy prompts through each; a wrong tensor split shows up as divergent tokens.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
PROXY=${PROXY:-/models/mimo-v26-pro/proxy/mimo-v26-proxy-4L.gguf}
PY=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-dsv41-tuned-0912/.venv/bin/python
run() { # name engine tp port
  local name=$1 engine=$2 tp=$3 port=$4
  echo "=== $name (ENGINE=$engine TP=$tp) $(date +%T)"
  ENGINE=$engine TP=$tp PORT=$port CTX=8192 MODEL=$PROXY LOAD=none \
    setsid ./launch-mimo-tp.sh > "logs/proxy-$name.log" 2>&1 < /dev/null &
  local pid=$!
  local t0=$(date +%s)
  for i in $(seq 900); do
    curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 $pid 2>/dev/null || { echo "  DIED during load"; grep -i -E 'error|abort|assert|GGML_ASSERT|fatal' "logs/proxy-$name.log" | tail -5; return 1; }
    sleep 2
  done
  curl -s -m 2 "http://127.0.0.1:$port/health" | grep -q '"ok"' || { echo "  load TIMEOUT"; return 1; }
  echo "  healthy after $(( $(date +%s) - t0 ))s"
  timeout 900 $PY golden.py "$port" "golden/proxy-$name.json" 16
  ~/InfoSystemic/AI-Server/serving/fleet-0912-ctx/stop-by-port.sh "$port" > /dev/null
  sleep 5
}
run upstream upstream 0 18191
run fork-notp tp 0 18192
run fork-tp4  tp 1 18193
echo "=== parity: fork-notp vs upstream"; $PY golden.py --compare golden/proxy-upstream.json golden/proxy-fork-notp.json
echo "=== parity: fork-tp4 vs upstream";  $PY golden.py --compare golden/proxy-upstream.json golden/proxy-fork-tp4.json
echo "=== PROXY PARITY DONE"
