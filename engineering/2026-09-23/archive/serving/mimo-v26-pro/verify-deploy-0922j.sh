#!/usr/bin/env bash
# Post-deploy verification for build-prod-0922j (= 0922g + paired-row-group VNNI prompt-batch kernels + exact MoE weighted-sum fusion + 1024-token micro-batches + vectorised flash-attention row max, all bit-identical; 0922i never finished loading: per-node OOM).
set -uo pipefail
cd "$(dirname "$0")"
R=results/deploy-0922j
mkdir -p "$R"
PORT=18190
pid=$(systemctl --user show -p MainPID --value mimo-v26-pro.service)
echo "=== waiting for health (pid $pid) $(date +%T)"
for i in $(seq 1 540); do
  case "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" in *'"ok"'*) echo "=== HEALTHY $(date +%T)"; break;; esac
  kill -0 "$pid" 2>/dev/null || { echo "=== DIED during load"; exit 1; }
  sleep 10
done
curl -s -m 600 -o /dev/null -X POST http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}'
kill -0 "$pid" 2>/dev/null && echo "canary SURVIVED" || { echo "DIED on first decode"; exit 1; }
echo; echo "=== 1. golden vs deploy-0922g (bit-exact changes: must be IDENTICAL)"
python3 golden.py $PORT "$R/golden.json" 48 > /dev/null 2>&1
python3 golden.py --compare results/deploy-0922g/golden.json "$R/golden.json" | tee "$R/golden-compare.txt"
echo; echo "=== 2. tool calls at temperature 1.0 (0922g: see results/deploy-0922g) $(date +%T)"
python3 tools/toolcall-probe.py $PORT 20 2>&1 | tee "$R/toolcall-probe.txt"
echo; echo "=== 3. functional smoke $(date +%T)"
timeout 2400 python3 smoke-mimo.py $PORT deploy-0922j 2>&1 | tail -9 | tee "$R/smoke.txt"
echo; echo "=== 4. vision + audio $(date +%T)"
python3 tools/vision-real-probe.py $PORT 2>&1 | tail -1 | tee "$R/vision-real.txt"
python3 tools/audio-probe.py $PORT 2>&1 | tee "$R/audio.txt"
echo "=== done $(date +%T)"
