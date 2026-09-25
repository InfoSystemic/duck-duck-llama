#!/usr/bin/env bash
# Everything that has to be true for "MiMo works well on this machine", in one pass.
set -uo pipefail
cd "$(dirname "$0")"
pid=$(ss -ltnpH "sport = :18190" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
last=0
for i in $(seq 1 300); do
  case "$(curl -s -m 3 http://127.0.0.1:18190/health 2>/dev/null)" in *'"ok"'*) echo "=== HEALTHY $(date +%H:%M:%S) ==="; break;; esac
  kill -0 "$pid" 2>/dev/null || { echo "=== DIED ==="; grep -a -iE "error|assert|abort" /models/mimo-production.log | grep -v unused | tail -6 | cut -c1-175; exit 1; }
  g=$(( $(awk '/^read_bytes/{print $2}' "/proc/$pid/io" 2>/dev/null || echo 0) / 1073741824 ))
  (( g - last >= 150 )) && { echo "loading: ${g} GiB at $(date +%H:%M)"; last=$g; }
  sleep 20
done
curl -s -m 300 -o /dev/null -X POST http://127.0.0.1:18190/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}'
kill -0 "$pid" 2>/dev/null && echo "canary SURVIVED" || { echo "DIED on first decode"; exit 1; }

echo; echo "=== 1. context actually available ==="
curl -s -m 10 http://127.0.0.1:18190/props | python3 -c "import json,sys; d=json.load(sys.stdin); print('   n_ctx =', d.get('default_generation_settings',{}).get('n_ctx'), ' slots =', d.get('total_slots'))"
free -g | awk 'NR==2{print "   RAM used", $3, "GiB, available", $7}'

echo; echo "=== 2. exactness: speculation must not change output ==="
python3 golden.py 18190 results/golden-production.json 48 >/dev/null 2>&1
python3 golden.py --compare results/golden-mtp3-q8_0.json results/golden-production.json

echo; echo "=== 3. full functional smoke, vision included ==="
timeout 2400 python3 smoke-mimo.py 18190 mimo-production 2>&1 | tail -9

echo; echo "=== 4. speed by workload ==="
python3 draft-probe.py 18190 96

echo; echo "=== 5. is 256K usable, or does decode collapse at depth? ==="
timeout 5400 python3 depth-probe.py 18190 4096 16384 65536
