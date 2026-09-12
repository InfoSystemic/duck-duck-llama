#!/usr/bin/env bash
# Per-request speculative sweep on the running full-model server (no restart).
# Usage: mtp_sweep.sh <label> [port]
set -u
label=$1; port=${2:-18091}
DB=~/InfoSystemic/AI-Server/engines/llama-llama-duck/tools/decode_bench.py
OUT=/dev/shm/glm-dev/results; mkdir -p $OUT
for arm in "2 0" "3 0" "4 0" "3 0.5" "4 0.6" "6 0.6"; do
  set -- $arm; n=$1; p=$2
  python3 $DB --port $port --model glm-sr950 --label ${label}-n${n}p${p} --workloads prose,code,novel --reps 2 --max-tokens 160 --reasoning-effort low --spec-n-max $n --spec-p-min $p --out $OUT/full-${label}-n${n}p${p}.json 2>&1 | grep -E "SUMMARY"
done
echo "[mtp_sweep] done $(date +%T)"
