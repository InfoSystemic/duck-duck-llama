#!/usr/bin/env bash
# Correctness probe + decode benchmark against a running llama-server.
# usage: bench.sh <port> <model-alias> <label> [n_max] [p_min] [workloads]
set -u
PORT=$1; MODEL=$2; LABEL=$3; NMAX=${4:-}; PMIN=${5:-}; WL=${6:-prose,code,novel}
DIR=/home/kwebb/InfoSystemic/AI-Server/serving/fleet-0903
DB=/home/kwebb/InfoSystemic/AI-Server/engines/llama-llama-duck/tools/decode_bench.py
OUT=$DIR/results; mkdir -p "$OUT"

echo "[$LABEL] probe:"
if python3 "$DIR/probe.py" "$PORT" "$MODEL" | sed "s/^/[$LABEL]   /"; then
  echo "[$LABEL] CORRECT"
else
  echo "[$LABEL] *** INCORRECT — do not trust throughput ***"
fi

ARGS=()
[ -n "$NMAX" ] && ARGS+=(--spec-n-max "$NMAX")
[ -n "$PMIN" ] && ARGS+=(--spec-p-min "$PMIN")
python3 "$DB" --port "$PORT" --model "$MODEL" --label "$LABEL" --workloads "$WL" \
  --reps 2 --max-tokens 160 --reasoning-effort low "${ARGS[@]}" \
  --out "$OUT/$LABEL.json" 2>&1 | grep -E "SUMMARY|ERROR|Traceback"
