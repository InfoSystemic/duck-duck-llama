#!/usr/bin/env bash
# Start one model server, wait for health, prove it is not producing garbage,
# then benchmark it. Leaves the server running so follow-up sweeps are free.
#
# usage: run-model.sh <launcher> <port> <alias> <label> [extra env assignments...]
set -u
LAUNCHER=$1; PORT=$2; ALIAS=$3; LABEL=$4; shift 4
DIR=/home/user/InfoSystemic/AI-Server/serving/fleet-0903
LOG=$DIR/results/$LABEL.server.log
mkdir -p "$DIR/results"

# stop anything already on this port
for p in $(pgrep -f "llama-server.*--port $PORT" || true); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f "llama-server.*--port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
sleep 2

echo "[$LABEL] launching $LAUNCHER on $PORT  $(date -u +%FT%TZ)"
echo "[$LABEL] env: $*"
( setsid env "$@" "$LAUNCHER" "$PORT" > "$LOG" 2>&1 < /dev/null & )

t0=$(date +%s)
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! pgrep -f "llama-server.*--port $PORT" >/dev/null; then
    echo "[$LABEL] SERVER DIED after $(( $(date +%s)-t0 ))s"
    grep -iE 'error|assert|abort|failed|terminate|what\(\)' "$LOG" | tail -15
    tail -5 "$LOG"
    exit 1
  fi
  sleep 10
done
echo "[$LABEL] healthy after $(( ($(date +%s)-t0) ))s"

"$DIR/bench.sh" "$PORT" "$ALIAS" "$LABEL"
