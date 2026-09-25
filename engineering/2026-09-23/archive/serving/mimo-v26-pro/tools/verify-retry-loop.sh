#!/bin/bash
# verify-retry-loop.sh <verify-deploy-XXXX.sh> [log]: run a post-deploy verification once mimo-v26-pro.service has a
# real server process, and again if that load dies (load-watchdog kill / OOM) before verification completes.
# Needed because the unit may sit in its RAM guard (activating start-pre, MainPID 0) for a long time.
cd "$(dirname "$0")/.."
v=$1
log=${2:-/tmp/mimo-vis/$(basename "$v" .sh).log}
for attempt in $(seq 1 20); do
  until p=$(systemctl --user show -p MainPID --value mimo-v26-pro.service); [ -n "$p" ] && [ "$p" != 0 ] && kill -0 "$p" 2>/dev/null; do sleep 30; done
  echo "=== attempt $attempt: server pid $p since $(date +%T)"
  "./$v" > "$log" 2>&1
  if grep -q '=== done' "$log"; then echo "=== verified $(date +%T)"; exit 0; fi
  echo "=== attempt $attempt did not complete: $(tail -1 "$log")"; sleep 60
done
