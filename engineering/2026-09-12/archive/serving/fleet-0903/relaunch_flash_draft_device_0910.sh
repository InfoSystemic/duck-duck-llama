#!/usr/bin/env bash
# relaunch_flash_draft_device_0910.sh - restart the selected GLM-5.3-Flash server with a
# different --spec-draft-device, keeping every other flag and every GGML_* knob identical.
#
# Why: traffic accounting on the 09-10 quiet run showed the MTP draft consuming 44% of all
# DRAM traffic at ~6.2 GB per draft pass vs a ~1.06 GB ideal -- it is sharded across all four
# sockets, so its small tensors are mirrored and re-streamed. See GOAL-BANDWIDTH-20260910.md.
#
# Usage: ./relaunch_flash_draft_device_0910.sh "CPU-NUMA0"                 # single-socket draft
#        ./relaunch_flash_draft_device_0910.sh "CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3"  # restore
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
RESTORE="$BASE/results/goal-0910-restore"
DRAFT_DEV="${1:?usage: $0 <spec-draft-device> [env-file]}"
ENVFILE="${2:-$RESTORE/env.txt}"
DRAFT_MODEL="${3:-}"
LOG="$BASE/results/goal-0910-restore/server-$(date -u +%H%M%S).log"

[[ -s "$RESTORE/argv.txt" && -s "$ENVFILE" ]] || { echo "missing captured argv/env"; exit 1; }

# stop whatever currently owns 18131
old=$(ss -ltnpH 'sport = :18131' 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
if [[ -n "${old:-}" ]]; then
  echo "stopping existing server pid=$old"
  kill "$old"
  for _ in $(seq 1 120); do kill -0 "$old" 2>/dev/null || break; sleep 1; done
  kill -0 "$old" 2>/dev/null && { echo "server did not exit"; exit 1; }
fi

# rebuild argv with the requested draft device
mapfile -t A < "$RESTORE/argv.txt"
ARGS=(); i=0
while (( i < ${#A[@]} )); do
  if [[ "${A[$i]}" == "--spec-draft-device" ]]; then ARGS+=("--spec-draft-device" "$DRAFT_DEV"); i=$((i+2)); continue; fi
  if [[ "${A[$i]}" == "--spec-draft-model" && -n "$DRAFT_MODEL" ]]; then ARGS+=("--spec-draft-model" "$DRAFT_MODEL"); i=$((i+2)); continue; fi
  ARGS+=("${A[$i]}"); i=$((i+1))
done

# exact runtime env
ENVARGS=()
while IFS= read -r line; do [[ -n "$line" ]] && ENVARGS+=("$line"); done < "$ENVFILE"

echo "starting with --spec-draft-device $DRAFT_DEV"
cd "$BASE"
nohup env "${ENVARGS[@]}" "${ARGS[@]}" > "$LOG" 2>&1 &
new=$!
echo "pid=$new log=$LOG"
for _ in $(seq 1 600); do
  if curl -sf -m 3 http://127.0.0.1:18131/health >/dev/null 2>&1; then
    echo "healthy after ${SECONDS}s"; echo "$new" > "$RESTORE/current.pid"; exit 0
  fi
  kill -0 "$new" 2>/dev/null || { echo "server died; tail:"; tail -20 "$LOG"; exit 1; }
  sleep 2
done
echo "timeout waiting for health"; tail -20 "$LOG"; exit 1
