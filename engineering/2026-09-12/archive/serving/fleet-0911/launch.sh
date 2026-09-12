#!/bin/bash
# launch.sh "KEY=VAL KEY=VAL ..." [extra llama-server args...]
# Starts GLM-5.3-Flash Q4 on 18131 with the captured baseline env + overrides.
set -uo pipefail
D=~/InfoSystemic/AI-Server/serving/fleet-0911
OVR="${1:-}"; shift || true
# Only ports this session owns -- a broad name match kills other sessions' servers.
for p in $(ss -ltnpH "sport = :18131" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill "$p" 2>/dev/null; done
for i in $(seq 120); do ss -ltn 2>/dev/null | grep -q ':18131 ' || break; sleep 1; done
for i in $(seq 120); do ss -ltn 2>/dev/null | grep -q ':18131 ' || break; sleep 1; done
sleep 2
set -a; while read -r l; do [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
for kv in $OVR; do export "$kv"; done
mapfile -t ARGV < <(grep -v '^$' $D/restore-argv.txt)
LOG=$D/results/server-$(date +%H%M%S).log
T0=$(date +%s)
nohup "${ARGV[@]}" "$@" > "$LOG" 2>&1 &
echo "pid=$! log=$LOG"
for i in $(seq 900); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null)
  [[ "$s" == *'"ok"'* ]] && { echo "UP in $(( $(date +%s)-T0 ))s"; exit 0; }
  sleep 2
done
echo "TIMEOUT"; tail -5 "$LOG"; exit 1
