#!/bin/bash
# Restore the native DeepSeek-V4.1-Flash CPU endpoint on 18170.
# The server requires the existing lifecycle.lock inode (not a new file).
set -euo pipefail
BASE=/home/user/InfoSystemic/AI-Server/serving/fleet-0903
LOCK=$BASE/results/qwen-q6-trial-0907/lifecycle.lock
PY=/home/user/InfoSystemic/AI-Server/tools/deepseek-v41-cpu-reference-0910/venv/bin/python
CACHE=/dev/shm/deepseek-v41-native-fb2764-0910
OUT=$BASE/results/deepseek-v41-goal2-promoted-0910/server
LOG=$BASE/results/deepseek-v41-restore.log

if curl -s -m 2 http://127.0.0.1:18170/health 2>/dev/null | grep -q DeepSeek; then
  echo "DeepSeek-V4.1-Flash already healthy on 18170"
  exit 0
fi

export DEEPSEEK_GOAL2_EXACT16=1
export DEEPSEEK_GOAL2_NATIVE_SPARSE=1
export DEEPSEEK_GOAL2_PACKED_CAP_GIB=64
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OMP_WAIT_POLICY=PASSIVE
export TOKENIZERS_PARALLELISM=false

# Hold the required lock inode on fd 5 for the child.
exec 5>>"$LOCK"
if ! flock -n 5; then
  echo "lifecycle.lock is held; not starting a second server"
  exit 1
fi

cd "$BASE"
rm -f "$OUT/ready.json"
nohup taskset -c 48-63 "$PY" "$BASE/deepseek_v41_goal2_server_0910.py" \
  --cache "$CACHE" --output "$OUT" --port 18170 --lifecycle-lock-fd 5 \
  > "$LOG" 2>&1 &
SRV=$!
echo "restoring DeepSeek-V4.1 pid=$SRV"
for i in $(seq 300); do
  s=$(curl -s -m 2 http://127.0.0.1:18170/health 2>/dev/null || true)
  if [[ "$s" == *DeepSeek* ]]; then
    echo "DeepSeek-V4.1-Flash RESTORED on 18170"
    exit 0
  fi
  kill -0 "$SRV" 2>/dev/null || { echo "RESTORE DIED"; tail -20 "$LOG"; exit 1; }
  sleep 2
done
echo "restore TIMEOUT"; tail -20 "$LOG"; exit 1
