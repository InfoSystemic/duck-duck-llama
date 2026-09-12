#!/bin/bash
# Does Qwen have the ops the GLM fixes target? Profile with the q4e runtime's own profiler.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
export GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/bwprobe/opprof.arm
export GGML_CPU_OP_PROFILE_COUNT=3 GGML_CPU_OP_PROFILE_SKIP=2
$D/launch-qwen.sh raw > $D/results/launch-qwenprof.log 2>&1 || { echo LAUNCH_FAIL; tail -5 $D/results/launch-qwenprof.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
LOG=$(ls -t $D/results/qwen-*.log | head -1); MARK=$(wc -l < "$LOG")
touch /dev/shm/bwprobe/opprof.arm
curl -s -m 200 http://127.0.0.1:18131/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Count from one to twenty in words.","n_predict":20,"temperature":0,"cache_prompt":false}' >/dev/null
rm -f /dev/shm/bwprobe/opprof.arm
tail -n +$((MARK+1)) "$LOG" | grep CPU_OP_PROFILE > $D/results/qwenprof.txt
echo "captured $(wc -l < $D/results/qwenprof.txt) lines"
echo "=== QWENPROF DONE ==="
