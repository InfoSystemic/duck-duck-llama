#!/bin/bash
# Max tok/s hunt on Qwen: pinned MTP4 + the UNARY/SCALE fix + concurrency, one load.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
PIN=~/InfoSystemic/AI-Server/serving/fleet-0903/qwen-flash-20tps.json
BASE_LDP=$(python3 -c "import json;print(json.load(open('$PIN'))['runtime_env']['LD_LIBRARY_PATH'])")
export GGML_CPU_PARALLEL_UNARY=4096
LDP="/dev/shm/q4ebuild:$BASE_LDP" $D/launch-qwen-patched.sh spec --parallel 8 --ctx-size 32768 \
  > $D/results/launch-qwenmax.log 2>&1 || { echo LAUNCH_FAIL; tail -6 $D/results/launch-qwenmax.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
echo "### single-stream (MTP4 + fix), 3 prompts"
$D/bench2.sh 18131 qmax1 192
echo "### concurrency sweep (aggregate)"
for C in 2 4 8; do $D/benchpar.sh 18131 qmax$C $C 192; done
echo "=== QWENMAX DONE ==="
