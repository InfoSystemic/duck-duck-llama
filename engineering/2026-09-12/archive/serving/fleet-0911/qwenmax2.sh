#!/bin/bash
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
PIN=~/InfoSystemic/AI-Server/serving/fleet-0903/qwen-flash-20tps.json
BASE_LDP=$(python3 -c "import json;print(json.load(open('$PIN'))['runtime_env']['LD_LIBRARY_PATH'])")
export GGML_CPU_PARALLEL_UNARY=4096
LDP="/dev/shm/q4ebuild:$BASE_LDP" $D/launch-qwen-patched.sh spec --parallel 8 --ctx-size 32768 \
  > $D/results/launch-qwenmax2.log 2>&1 || { echo LAUNCH_FAIL; tail -6 $D/results/launch-qwenmax2.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
L=$(ls -t $D/results/qwen-*.log | head -1)
grep -oE 'n_parallel *= *[0-9]+|n_seq_max *= *[0-9]+|n_ctx *= *[0-9]+' "$L" | sort -u | head -4 | sed 's/^/  server: /'
for C in 1 2 4 8; do $D/benchpar.sh 18131 q2max$C $C 192; done
echo "=== QWENMAX2 DONE ==="
