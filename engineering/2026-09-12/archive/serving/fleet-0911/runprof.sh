#!/bin/bash
D=~/InfoSystemic/AI-Server/serving/fleet-0911
while pgrep -f 'bash ./sweep.sh' >/dev/null || pgrep -f 'bash ./runpar.sh' >/dev/null; do sleep 20; done
echo "=== prior tests done; launching profile server (single stream, op profiler armed) ==="
$D/launch.sh "GGML_CPU_OP_PROFILE=* GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/bwprobe/opprof.arm GGML_CPU_OP_PROFILE_COUNT=4" > $D/results/launch-prof.log 2>&1 || { echo LAUNCH_FAIL; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1
sleep 3
$D/opprofile.sh main
echo "=== op profile done ==="
