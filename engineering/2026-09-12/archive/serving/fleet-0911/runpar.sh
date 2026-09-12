#!/bin/bash
# Wait for sweep1 to finish, then test batching/concurrency from a single load.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
while pgrep -f 'bash ./sweep.sh' >/dev/null; do sleep 20; done
echo "=== sweep1 done, starting concurrency test ==="
$D/launch.sh "GGML_CPU_DUMMY=0" --parallel 8 --ctx-size 32768 > $D/results/launch-par8.log 2>&1 || { echo LAUNCH_FAIL; tail -5 $D/results/launch-par8.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1
grep -E 'n_ctx|n_parallel|n_seq_max' $D/results/$(ls -t $D/results | grep server- | head -1) 2>/dev/null | head -4
sleep 3
for C in 1 2 4 8; do $D/benchpar.sh 18131 par$C $C 192; done
echo "=== concurrency test done ==="
