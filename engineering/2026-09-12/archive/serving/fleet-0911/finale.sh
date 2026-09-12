#!/bin/bash
D=~/InfoSystemic/AI-Server/serving/fleet-0911
for i in $(seq 400); do grep -q 'DSV4 DONE' $D/results/chain_ds.out 2>/dev/null && break; sleep 10; done
echo "=== deepseek done; GLM-5.3-Flash raw (max-bandwidth config) ==="
if $D/launch-glm-raw.sh > $D/results/launch-glmraw.log 2>&1; then
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  $D/bench2.sh 18131 glm-raw 192
else echo "glm-raw LAUNCH_FAIL"; tail -5 $D/results/launch-glmraw.log; fi
echo "=== restoring production ==="
$D/finish.sh
echo "=== FINALE DONE ==="
