#!/bin/bash
# Wait for the QWEN DONE marker in the output FILE (never poll a process pattern -- that
# livelocks when the writing shell's own cmdline contains the pattern).
D=~/InfoSystemic/AI-Server/serving/fleet-0911
for i in $(seq 360); do grep -q 'QWEN DONE' $D/results/qwen5.out 2>/dev/null && break; sleep 10; done
echo "=== qwen done, starting DeepSeek-V4-Flash ==="
$D/dsrun.sh
