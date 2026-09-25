#!/bin/bash
# Op-level profile of one PREFILL on the production server, using its dormant file-armed profiler
# (GGML_CPU_OP_PROFILE='*', ARM_FILE=/tmp/mimo-prof-arm, COUNT=16 -- set by launch-mimo-production.sh).
# 16 graphs = 4 ubatches x 4 NUMA nodes. Only run when nothing else is using :18190.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-/tmp/mimo-vis/prodprof}
mkdir -p "$OUT"
since=$(date '+%Y-%m-%d %H:%M:%S')
rm -f /tmp/mimo-prof-arm
touch /tmp/mimo-prof-arm
python3 tools/longprobe.py 18190 "$OUT/probe.json" ${N_LINES:-220}
sleep 2
rm -f /tmp/mimo-prof-arm
journalctl --user -u mimo-v26-pro.service --since "$since" --no-pager -o cat | grep -a 'CPU_OP_PROFILE' > "$OUT/prof.log"
echo "records: $(grep -c 'CPU_OP_PROFILE index=' "$OUT/prof.log") graphs"
python3 tools/opsum.py "$OUT/prof.log"
