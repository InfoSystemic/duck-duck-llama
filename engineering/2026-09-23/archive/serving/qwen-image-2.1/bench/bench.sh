#!/usr/bin/env bash
# ggml CPU mul_mat throughput at the Qwen-Image-2.1 DiT shapes, per weight type and NUMA layout.
# Every run goes through yield-gate.py, like real jobs, so it never competes with a decode.
#   bench.sh [N tokens, default 1100 = 512x512]      -> results/mmbench-<N>.txt
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-1100}
SPARE=15,31,47,63,79,95,111,127
mkdir -p results
out=results/mmbench-$N.txt
: > "$out"

run() {  # <label> <cpus> <threads> <numactl policy> <type> <M> <K>
    local label=$1 cpus=$2 nth=$3 pol=$4; shift 4
    printf '%-24s ' "$label" | tee -a "$out"
    GATE_CPUS=$cpus taskset -c "$SPARE" python3 ../yield-gate.py \
        numactl "$pol" taskset -c "$cpus" nice -n 10 ./mmbench "$@" "$N" "$nth" 5 | tee -a "$out"
}

for shape in "4096 4096" "24576 4096" "4096 12288"; do
    for t in q8_0 f16 f32; do
        run "1 socket, local"      32-46             15 --membind=2        $t $shape
        run "3 sockets, 1 node"    0-14,16-30,32-46  45 --membind=2        $t $shape
        run "3 sockets, interleave" 0-14,16-30,32-46 45 --interleave=0,1,2 $t $shape
    done
done
