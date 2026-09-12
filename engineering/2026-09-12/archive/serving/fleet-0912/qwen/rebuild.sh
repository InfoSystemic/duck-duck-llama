#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
export OUT="${1:-$D/build}"
bash "$D/build.sh"
bash "$D/relink.sh"
ln -sf libggml-cpu.so.0.22.0 "$OUT/libggml-cpu.so.0"
ln -sf libggml-cpu.so.0 "$OUT/libggml-cpu.so"
cmp "$OUT/base.so" /home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0
echo "Baseline relink is byte-identical to pinned CPU library."
