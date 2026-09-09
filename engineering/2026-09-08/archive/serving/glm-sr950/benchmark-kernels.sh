#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
window_guard="${GLM_BENCH_WINDOW_GUARD:-$script_dir/assert-benchmark-window.sh}"
config_file="${GLM_SR950_CONFIG:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.env}"
if [[ ! -r "$config_file" ]]; then
    echo "GLM configuration is not readable: $config_file" >&2
    exit 1
fi
"$window_guard"

set -a
# shellcheck source=/dev/null
source "$config_file"
set +a

binary="${GLM_KERNEL_BENCH_BINARY:-$(dirname "$GLM_BINARY")/test-backend-ops}"
device="${GLM_KERNEL_BENCH_DEVICE:-CPU-NUMA2}"
threads="${GLM_KERNEL_BENCH_THREADS:-$GLM_THREADS}"
weight_buffer="${GLM_KERNEL_BENCH_WEIGHT_BUFFER:-${device}_REPACK}"

if [[ ! -x "$binary" ]]; then
    echo "Kernel benchmark binary is not executable: $binary" >&2
    exit 1
fi
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "GLM_KERNEL_BENCH_THREADS must be a positive integer" >&2
    exit 2
fi

export GGML_TEST_N_THREADS="$threads"
export GGML_TEST_WEIGHT_BUFFER_TYPE="$weight_buffer"

run_case() {
    local label="$1"
    local op="$2"
    local params="$3"

    "$window_guard"
    printf '\n[%s] device=%s buffer=%s threads=%s\n' \
        "$label" "$device" "$weight_buffer" "$threads"
    "$binary" perf -b "$device" -o "$op" -p "$params"
}

run_case q5-dense-fused \
    MUL_MAT_SWIGLU \
    'type_a=q5_K,m=512,n=1,k=6144'
run_case iq2-routed-fused \
    MUL_MAT_ID_SWIGLU \
    'type_a=iq2_xs.*n_mats=8,n_used=8,m=512,n=1,k=6144'
run_case iq3-routed \
    MUL_MAT_ID \
    'type_a=iq3_xxs.*n_mats=8,n_used=8.*m=6144,n=1,k=2048'
run_case iq3-routed-weighted-fused \
    MUL_MAT_ID_WEIGHTED_SUM \
    'type_a=iq3_xxs,n_mats=8,n_used=8,m=6144,n=1,k=2048'
