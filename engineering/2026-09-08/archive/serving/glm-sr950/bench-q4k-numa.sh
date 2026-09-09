#!/usr/bin/env bash
# Characterise the Q4_K decode path on the SR950: interleaved vs socket-local
# numa-tensor, with and without the per-node repack wrapper.
#
# Purpose: Q4_K selects `q4_K_8x8_q8_K` (repack.cpp:5971) with NO env gate,
# unlike IQ2_XS/IQ3_XXS/Q5_K. This measures what fraction of the machine's
# measured 379 GB/s socket-local ceiling that path actually extracts on a MoE,
# so the GLM-5.3 UD-Q4_K_XL projection rests on a measurement, not a hope.
set -uo pipefail

BUILD=${BUILD:-/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/build-sr950-glm-pgo-q5-r8-20260826}
BIN="$BUILD/bin/llama-bench"
export LD_LIBRARY_PATH="$BUILD/bin:${LD_LIBRARY_PATH:-}"
MODEL=${MODEL:-/models/gguf/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf}
NGEN=${NGEN:-32}
REPS=${REPS:-2}

echo "model:  $MODEL"
echo "binary: $BIN"
echo

run() {
  local label="$1"; shift
  echo "=============================================================="
  echo "== $label"
  echo "=============================================================="
  "$@" 2>&1 | grep -Ev '^(ggml_|load_|llama_model_load|print_info|init_tokenizer)' | tail -20
  echo
}

# --- A: plain CPU backend, interleaved across all 4 nodes (the naive config) --
GGML_CPU_NUMA_DEVICES=0 \
run "A. plain CPU, numactl --interleave=all, 64 threads" \
  numactl --interleave=all "$BIN" -m "$MODEL" -p 0 -n "$NGEN" -t 64 -r "$REPS" --numa numactl

# --- B: socket-local tensor-parallel over the 4 NUMA devices --------------
GGML_CPU_NUMA_DEVICES=1 GGML_CPU_NUMA_THREADS=16 GGML_CPU_NUMA_POLL=50 \
GGML_CPU_NUMA_HUGEPAGES=1 GGML_CPU_NUMA_DIRECT_ALLREDUCE=1 \
GGML_CPU_NUMA_REPACK=0 \
run "B. numa-tensor socket-local, repack OFF" \
  "$BIN" -m "$MODEL" -p 0 -n "$NGEN" -t 16 -r "$REPS" \
    --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
    --split-mode tensor --tensor-split 1,1,1,1 -ngl 999

# --- C: same, with the per-node repack wrapper (Q4_K -> q4_K_8x8_q8_K) ----
GGML_CPU_NUMA_DEVICES=1 GGML_CPU_NUMA_THREADS=16 GGML_CPU_NUMA_POLL=50 \
GGML_CPU_NUMA_HUGEPAGES=1 GGML_CPU_NUMA_DIRECT_ALLREDUCE=1 \
GGML_CPU_NUMA_REPACK=1 GGML_CPU_REPACK_LOAD_THREADS=8 \
run "C. numa-tensor socket-local, NUMA_REPACK=1 (Q4_K 8x8)" \
  "$BIN" -m "$MODEL" -p 0 -n "$NGEN" -t 16 -r "$REPS" \
    --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
    --split-mode tensor --tensor-split 1,1,1,1 -ngl 999

echo "done."
