#!/usr/bin/env bash
# Qwen3.8-27B (arch qwen35, dense + hybrid SSM, in-model MTP head) on the SR950
# using the GLM fast fork (build-dev2): 4-socket NUMA tensor-parallel, spin
# dispatch, fused/merged all-reduce, x16 AVX-512-VNNI kernels and optional
# load-time requantization of the Q8_0 attention / dense-FFN / output tensors.
#
# Baseline to beat (2026-08-29, old b249 NUMA stack, Q8_K_XL):
#   raw 7.169 tok/s | general MTP 16.49 | agentic replay 23.58
set -u
PORT=${1:-18094}
BIN=${QWEN27_BIN:-/home/user/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/build-dev2/bin/llama-server}
MODEL=${QWEN27_MODEL:-/home/user/.local/share/ai-models/Qwen3.8-27B-UD-Q8_K_XL/Qwen3.8-27B-UD-Q8_K_XL.gguf}
CTX=${QWEN27_CTX:-32768}
THREADS=${QWEN27_THREADS:-15}
ALIAS=${QWEN27_ALIAS:-qwen38-27b-fast,qwen3.8-27b,qwen38,Qwen3.8-27B}

[ -x "$BIN" ]     || { echo "missing binary: $BIN" >&2; exit 1; }
[ -r "$MODEL" ]   || { echo "missing model: $MODEL" >&2; exit 1; }

# --- ggml knobs (all opt-in; unset ones keep upstream behaviour) -------------
export GGML_CPU_NUMA_DEVICES=${GGML_CPU_NUMA_DEVICES:-1}
export GGML_CPU_NUMA_THREADS=${GGML_CPU_NUMA_THREADS:-$THREADS}
export GGML_CPU_NUMA_POLL=${GGML_CPU_NUMA_POLL:-100}
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=${GGML_CPU_NUMA_DIRECT_ALLREDUCE:-1}
export GGML_CPU_NUMA_FUSED_REDUCE=${GGML_CPU_NUMA_FUSED_REDUCE:-1}
export GGML_CPU_NUMA_MERGE_REDUCE=${GGML_CPU_NUMA_MERGE_REDUCE:-1}
export GGML_CPU_NUMA_DISPATCH_SPIN_US=${GGML_CPU_NUMA_DISPATCH_SPIN_US:-20000}
export GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US=${GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US:-300}
export GGML_CPU_NUMA_HUGEPAGES=${GGML_CPU_NUMA_HUGEPAGES:-0}
export GGML_CPU_NUMA_REPACK=${GGML_CPU_NUMA_REPACK:-1}
export GGML_CPU_REPACK_LOAD_THREADS=${GGML_CPU_REPACK_LOAD_THREADS:-16}
export GGML_CPU_Q8_0_REPACK=${GGML_CPU_Q8_0_REPACK:-1}
export GGML_CPU_Q8_0_REPACK_FFN=${GGML_CPU_Q8_0_REPACK_FFN:-1}
export GGML_CPU_Q8_0_REPACK_X_TILE=${GGML_CPU_Q8_0_REPACK_X_TILE:-auto}
export GGML_CPU_Q4_K_REPACK=${GGML_CPU_Q4_K_REPACK:-1}
export GGML_CPU_Q5_K_REPACK=${GGML_CPU_Q5_K_REPACK:-1}
export GGML_CPU_X16_Q4_K=${GGML_CPU_X16_Q4_K:-1}
export GGML_CPU_X16_Q5_K=${GGML_CPU_X16_Q5_K:-1}
export GGML_CPU_X16_Q6_K=${GGML_CPU_X16_Q6_K:-1}
export GGML_CPU_FFN_GATE_UP_FUSION=${GGML_CPU_FFN_GATE_UP_FUSION:-1}
export GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=${GGML_CPU_SINGLE_TASK_MAX_ELEMENTS:-32768}
# requant is quality-affecting: empty by default, set to q6_K / q5_K to measure
export GGML_CPU_ATTN_REQUANT=${GGML_CPU_ATTN_REQUANT:-}
export GGML_CPU_DENSE_FFN_REQUANT=${GGML_CPU_DENSE_FFN_REQUANT:-}
export GGML_CPU_OUTPUT_REQUANT=${GGML_CPU_OUTPUT_REQUANT:-}

# Draft sources. The in-model NextN layer (blk.64) is a FULL transformer layer
# plus a 1.04 GB lm_head, so every drafted token costs a real forward pass -- which
# is why depth beyond n=3 loses (measured: n3 14.96 > n4 14.26 > n8 11.79 > n12 10.65).
# ngram-mod drafts from repetition already in the context and costs nothing per
# token, so the composite is what pays on code and agentic traffic.
SPEC=()
if [ "${QWEN27_SPEC:-1}" != "0" ]; then
  SPEC=(--spec-type "${QWEN27_SPEC_TYPE:-draft-mtp}"
        --spec-draft-n-max "${QWEN27_SPEC_N_MAX:-3}"
        --spec-draft-p-min "${QWEN27_SPEC_P_MIN:-0.2}"
        --spec-draft-threads "$THREADS" --spec-draft-threads-batch "$THREADS")
  case "${QWEN27_SPEC_TYPE:-draft-mtp}" in
    *ngram*) SPEC+=(--spec-ngram-mod-n-match "${QWEN27_NGRAM_MATCH:-8}"
                    --spec-ngram-mod-n-max   "${QWEN27_NGRAM_MAX:-64}"
                    --spec-ngram-mod-n-min   "${QWEN27_NGRAM_MIN:-2}") ;;
  esac
  [ -n "${QWEN27_DRAFT_MODEL:-}" ] && SPEC+=(--model-draft "$QWEN27_DRAFT_MODEL" --spec-draft-ngl all)
fi

exec "$BIN" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --alias "$ALIAS" \
  --load-mode mmap --fit off \
  --ctx-size "$CTX" --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 --parallel 1 --cache-prompt --cache-reuse 256 \
  --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  --threads "$THREADS" --threads-batch "$THREADS" \
  "${SPEC[@]}" \
  --jinja --reasoning-format deepseek --reasoning-preserve \
  --predict 32768 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --timeout 3600 --threads-http 4 --cors-origins localhost --no-webui --metrics --log-timestamps
