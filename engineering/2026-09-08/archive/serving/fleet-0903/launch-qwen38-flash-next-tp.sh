#!/usr/bin/env bash
# Qwen3.8-Flash-Next (arch qwen4exp, 125B-A6B, 512 experts / 10 used, SSM hybrid
# + PLE) across all four SR950 sockets with the Meta tensor-parallel backend.
#
# The shipped launcher (~/.local/bin/launch-qwen38-flash-next.sh) pins the whole
# model to ONE socket (numactl --cpunodebind=1 --membind=1) and measured
# 6.32-6.47 tok/s.  That is a single-socket ~95 GB/s ceiling; this profile gives
# each socket a quarter of every tensor so all four memory controllers stream.
#
# HARD CONSTRAINT: f16 KV only.  A quantized KV cache trips
#   GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr)
# in qwen4exp.cpp build_attn_qsa, after the weights load cleanly.
set -u
PORT=${1:-18095}
BIN=${QWEN4E_BIN:-/home/user/InfoSystemic/AI-Server/engines/llama.cpp-qwen4exp-current/build-sr950/bin/llama-server}
ROOT=${QWEN4E_ROOT:-/models/gguf/Qwen3.8-Flash-Next}
MODEL=${QWEN4E_MODEL:-$ROOT/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf}
CTX=${QWEN4E_CTX:-32768}
THREADS=${QWEN4E_THREADS:-15}
ALIAS=${QWEN4E_ALIAS:-qwen38-flash-next-tp,qwen3.8-flash-next,flash-next}

[ -x "$BIN" ]   || { echo "missing binary: $BIN" >&2; exit 1; }
[ -r "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 1; }

export GGML_CPU_NUMA_DEVICES=${GGML_CPU_NUMA_DEVICES:-1}
export GGML_CPU_NUMA_THREADS=${GGML_CPU_NUMA_THREADS:-$THREADS}
export GGML_CPU_NUMA_POLL=${GGML_CPU_NUMA_POLL:-100}
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=${GGML_CPU_NUMA_DIRECT_ALLREDUCE:-1}
export GGML_CPU_NUMA_HUGEPAGES=${GGML_CPU_NUMA_HUGEPAGES:-0}

SPEC=()
if [ -n "${QWEN4E_MTP_MODEL:-}" ]; then
  SPEC=(--model-draft "$QWEN4E_MTP_MODEL"
        --spec-type "${QWEN4E_SPEC_TYPE:-draft-mtp}"
        --spec-draft-n-max "${QWEN4E_SPEC_N_MAX:-3}"
        --spec-draft-p-min "${QWEN4E_SPEC_P_MIN:-0.6}"
        --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3
        --spec-draft-ngl all
        --spec-draft-threads "$THREADS" --spec-draft-threads-batch "$THREADS")
fi

MMPROJ=()
[ -r "$ROOT/mmproj-F16.gguf" ] && [ "${QWEN4E_VISION:-0}" = "1" ] && MMPROJ=(--mmproj "$ROOT/mmproj-F16.gguf" --image-min-tokens 1024)

exec "$BIN" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --alias "$ALIAS" \
  "${MMPROJ[@]}" \
  --load-mode "${QWEN4E_LOAD_MODE:-mmap}" --fit off \
  --ctx-size "$CTX" --flash-attn on \
  --batch-size 2048 --ubatch-size 256 --parallel 1 --cache-prompt --cache-reuse 256 \
  --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  --threads "$THREADS" --threads-batch "$THREADS" \
  "${SPEC[@]}" \
  --jinja --reasoning-format deepseek --reasoning-preserve \
  --predict 32768 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --timeout 3600 --threads-http 4 --cors-origins localhost --no-webui --metrics --log-timestamps
