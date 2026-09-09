#!/usr/bin/env bash
# GLM-5.3-Flash (arch glm5next, 46 layers, 288 experts / 8 used, MLA + lightning
# indexer + linear-attention layers, nextn=1) across all four SR950 sockets.
#
# The shipped launcher pins it to ONE socket (numactl --cpunodebind=0
# --preferred=0, 32 threads).  This profile uses the Meta tensor-parallel
# backend so each socket streams a quarter of every tensor.
#
# The shipped single-socket launcher sets --flash-attn off, citing the support
# PR.  Tensor-parallel REFUSES to start without flash attention
# ("SPLIT_MODE_TENSOR requires flash_attn to be enabled", llama-context.cpp),
# so the two constraints collide and the FA-off claim has to be re-tested rather
# than assumed.  GLM-5.3 Full (glm-dsa, the same lightning-indexer family) runs
# with --flash-attn on in production, so it is plausible the restriction is
# stale.  GLM53F_FA exists to settle it by measurement.
set -u
PORT=${1:-18096}
BIN=${GLM53F_BIN:-/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server}
ROOT=${GLM53F_ROOT:-/models/gguf/GLM-5.3-Flash}
QUANT=${GLM53F_QUANT:-UD-IQ2_XXS}
MODEL="$ROOT/$QUANT/GLM-5.3-Flash-$QUANT-00001-of-00004.gguf"
CTX=${GLM53F_CTX:-32768}
THREADS=${GLM53F_THREADS:-15}
ALIAS=${GLM53F_ALIAS:-glm53-flash-tp,glm-5.3-flash,GLM-5.3-Flash}

[ -x "$BIN" ]   || { echo "missing binary: $BIN" >&2; exit 1; }
[ -r "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 1; }

export GGML_CPU_NUMA_DEVICES=${GGML_CPU_NUMA_DEVICES:-1}
export GGML_CPU_NUMA_THREADS=${GGML_CPU_NUMA_THREADS:-$THREADS}
export GGML_CPU_NUMA_POLL=${GGML_CPU_NUMA_POLL:-100}
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=${GGML_CPU_NUMA_DIRECT_ALLREDUCE:-1}
export GGML_CPU_NUMA_HUGEPAGES=${GGML_CPU_NUMA_HUGEPAGES:-0}
# node-local repack buffers: without this the TP path runs the STOCK kernels
export GGML_CPU_NUMA_REPACK=${GGML_CPU_NUMA_REPACK:-1}
export GGML_CPU_REPACK_LOAD_THREADS=${GGML_CPU_REPACK_LOAD_THREADS:-16}
export GGML_CPU_NUMA_FUSED_REDUCE=${GGML_CPU_NUMA_FUSED_REDUCE:-1}
export GGML_CPU_NUMA_MERGE_REDUCE=${GGML_CPU_NUMA_MERGE_REDUCE:-1}
export GGML_CPU_NUMA_DISPATCH_SPIN_US=${GGML_CPU_NUMA_DISPATCH_SPIN_US:-20000}
export GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US=${GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US:-300}
export GGML_CPU_ROUTER_F16=${GGML_CPU_ROUTER_F16:-1}
export GGML_CPU_X16_Q4_K=${GGML_CPU_X16_Q4_K:-1}
export GGML_CPU_X16_Q5_K=${GGML_CPU_X16_Q5_K:-1}
export GGML_CPU_X16_Q6_K=${GGML_CPU_X16_Q6_K:-1}
export GGML_CPU_IQ2_XS_REPACK=${GGML_CPU_IQ2_XS_REPACK:-1}
export GGML_CPU_IQ3_XXS_REPACK=${GGML_CPU_IQ3_XXS_REPACK:-1}
export GGML_CPU_IQ3_XXS_MOE_2ROW=${GGML_CPU_IQ3_XXS_MOE_2ROW:-1}
export GGML_CPU_MOE_GATE_UP_FUSION=${GGML_CPU_MOE_GATE_UP_FUSION:-1}
export GGML_CPU_FFN_GATE_UP_FUSION=${GGML_CPU_FFN_GATE_UP_FUSION:-1}
export NVIDIA_TF32_OVERRIDE=0

MMPROJ=()
[ "${GLM53F_VISION:-0}" = "1" ] && [ -r "$ROOT/mmproj-F16.gguf" ] && MMPROJ=(--mmproj "$ROOT/mmproj-F16.gguf")

exec "$BIN" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --alias "$ALIAS" \
  "${MMPROJ[@]}" \
  --load-mode "${GLM53F_LOAD_MODE:-mmap}" --fit off \
  --ctx-size "$CTX" --flash-attn "${GLM53F_FA:-on}" \
  --batch-size 512 --ubatch-size 256 --parallel 1 --cache-prompt --cache-reuse 256 \
  --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  --threads "$THREADS" --threads-batch "$THREADS" \
  --jinja --reasoning-format deepseek --reasoning-preserve \
  --predict 131072 --temp 1.0 --top-p 0.95 --presence-penalty 0.0 \
  --timeout 3600 --threads-http 4 --cors-origins localhost --no-webui --metrics --log-timestamps
