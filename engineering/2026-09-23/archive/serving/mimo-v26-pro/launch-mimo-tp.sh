#!/bin/bash
# MiMo-V2.6-Pro on the NUMA-tuned fork (engines/llama.cpp-mimo-tp) -- 4-way CPU tensor parallelism,
# one shard of every tensor per socket, so each socket reads its own memory instead of crossing UPI.
#   env ENGINE=tp|upstream  TP=1|0  MODEL=<gguf>  PORT  CTX  SPEC=none|mtp  NMAX  THREADS (per device)
#       LOAD=mmap|none  MMPROJ=1|0  MMPROJ_FILE  IMAGE_MAX_TOKENS  EXTRA="<more flags>"
set -euo pipefail
ENGINE=${ENGINE:-tp}
TP=${TP:-1}
PORT=${PORT:-18190}
CTX=${CTX:-8192}
SPEC=${SPEC:-none}
MODEL=${MODEL:-/models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE-00001-of-00013.gguf}
THREADS=${THREADS:-15}
LOAD=${LOAD:-mmap}
if [ "$ENGINE" = tp ]; then
  # BUILD selects which build tree of the fork to run; the binary and its libraries must come from the SAME one,
  # because substituting either silently changes the configuration. Default is the shared build/.
  BUILD=${BUILD:-build}
  BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-mimo-tp/$BUILD/bin/llama-server
  export LD_LIBRARY_PATH=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-mimo-tp/$BUILD/bin
  # the fork's NUMA engine + the kernels that matter for this model:
  #   MXFP4 x16 VNNI experts (the V4.1 port), x16 Q8_0 for MiMo's very large attention matrices
  export GGML_CPU_NUMA_DEVICES=1 GGML_CPU_NUMA_THREADS="$THREADS" GGML_CPU_NUMA_POLL=100
  export GGML_CPU_NUMA_REPACK=1 GGML_CPU_NUMA_FUSED_REDUCE=1 GGML_CPU_NUMA_MERGE_REDUCE=1
  export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1 GGML_CPU_NUMA_HUGEPAGES=0
  export GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS=65536
  export GGML_CPU_REPACK_LOAD_THREADS=16
  export GGML_CPU_X16_MXFP4=1 GGML_CPU_X16_Q8_0=1 GGML_CPU_X16_Q8_BATCH=1 GGML_CPU_X16_Q8_EXPERTS=1
  export GGML_CPU_X16_Q4_K=1 GGML_CPU_X16_Q5_K=1 GGML_CPU_X16_Q6_K=1 GGML_CPU_X16_CHUNK_MAX=16
  export GGML_CPU_MOE_GATE_UP_FUSION=1 GGML_CPU_FFN_GATE_UP_FUSION=1 GGML_CPU_ARGSORT_TOP_K=1
  export GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=4096 GGML_CPU_PARALLEL_COPY=1 GGML_CPU_Q8_FAST_SUM=1
  export GGML_CPU_RMS_F64_SIMD=1 GGML_CPU_SOFTMAX_POOL_FUSION=1 GOMP_SPINCOUNT=20000
else
  BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-mimo-v26/build/bin/llama-server
fi

args=(--host 127.0.0.1 --port "$PORT" --model "$MODEL" --alias mimo-v2.6-pro
      --ctx-size "$CTX" --parallel 1 --load-mode "$LOAD" --flash-attn on
      --batch-size "${NBATCH:-2048}" --ubatch-size "${UBATCH:-512}" --jinja --reasoning-format deepseek
      --metrics --no-webui --verbosity 2 --temp 1.0 --top-p 0.95)
# F32 projector (2026-09-22). The F16 one overflows: ViT block 27's SwiGLU output reaches ~1.1e5 on ordinary images,
# the CPU mul_mat converts that activation to F16 (max 65504) for the F16 ffn_down weight, the row becomes inf/NaN and
# the model answers '????'. F32 matches Xiaomi's reference to cos 0.9999 and encodes no slower. See STATE-VISION-20260922.md.
MMPROJ_FILE=${MMPROJ_FILE:-/models/mimo-v26-pro/gguf/mmproj-MiMo-V2.6-Pro-RL-F32.gguf}
[ "${MMPROJ:-0}" = 1 ] && args+=(--mmproj "$MMPROJ_FILE")
[ -n "${IMAGE_MAX_TOKENS:-}" ] && args+=(--image-max-tokens "$IMAGE_MAX_TOKENS")
if [ "$ENGINE" = tp ] && [ "$TP" = 1 ]; then
  args+=(--gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1
         --threads "$THREADS" --threads-batch "$THREADS" --fit off)
else
  args+=(--threads $((THREADS * 4)) --threads-batch $((THREADS * 4)))
fi
if [ "$SPEC" = mtp ]; then
  args+=(--spec-type draft-mtp --spec-draft-n-max "${NMAX:-2}" --spec-draft-p-min 0.0)
  [ "$TP" = 1 ] && args+=(--spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --spec-draft-ngl all)
fi
if [ "$SPEC" = dflash ]; then
  # MiMo's OWN speculative decoder, per its model card: "5-layer SWA MTP drafter (DFlash-style), predicts 7 subsequent
  # tokens per forward pass". The three model.mtp.layers.* blocks inside the main checkpoint are NOT this -- they draft at
  # 4-16% acceptance because they were never the shipped drafter. DFlash is a real second model (2.9 GiB) that reads the
  # target's hidden states at layers 1/16/32/48/70 and shares its embeddings and LM head through ctx_other.
  # n-max is block_size - 1 = 7 (in-place denoising yields at most block_size-1 draft tokens).
  DFLASH_MODEL=${DFLASH_MODEL:-/models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-DFlash-Q8_0.gguf}
  args+=(--spec-type draft-dflash --spec-draft-model "$DFLASH_MODEL"
         --spec-draft-n-max "${NMAX:-7}" --spec-draft-p-min "${PMIN:-0.0}")
  [ "$TP" = 1 ] && args+=(--spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --spec-draft-ngl all)
fi
echo 1000 > /proc/self/oom_score_adj
exec taskset -c 0-127 "$BIN" "${args[@]}" ${EXTRA:-}
