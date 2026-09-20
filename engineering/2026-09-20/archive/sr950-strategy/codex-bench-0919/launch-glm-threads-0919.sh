#!/bin/bash
# GLM-5.3-Flash UD-Q4_K_XL production launcher — NATIVE CONTEXT edition (2026-09-12).
# Identical binary + env + flags to fleet-0911/restore-baseline-0911.sh (the validated 15-16 tok/s
# recipe) EXCEPT the context/caching/slot flags below, which are parameterised. Sized on the 4-layer
# proxy (fleet-0912-ctx/results/kvsize-*): the full model at ctx 1M costs ~22 GB over the 4K figure.
#   CTX        context tokens            (default 1048576 = n_ctx_train)
#   PARALLEL   server slots              (default 2, unified KV so either slot may use the whole window)
#   BATCH/UBATCH prompt-processing batch (default from the prefill sweep)
#   CACHE_RAM  MiB of saved prompt states (default 32768)
set -euo pipefail
CTX=${CTX:-1048576}; PARALLEL=${PARALLEL:-2}; BATCH=${BATCH:-2048}; UBATCH=${UBATCH:-1024}   # 09-14: +8-9% prefill (76-78 vs 70-71 tok/s at 2.25K), decode unchanged, window-flash-pp
CACHE_RAM=${CACHE_RAM:-32768}; CKPT=${CKPT:-128}; CKPT_STEP=${CKPT_STEP:-2048}; PORT=${PORT:-18131}
eval "$(grep '^export ' /home/user/InfoSystemic/AI-Server/serving/fleet-0911/restore-baseline-0911.sh)"
# Temporary thread-scaling experiment; normal launcher is unchanged.
export GGML_CPU_NUMA_THREADS=16
# UNARY=1 (default): fleet-0911 parallel UNARY/SCALE library, bit-exact, +2.1% measured 09-11
if [ "${UNARY:-1}" = "1" ]; then export LD_LIBRARY_PATH=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/unary-lib:$LD_LIBRARY_PATH; export GGML_CPU_PARALLEL_UNARY=4096; fi
# NT=1 (default): non-temporal stores in the cross-socket all-reduce (glm-fix/libggml-base, the kernel promoted for Qwen in
# window 44). Bit-identical: parity IDENTICAL on p2, +2% single stream and +3.4% two-stream aggregate (09-14). NT=0 to disable.
if [ "${NT:-1}" = "1" ] && [ -f /home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix/libggml-base.so.0 ]; then
  export LD_LIBRARY_PATH=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix:$LD_LIBRARY_PATH
  export GGML_META_REDUCE_NT=1
fi
# LIB_PREPEND=DIR: a private library directory searched first (A/B of CPU-library variants, e.g. glm-barrier/build)
[ -n "${LIB_PREPEND:-}" ] && export LD_LIBRARY_PATH="$LIB_PREPEND:$LD_LIBRARY_PATH"
KVU=(); [ "$PARALLEL" -gt 1 ] && KVU=(--kv-unified)
# per-node decode trace hook (inert until /dev/shm/flash-optrace.arm is created; parse with optrace.py)
export GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/flash-optrace.arm GGML_CPU_OP_PROFILE_COUNT=8 GGML_CPU_OP_PROFILE_SKIP=4
exec taskset -c 0-127 /home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server \
  --host 127.0.0.1 --port "$PORT" --load-mode mmap --fit off \
  --ctx-size "$CTX" --parallel "$PARALLEL" "${KVU[@]}" \
  --cache-prompt --cache-ram "$CACHE_RAM" --ctx-checkpoints "$CKPT" --checkpoint-min-step "$CKPT_STEP" \
  --flash-attn on --batch-size "$BATCH" --ubatch-size "$UBATCH" --gpu-layers 999 \
  --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1 \
  --jinja --reasoning-format deepseek --reasoning-preserve --no-webui --metrics --verbosity 2 \
  --spec-type draft-mtp --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --spec-draft-ngl all \
  --spec-draft-p-min 0.0 --spec-draft-n-max 2 \
  --spec-draft-model /home/user/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9/MTP/GLM-5.3-Flash-MTP-Q8_0-621d456e93e9.gguf \
  --threads 15 --threads-batch 15 \
  --model /home/user/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --alias glm-flash-goal,glm-flash-q4,GLM-5.3-Flash,glm-5.3-flash ${EXTRA_ARGS:-}
