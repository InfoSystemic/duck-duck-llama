#!/bin/bash
# proxy.sh <tag> [port] -- run the production GLM-5.3-Flash engine stack on the 8-layer proxy model on a private port.
# Env: LIB_PREPEND (private libs first), F18_FEATURES (mask), F18_CONTROL (control file), CTX, NMAX, EXTRA_ARGS, SPEC=0 to disable MTP.
set -euo pipefail
TAG=${1:?tag}; PORT=${2:-18141}
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
MODEL=${MODEL:-/models/f18-proxy/glmflash-q4-8L-nextn.gguf}
CTX=${CTX:-32768}; NMAX=${NMAX:-2}
eval "$(grep '^export ' /home/user/InfoSystemic/AI-Server/serving/fleet-0911/restore-baseline-0911.sh)"
F=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx
# production library stack (mtp-kv-only libllama, kpool-wide cpu, glm-fix base), production env
export LD_LIBRARY_PATH=$F/glm-mtp-kv-only-0919/deploy:$F/glm-kv-range-0919/deploy-r2:$F/glm-kpool-wide-0919/deploy:$F/glm-fix:$F/unary-lib:$LD_LIBRARY_PATH
export GGML_CPU_PARALLEL_UNARY=4096 GGML_META_REDUCE_NT=1 GGML_CPU_GLM_POOL_FUSION=1 GGML_CPU_CPY_FLAT=1 GGML_CPU_GLM_POOL_WIDE=1 LLAMA_KV_SEQ_RM_USED_PREFIX=1 GGML_GLM5N_MTP_KV_ONLY=1
[ -n "${LIB_PREPEND:-}" ] && export LD_LIBRARY_PATH="$LIB_PREPEND:$LD_LIBRARY_PATH"
[ -n "${F18_FEATURES:-}" ] && export GGML_F18_FEATURES="$F18_FEATURES"
[ -n "${F18_CONTROL:-}" ] && export GGML_F18_CONTROL_FILE="$F18_CONTROL"
export GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/f18-optrace.arm GGML_CPU_OP_PROFILE_COUNT=${PROFILE_COUNT:-8} GGML_CPU_OP_PROFILE_SKIP=${PROFILE_SKIP:-4}
export LLAMA_GRAPH_PHASE_ARM_FILE=/dev/shm/f18-graph-phase.arm
SPEC_ARGS=()
if [ "${SPEC:-1}" = "1" ]; then
  SPEC_ARGS=(--spec-type draft-mtp --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --spec-draft-ngl all --spec-draft-p-min ${PMIN:-0.0} --spec-draft-n-max $NMAX
    --spec-draft-model /home/user/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9/MTP/GLM-5.3-Flash-MTP-Q8_0-621d456e93e9.gguf)
fi
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 2; }
LOG=$W/run/proxy-$TAG.log
( echo 1000 > /proc/self/oom_score_adj; exec taskset -c 0-127 /home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server \
  --host 127.0.0.1 --port $PORT --load-mode mmap --fit off --ctx-size $CTX --parallel 1 --cache-prompt --flash-attn on \
  --batch-size 2048 --ubatch-size 1024 --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1 \
  --jinja --reasoning-format deepseek --no-webui --metrics --verbosity 2 "${SPEC_ARGS[@]}" --threads 15 --threads-batch 15 \
  --model $MODEL --alias f18-proxy ${EXTRA_ARGS:-} ) > $LOG 2>&1 &
echo "launched tag=$TAG port=$PORT log=$LOG"
for i in $(seq 600); do
  s=$(curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null || true); [[ "$s" == *'"ok"'* ]] && { echo "healthy after ${i}s"; exit 0; }
  ss -ltn | grep -q ":$PORT " || { [ $i -gt 5 ] && ! pgrep -f -- "--port $PORT --load-mode" >/dev/null && { echo DIED; tail -20 $LOG; exit 1; }; }
  sleep 1
done
echo TIMEOUT; exit 1
