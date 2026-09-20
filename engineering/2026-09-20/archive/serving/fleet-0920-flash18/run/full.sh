#!/bin/bash
# full.sh <tag> -- full GLM-5.3-Flash on private port 18141 with the production flags/env plus the f18 candidate libraries.
# Env: CPU_LIB (dir with libggml-cpu), LLAMA_LIB (dir with libllama), COMMON_LIB (dir with libllama-common / server-impl), NMAX, PMIN, EXTRA_ENV="K=V ..."
set -euo pipefail
TAG=${1:?tag}
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
F=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx
PROD_LIBS=$F/glm-mtp-kv-only-0919/deploy:$F/glm-kv-range-0919/deploy-r2:$F/glm-kpool-wide-0919/deploy
PRE=""
for d in "${COMMON_LIB:-}" "${LLAMA_LIB:-}" "${CPU_LIB:-}"; do [ -n "$d" ] && PRE="$PRE$d:"; done
export LIB_PREPEND="${PRE}${PROD_LIBS}"
export CTX=${CTX:-1048576} PARALLEL=${PARALLEL:-2} BATCH=2048 UBATCH=1024 PORT=${PORT:-18141}
export GGML_CPU_GLM_POOL_FUSION=1 GGML_CPU_CPY_FLAT=1 GGML_CPU_GLM_POOL_WIDE=1 LLAMA_KV_SEQ_RM_USED_PREFIX=1 GGML_GLM5N_MTP_KV_ONLY=1
export LLAMA_GRAPH_PHASE_ARM_FILE=/dev/shm/f18-graph-phase.arm
export GGML_F18_CONTROL_FILE=/dev/shm/f18-control.u32
export GGML_CPU_GLM_POOL_CACHE=1 GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE=/dev/shm/f18-poolcache.u32
for kv in ${EXTRA_ENV:-}; do export "$kv"; done
[ -f /dev/shm/f18-control.u32 ]   || python3 -c "import struct; open('/dev/shm/f18-control.u32','wb').write(struct.pack('<I',0))"
[ -f /dev/shm/f18-poolcache.u32 ] || python3 -c "import struct; open('/dev/shm/f18-poolcache.u32','wb').write(struct.pack('<I',0))"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 2; }
LOG=$W/run/full-$TAG.log
# the production launcher pins --spec-draft-n-max 2 / p-min 0.0; later flags win in llama.cpp arg parsing
EA=""; [ -n "${NMAX:-}" ] && EA="$EA --spec-draft-n-max $NMAX"; [ -n "${PMIN:-}" ] && EA="$EA --spec-draft-p-min $PMIN"
export EXTRA_ARGS="$EA ${EXTRA_ARGS:-}"
setsid nohup $F/launch-glm-flash-native.sh > $LOG 2>&1 < /dev/null &
echo "launched full tag=$TAG port=$PORT log=$LOG"
for i in $(seq 1200); do
  s=$(curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null || true); [[ "$s" == *'"ok"'* ]] && { echo "healthy after ${i}s"; exit 0; }
  if [ $i -gt 10 ] && ! ss -ltn | grep -q ":$PORT "; then echo "DIED"; tail -20 $LOG; exit 1; fi
  sleep 1
done
echo TIMEOUT; exit 1
