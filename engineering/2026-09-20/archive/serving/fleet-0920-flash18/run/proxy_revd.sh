#!/bin/bash
# proxy_revd.sh <tag> [KEY=VALUE ...] -- 8-layer proxy in the revision d configuration (+ extra environment), spec control word mapped
set -euo pipefail
TAG=$1; shift
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
python3 - <<'PY'
import struct
for f, v in (('/dev/shm/f18-control.u32', 7), ('/dev/shm/f18-poolcache.u32', 1), ('/dev/shm/f18-spec.u32', 127), ('/dev/shm/f18-qrows.u32', 1)):
    open(f, 'wb').write(struct.pack('<I', v))
PY
$W/run/stop-port.sh 18141 > /dev/null
env GGML_F18_SPEC_CONTROL_FILE=/dev/shm/f18-spec.u32 GGML_CPU_GLM_POOL_CACHE=1 GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE=/dev/shm/f18-poolcache.u32 \
    LLAMA_F18_MTP_QROWS=1 LLAMA_F18_MTP_QROWS_CONTROL_FILE=/dev/shm/f18-qrows.u32 GGML_F18_MTP_PAD_MAX_CTX=1048576 \
    GOMP_SPINCOUNT=20000 GGML_CPU_NUMA_SHARED_TEAM=1 "$@" \
    LIB_PREPEND=${LIBS:-$W/deploy-0920d} F18_CONTROL=/dev/shm/f18-control.u32 $W/run/proxy.sh "$TAG" 18141 | tail -1
