#!/bin/bash
# cfuse_parity.sh <libdir> <tag> -- 8-layer proxy at 64K context with the given libggml-cpu directory prepended; greedy 64-token
# answers at ~8K and ~41K context (depth_trace.py histories, cold prefill allowed); prints the text hashes and rates.
# Run once with deploy-0920f and once with cpu/build-cfuse: the hashes must match (the fusion change is read-order only).
set -euo pipefail
LIBDIR=$(readlink -f "$1"); TAG=$2
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
CTX=65536 LIBS="$LIBDIR" timeout 600 "$W"/run/proxy_revd.sh "parity-$TAG" GGML_CPU_GLM_POOL_PROBE=1 | tail -1
# CONTROL=<mask> overrides the f18 control word after launch (proxy_revd.sh writes 7): e.g. CONTROL=31 enables the Q5 prefetch and
# the fast score kernel, matching a GGML_F18_FEATURES=31 production drop-in
if [ -n "${CONTROL:-}" ]; then python3 -c "import struct,sys; open('/dev/shm/f18-control.u32','wb').write(struct.pack('<I', int(sys.argv[1])))" "$CONTROL"; echo "control word $CONTROL"; fi
for turns in 1 5; do
  python3 "$W"/run/depth_trace.py --port 18141 --turns $turns --tag "parity-$TAG-t$turns" --max-prefill 100000 --gen 64 --arm-after 0.5 2>&1 \
    | grep -a "text sha\|^depth" | sed "s/^/t$turns /"
done
echo "pool probe lines: $(grep -a -c 'GLM_POOL_REJECT' "$W"/run/proxy-parity-$TAG.log) rejects, $(grep -a -c 'GLM_POOL_WIDE' "$W"/run/proxy-parity-$TAG.log) wide"
"$W"/run/stop-port.sh 18141 > /dev/null
