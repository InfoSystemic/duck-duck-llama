#!/bin/bash
# rq_window.sh <tag> <libdir> [K=V ...] -- a full-model window on :18141 (32K ctx, one slot) with the production drop-in environment
# from the given deploy directory plus extra variables, e.g. the dense requant knobs:
#   run/rq_window.sh rq6 deploy-0920h GGML_CPU_ATTN_REQUANT=q6_K GGML_CPU_SHEXP_REQUANT=q6_K GGML_CPU_OUTPUT_REQUANT=q6_K GGML_CPU_DENSE_FFN_REQUANT=q6_K
# Runs quick_rate.py greedy x3 and sampled x3 (rate, ms/cycle, text hash), then stops the instance. Never run during a
# production measurement: the second instance shares the memory bus. Needs ~215 GB free RAM.
set -euo pipefail
TAG=$1; LIBDIR=$(readlink -f "$2"); shift 2
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
CONF="$LIBDIR/95-f18-0920.conf.proposed"
# the drop-in's Environment= lines except LIB_PREPEND (full.sh composes it from CPU_LIB/LLAMA_LIB/COMMON_LIB + the 0919 deploys)
ENVS=$(grep '^Environment=' "$CONF" | sed 's/^Environment=//' | grep -v '^LIB_PREPEND=' | tr '\n' ' ')
export EXTRA_ENV="$ENVS $*"
export CPU_LIB="$LIBDIR" LLAMA_LIB="$LIBDIR" COMMON_LIB="$LIBDIR" CTX=32768 PARALLEL=1 PORT=18141
echo "window $TAG: libs $LIBDIR; extra: $*"
"$W"/run/stop-port.sh 18141 > /dev/null
timeout 900 "$W"/run/full.sh "$TAG" | tail -1
python3 "$W"/run/quick_rate.py --port 18141 --reps 3 --tag "$TAG-greedy" | grep -v "^{"
python3 "$W"/run/quick_rate.py --port 18141 --reps 3 --sampled --tag "$TAG-sampled" | grep -v "^{"
grep -a -i "requant\|x16.*requant" "$W"/run/full-$TAG.log | head -5 || true
"$W"/run/stop-port.sh 18141 > /dev/null
