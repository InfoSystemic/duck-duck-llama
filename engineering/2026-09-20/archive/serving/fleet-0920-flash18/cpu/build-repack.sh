#!/bin/bash
# build-repack.sh [--gate | <repack source> <outdir>] -- compile repack.cpp with the recorded production command
# (glm-flash-q8-r8-ordered-k-0908/private-cpu/manifest.json). --gate recompiles the unmodified source and compares the object.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROD=/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
if [ "${1:-}" = "--gate" ]; then SRC=$PROD/repack.cpp; OUT=$HERE/gate-repack; else SRC=$(readlink -f "$1"); OUT=$(readlink -f "$2"); fi
mkdir -p "$OUT"
MAP=""; [ "$SRC" != "$PROD/repack.cpp" ] && MAP="-fmacro-prefix-map=$SRC=$PROD/repack.cpp"
nice -n 10 taskset -c 15,31,47,63,79,95,111,127 /usr/bin/c++ -I$PROD -I$ENG/ggml/src/ggml-cpu -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I$ENG/ggml/src/.. -I$ENG/ggml/src/. -I$ENG/ggml/src/ggml-cpu -I$ENG/ggml/src/../include -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -march=native -fopenmp $MAP -o "$OUT"/repack.cpp.o -c "$SRC"
if [ "${1:-}" = "--gate" ]; then cmp "$OUT"/repack.cpp.o $PROD/repack.cpp.o && echo "LINEAGE GATE: repack.cpp.o byte-identical to the production object"; fi
