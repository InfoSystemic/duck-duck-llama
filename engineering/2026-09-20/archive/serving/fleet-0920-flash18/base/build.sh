#!/bin/bash
# build.sh [--gate | <meta source> <outdir>] -- private libggml-base: the production glm-fix link with ggml-backend-meta.cpp.o rebuilt.
# Recipe recovered 2026-09-20 (none was recorded): the engine's compile command plus -mavx512f -mavx512bw; --gate reproduces
# glm-fix/ggml-backend-meta.cpp.o and glm-fix/libggml-base.so.0.22.0 (sha256 598563d0...) byte for byte.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
FIX=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix
if [ "${1:-}" = "--gate" ]; then SRC=$FIX/ggml-backend-meta.cpp; OUT=$HERE/gate; else SRC=$(readlink -f "$1"); OUT=$(readlink -f "$2"); fi
mkdir -p "$OUT"; cd $ENG/build-goal/ggml/src
PIN="nice -n 10 taskset -c 15,31,47,63,79,95,111,127"
$PIN /usr/bin/c++ -DGGML_BUILD '-DGGML_COMMIT="unknown"' -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_OPENMP '-DGGML_VERSION="0.22.0"' -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_base_EXPORTS -I$ENG/ggml/src/. -I$ENG/ggml/src/../include -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -fopenmp -o "$OUT"/ggml-backend-meta.cpp.o -c "$SRC" -mavx512f -mavx512bw
$PIN /usr/bin/c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libggml-base.so.0 -o "$OUT"/libggml-base.so.0.22.0 CMakeFiles/ggml-base.dir/ggml.c.o CMakeFiles/ggml-base.dir/ggml.cpp.o CMakeFiles/ggml-base.dir/ggml-alloc.c.o CMakeFiles/ggml-base.dir/ggml-backend.cpp.o "$OUT"/ggml-backend-meta.cpp.o CMakeFiles/ggml-base.dir/ggml-opt.cpp.o CMakeFiles/ggml-base.dir/ggml-threading.cpp.o CMakeFiles/ggml-base.dir/ggml-quants.c.o CMakeFiles/ggml-base.dir/gguf.cpp.o -lm /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a
ln -sf libggml-base.so.0.22.0 "$OUT"/libggml-base.so.0; ln -sf libggml-base.so.0 "$OUT"/libggml-base.so
if [ "${1:-}" = "--gate" ]; then cmp "$OUT"/ggml-backend-meta.cpp.o $FIX/ggml-backend-meta.cpp.o && cmp "$OUT"/libggml-base.so.0.22.0 $FIX/libggml-base.so.0.22.0 && echo "LINEAGE GATE: object and library byte-identical to production glm-fix"; fi
sha256sum "$OUT"/libggml-base.so.0.22.0
