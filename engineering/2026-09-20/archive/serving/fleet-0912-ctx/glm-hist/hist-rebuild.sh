#!/bin/bash
# Build glm-hist/libggml-cpu: unary-lib content + TOPK_HIST probe. Adapted from
# fleet-0911/parallel-unary-0911/rebuild.sh: same flags, same parent objects,
# only ops.cpp.o comes from ops.hist.cpp (ops.patched.cpp + probe).
set -e
D="$(cd "$(dirname "$0")" && pwd)"
SRC_ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
R0903=/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results
OUT="${1:-$D/build}"
mkdir -p "$OUT"
PU=/home/user/InfoSystemic/AI-Server/serving/fleet-0911/parallel-unary-0911

echo "--- ggml-cpu.c (parent: parallel-unary-0911, unchanged) ---"
/usr/bin/cc -I"$PU" -I$R0903/glm-flash-q8-pool-0908c/private-cpu -I$SRC_ENG/ggml/src/ggml-cpu -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I$SRC_ENG/ggml/src/.. -I$SRC_ENG/ggml/src/. -I$SRC_ENG/ggml/src/ggml-cpu -I$SRC_ENG/ggml/src/../include -O3 -DNDEBUG -std=gnu11 -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -march=native -fopenmp -o "$OUT"/ggml-cpu.c.o -c "$PU"/ggml-cpu.patched.c

echo "--- ops.hist.cpp (ops.patched.cpp + TOPK_HIST probe) ---"
/usr/bin/c++ -I"$D" -I$R0903/glm-flash-rms-guard-0908/private-cpu -I$R0903/glm-flash-q8-pool-0908c/private-cpu -I$SRC_ENG/ggml/src/ggml-cpu -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I$SRC_ENG/ggml/src/.. -I$SRC_ENG/ggml/src/. -I$SRC_ENG/ggml/src/ggml-cpu -I$SRC_ENG/ggml/src/../include -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -march=native -fopenmp -o "$OUT"/ops.hist.cpp.o -c "$D"/ops.hist.cpp

echo "--- relink ---"
BG=$SRC_ENG/build-goal/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu
/usr/bin/c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libggml-cpu.so.0 -o "$OUT"/libggml-cpu.so.0.22.0 "$OUT"/ggml-cpu.c.o $BG/ggml-cpu.cpp.o $R0903/glm-flash-q8-r8-ordered-k-0908/private-cpu/repack.cpp.o $BG/hbm.cpp.o $BG/quants.c.o $BG/traits.cpp.o $BG/amx/amx.cpp.o $BG/amx/mmq.cpp.o $BG/binary-ops.cpp.o $BG/unary-ops.cpp.o $BG/vec.cpp.o "$OUT"/ops.hist.cpp.o $BG/llamafile/sgemm.cpp.o $BG/arch/x86/quants.c.o $R0903/glm-flash-q8-sum16-0908/private-cpu/repack-x86.cpp.o -Wl,-rpath,$SRC_ENG/build-goal/bin: $SRC_ENG/build-goal/bin/libggml-base.so.0.22.0 /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a

ln -sf libggml-cpu.so.0.22.0 "$OUT"/libggml-cpu.so.0
ln -sf libggml-cpu.so.0 "$OUT"/libggml-cpu.so
echo "built: $OUT/libggml-cpu.so.0.22.0"
