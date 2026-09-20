#!/bin/bash
# build-dispatch.sh [ops src] [ggml-cpu.cpp src] [outdir] -- private libggml-cpu with ops.cpp AND ggml-cpu.cpp rebuilt.
# Lineage: gate-dispatch/ggml-cpu.cpp.o (unmodified source, same command) is byte-identical to the object in the deployed library.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
OPS="${1:-$HERE/ops.f18.cpp}"; CPP="${2:-$HERE/ggml-cpu.f18.cpp}"; OUT="${3:-$HERE/build-dispatch}"
mkdir -p "$OUT"
"$HERE"/build.sh "$OPS" "$OUT" > /dev/null
nice -n 10 taskset -c 15,31,47,63,79,95,111,127 /usr/bin/c++ -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I$ENG/ggml/src/.. -I$ENG/ggml/src/. -I$ENG/ggml/src/ggml-cpu -I$ENG/ggml/src/../include -O3 -DNDEBUG -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -march=native -fopenmp -o "$OUT"/ggml-cpu.cpp.o -c "$CPP"
# relink: the recipe's link line with the engine's ggml-cpu.cpp.o swapped for ours
LINK=$(grep -- '-shared -Wl,-soname,libggml-cpu.so.0' "$HERE"/build.sh | sed -e 's#"\$OUT"#'"$OUT"'#g' -e "s#$ENG/build-goal/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o#$OUT/ggml-cpu.cpp.o#")
echo "$LINK" | grep -q "$OUT/ggml-cpu.cpp.o" || { echo "object substitution failed"; exit 1; }
eval "$LINK"
sha256sum "$OUT"/libggml-cpu.so.0.22.0
