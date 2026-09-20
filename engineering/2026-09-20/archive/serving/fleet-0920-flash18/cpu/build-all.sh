#!/bin/bash
# build-all.sh <outdir> -- private libggml-cpu with ops.cpp, ggml-cpu.cpp AND repack.cpp rebuilt from the f18 sources.
# Gates: cpu/gate (ops + library), cpu/gate-dispatch (ggml-cpu.cpp.o), cpu/gate-repack (repack.cpp.o): each unmodified source
# recompiles to the byte-identical production object.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
PROD_REPACK=/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu/repack.cpp.o
OUT=$(readlink -f "${1:-$HERE/build-all}"); mkdir -p "$OUT"
"$HERE"/build-dispatch.sh "$HERE/ops.f18.cpp" "$HERE/ggml-cpu.f18.cpp" "$OUT" > /dev/null
"$HERE"/build-repack.sh "$HERE/repack.f18.cpp" "$OUT" 2>/dev/null
LINK=$(grep -- '-shared -Wl,-soname,libggml-cpu.so.0' "$HERE"/build.sh | sed -e 's#"\$OUT"#'"$OUT"'#g' \
  -e "s#$ENG/build-goal/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o#$OUT/ggml-cpu.cpp.o#" -e "s#$PROD_REPACK#$OUT/repack.cpp.o#")
echo "$LINK" | grep -q "$OUT/ggml-cpu.cpp.o" && echo "$LINK" | grep -q "$OUT/repack.cpp.o" || { echo "object substitution failed"; exit 1; }
eval "$LINK"
sha256sum "$OUT"/libggml-cpu.so.0.22.0
