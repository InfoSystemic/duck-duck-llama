#!/bin/bash
# build-all-cfuse.sh <outdir> -- build-all.sh plus the C compute object (ggml-cpu.c.o) rebuilt from cpu/cfuse/ggml-cpu.combo.c,
# which carries the pool-fusion matcher (pool-fusion.inc). Lineage: the unpatched copy compiled with this exact command and
# -ffile-prefix-map reproduces glm-pool-copy-0919/build/ggml-cpu.c.o byte for byte (sha 1c80cc478a66327e..., checked 2026-09-20).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
PROD_REPACK=/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu/repack.cpp.o
ORIG=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-pool-copy-0919
PROD_C=$ORIG/build/ggml-cpu.c.o
OUT=$(readlink -f "${1:-$HERE/build-all-cfuse}"); mkdir -p "$OUT"
D=$HERE/cfuse
# 1. the C compute object from the (patched) copy, with the recorded command; source paths mapped to the original location
cmd=$(sed -n 10p "$ORIG"/rebuild.sh)
gcmd=$(echo "$cmd" | sed -e "s#\"\$D\"#$D#g" -e "s#\"\$OUT\"#$OUT#g" -e 's#\$D/#'"$D"'/#g' -e 's#\$OUT/#'"$OUT"'/#g' \
  -e "s#/usr/bin/cc #nice -n 10 taskset -c 15,31,47,63,79,95,111,127 /usr/bin/cc -ffile-prefix-map=$D=$ORIG #")
echo "$gcmd" | grep -q -- "-o $OUT/ggml-cpu.c.o" || { echo "C compile command not recognised"; exit 1; }
(cd "$D" && eval "$gcmd") 2>&1 | grep -v "Wcast-qual\|^ *[0-9]* |\|^ *|\|\^\|In function" || true
[ -f "$OUT/ggml-cpu.c.o" ] || { echo "ggml-cpu.c.o not built"; exit 1; }
# 2. ops + ggml-cpu.cpp + repack objects as build-all.sh does
"$HERE"/build-dispatch.sh "$HERE/ops.f18.cpp" "$HERE/ggml-cpu.f18.cpp" "$OUT" > /dev/null
"$HERE"/build-repack.sh "$HERE/repack.f18.cpp" "$OUT" 2>/dev/null
# 3. link with all four objects substituted
LINK=$(grep -- '-shared -Wl,-soname,libggml-cpu.so.0' "$HERE"/build.sh | sed -e 's#"\$OUT"#'"$OUT"'#g' \
  -e "s#$ENG/build-goal/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o#$OUT/ggml-cpu.cpp.o#" -e "s#$PROD_REPACK#$OUT/repack.cpp.o#" \
  -e "s#$PROD_C#$OUT/ggml-cpu.c.o#")
for o in ggml-cpu.cpp.o repack.cpp.o ggml-cpu.c.o; do echo "$LINK" | grep -q "$OUT/$o" || { echo "object substitution failed: $o"; exit 1; }; done
eval "$LINK"
sha256sum "$OUT"/libggml-cpu.so.0.22.0 "$OUT"/ggml-cpu.c.o "$OUT"/ops.cpp.o | awk '{print substr($1,1,16), $2}'
