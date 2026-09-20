#!/bin/bash
# build2.sh <speculative.cpp> <sampling.cpp> <outdir> -- private libllama-common: build-goal objects with speculative.cpp AND sampling.cpp rebuilt.
# Lineage: gate/sampling.cpp.o (unmodified source) is byte-identical to the build-goal object; build.sh reproduces the production library md5.
set -euo pipefail
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
HERE=$(dirname "$(readlink -f "$0")")
SPEC=$(readlink -f "$1"); SAMP=$(readlink -f "$2"); OUT=$(readlink -f "$3"); mkdir -p "$OUT"
cd $ENG/build-goal/common
CXX="nice -n 10 taskset -c 15,31,47,63,79,95,111,127 /usr/bin/c++"
FLAGS="-DCPPHTTPLIB_OPENSSL_SUPPORT -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED -DLLAMA_SUBPROCESS -Dllama_common_EXPORTS -I$HERE -I$ENG/common/. -I$ENG/vendor/nlohmann/.. -I$ENG/vendor/sheredom/.. -I$ENG/src/../include -I$ENG/ggml/src/../include -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi"
$CXX $FLAGS -o "$OUT"/speculative.cpp.o -c "$SPEC" &
$CXX $FLAGS -o "$OUT"/sampling.cpp.o    -c "$SAMP" &
wait
OBJS=$(sed -e 's/.*-o ..\/bin\/libllama-common.so.0.3.0 //' -e 's/ -Wl,-rpath.*//' CMakeFiles/llama-common.dir/link.txt | tr -d '"' \
  | sed -e "s#CMakeFiles/llama-common.dir/speculative.cpp.o#$OUT/speculative.cpp.o#" -e "s#CMakeFiles/llama-common.dir/sampling.cpp.o#$OUT/sampling.cpp.o#")
echo "$OBJS" | tr ' ' '\n' | grep -c "$OUT/" | grep -qx 2 || { echo "object substitution failed"; exit 1; }
$CXX -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libllama-common.so.0 -o "$OUT"/libllama-common.so.0.3.0 $OBJS -Wl,-rpath,$ENG/build-goal/bin: libllama-common-base.a ../vendor/cpp-httplib/libcpp-httplib.a ../bin/libllama.so.0.3.0 /usr/lib/x86_64-linux-gnu/libssl.so /usr/lib/x86_64-linux-gnu/libcrypto.so ../bin/libggml.so.0.22.0 ../bin/libggml-cpu.so.0.22.0 ../bin/libggml-base.so.0.22.0
ln -sf libllama-common.so.0.3.0 "$OUT"/libllama-common.so.0; ln -sf libllama-common.so.0 "$OUT"/libllama-common.so
sha256sum "$OUT"/libllama-common.so.0.3.0
