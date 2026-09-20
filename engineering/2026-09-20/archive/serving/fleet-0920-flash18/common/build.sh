#!/bin/bash
# build.sh <src.cpp> <outdir> -- private libllama-common: build-goal objects, only speculative.cpp rebuilt from <src>.
set -euo pipefail
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
SRC=$(readlink -f "$1"); OUT=$(readlink -f "$2"); mkdir -p "$OUT"
cd $ENG/build-goal/common
nice -n 10 taskset -c 15,31,47,63,79,95,111,127 /usr/bin/c++ -DCPPHTTPLIB_OPENSSL_SUPPORT -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED -DLLAMA_SUBPROCESS -Dllama_common_EXPORTS -I$ENG/common/. -I$ENG/vendor/nlohmann/.. -I$ENG/vendor/sheredom/.. -I$ENG/src/../include -I$ENG/ggml/src/../include -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -o "$OUT"/speculative.cpp.o -c "$SRC"
OBJS=$(sed -e 's/.*-o ..\/bin\/libllama-common.so.0.3.0 //' -e 's/ -Wl,-rpath.*//' CMakeFiles/llama-common.dir/link.txt | tr -d '"' | sed "s#CMakeFiles/llama-common.dir/speculative.cpp.o#$OUT/speculative.cpp.o#")
nice -n 10 taskset -c 15,31,47,63,79,95,111,127 /usr/bin/c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libllama-common.so.0 -o "$OUT"/libllama-common.so.0.3.0 $OBJS -Wl,-rpath,$ENG/build-goal/bin: libllama-common-base.a ../vendor/cpp-httplib/libcpp-httplib.a ../bin/libllama.so.0.3.0 /usr/lib/x86_64-linux-gnu/libssl.so /usr/lib/x86_64-linux-gnu/libcrypto.so ../bin/libggml.so.0.22.0 ../bin/libggml-cpu.so.0.22.0 ../bin/libggml-base.so.0.22.0
ln -sf libllama-common.so.0.3.0 "$OUT"/libllama-common.so.0; ln -sf libllama-common.so.0 "$OUT"/libllama-common.so
md5sum "$OUT"/libllama-common.so.0.3.0
