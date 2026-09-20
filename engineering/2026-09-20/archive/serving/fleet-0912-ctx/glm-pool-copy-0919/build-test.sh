#!/usr/bin/env bash
set -euo pipefail
D=$(cd -- "$(dirname -- "$0")" && pwd)
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
for TEST in pool copy; do
    c++ -std=c++17 -O2 -I"$ENG/ggml/include" "$D/test_${TEST}.cpp" -L"$D/build" -L"$ENG/validated-chunk16-bin" -lggml-cpu -lggml-base -ldl -o "$D/test_${TEST}"
done
if [ "${1:-}" != "--compile-only" ]; then python3 "$D/test.py"; fi
