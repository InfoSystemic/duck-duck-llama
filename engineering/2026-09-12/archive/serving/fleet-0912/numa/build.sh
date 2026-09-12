#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
exec /usr/bin/c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -I/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/vendor \
  "$D/rebalance_tmpfs.cpp" -lcrypto -o "$D/rebalance-tmpfs"
