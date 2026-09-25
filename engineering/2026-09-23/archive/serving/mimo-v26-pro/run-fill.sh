#!/bin/bash
# Pass 2: stream the 128 expert shards into the skeleton GGUF (resumable; see fill_experts.py).
set -euo pipefail
cd "$(dirname "$0")"
exec taskset -c 15,31,47,63,79,95,111,127 nice -n 10 ionice -c2 -n7 \
  ~/InfoSystemic/AI-Server/engines/llama.cpp-dsv41-tuned-0912/.venv/bin/python fill_experts.py "$@"
