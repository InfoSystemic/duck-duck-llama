#!/bin/bash
# Pass 1: GGUF header + every non-expert tensor (Q8_0 from FP8/BF16), with sparse holes where the
# 207 stacked MXFP4 expert tensors go. Runs in the background-load pen, away from the llama workers.
set -euo pipefail
cd "$(dirname "$0")/convert"
export MIMO_MXFP4_SKELETON=1
exec taskset -c 15,31,47,63,79,95,111,127 nice -n 10 ionice -c2 -n7 \
  ~/InfoSystemic/AI-Server/engines/llama.cpp-dsv41-tuned-0912/.venv/bin/python convert_hf_to_gguf.py \
  /models/mimo-v26-pro/skel-src --outtype q8_0 --model-name MiMo-V2.6-Pro-RL --split-max-size 48G \
  --outfile /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE.gguf
