#!/bin/bash
# MiMo-V2.6-Pro-RL (1.02T MoE / 42B active; MXFP4 experts + Q8_0 rest, 518 GiB) on upstream llama.cpp, CPU-only.
# RAM-EXCLUSIVE: needs glm53-flash-production stopped AND the Flash tmpfs payload evicted
# (window-mimo-smoke.sh does both and restores Flash afterwards).
#   env PORT (18190) CTX (131072) SPEC (none|mtp) NMAX (MTP draft depth, 2) THREADS (60) LOAD (none) EXTRA (more flags)
# LOAD=none (plain sequential read()) is deliberate: the CPU backend repacks all 495 GiB of MXFP4 experts into
# anonymous memory (mxfp4_8x8_q8_0), and under mmap + --numa (MADV_RANDOM) the loader faulted evicted pages back
# 4 KB at a time (~18 MB/s) -- the 09-21 first load stalled at 87 of ~530 GB after 14 min.
set -euo pipefail
PORT=${PORT:-18190}
CTX=${CTX:-131072}
SPEC=${SPEC:-none}
BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-mimo-v26/build/bin/llama-server
M=/models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE-00001-of-00013.gguf
MM=/models/mimo-v26-pro/gguf/mmproj-MiMo-V2.6-Pro-RL-F16.gguf
# the 60 fleet worker cores; 15/31/47/63 (+ HT siblings) stay the background pen
WORKERS=0-14,16-30,32-46,48-62

args=(--host 127.0.0.1 --port "$PORT" --model "$M" --mmproj "$MM" --alias mimo-v2.6-pro
      --ctx-size "$CTX" --parallel 1 --threads "${THREADS:-60}" --threads-batch "${THREADS:-60}"
      --load-mode "${LOAD:-none}" --numa numactl --flash-attn auto --batch-size 2048 --ubatch-size 512
      --jinja --reasoning-format deepseek --metrics --no-webui --verbosity 2
      --temp 1.0 --top-p 0.95)
D=$(cd "$(dirname "$0")" && pwd)
# a window can drop its MTP phase by creating SKIP-MTP (each launch is a full ~26 min read + MXFP4 repack)
if [ "$SPEC" = mtp ] && [ -e "$D/SKIP-MTP" ]; then echo "SPEC=mtp launch skipped: $D/SKIP-MTP exists"; exit 3; fi
if [ "$SPEC" = mtp ]; then
  # the 3 nextn layers live inside the main GGUF; draft-mtp builds its context from the target model
  args+=(--spec-type draft-mtp --spec-draft-n-max "${NMAX:-2}" --spec-draft-p-min 0.0)
fi
# page cache for the 518 GiB mmap interleaved over the 4 nodes; the server is the one to OOM, never the desktop
echo 1000 > /proc/self/oom_score_adj
exec numactl --interleave=all taskset -c "$WORKERS" "$BIN" "${args[@]}" ${EXTRA:-}
