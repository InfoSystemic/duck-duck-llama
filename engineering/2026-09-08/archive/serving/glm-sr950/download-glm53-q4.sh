#!/usr/bin/env bash
# Download GLM-5.3 UD-Q4_K_XL (467.3 GB, 11 shards) to /models.
#
# Q4_K_XL is the right tier for THROUGHPUT on this box, not just quality:
#   Q4_K 58.2% + Q5_K 35.1% = 93.3% of the file has a fast interleaved repack
#   path (repack.cpp:5971 q4_K_8x8_q8_K, ungated; q5_K_r8_q8_K behind
#   GGML_CPU_Q5_K_REPACK=1). The resident UD-Q3_K_XL is IQ3_XXS/IQ4_XS-dominated
#   and IQ4_XS has NO repack path at all. Q4 costs +14% bytes/token
#   (33.48 vs 29.28 GB) and buys ~93% kernel coverage.
#
# Goes to DISK deliberately, not /dev/shm. tmpfs is RAM, and numa-tensor
# allocates a second node-bound copy -- a tmpfs-resident 467 GB model would be
# billed twice (934 GB) against 755 GB of RAM. On disk it is loaded once into
# node-bound buffers: 467 GB total, 117 GB per NUMA node against 193 GB each.
set -uo pipefail

REPO=${REPO:-unsloth/GLM-5.3-GGUF}
SUBDIR=${SUBDIR:-UD-Q4_K_XL}
DEST=${DEST:-/models/GLM-5.3-GGUF}
NEED_GB=${NEED_GB:-480}

mkdir -p "$DEST"

avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
echo "destination : $DEST"
echo "free space  : ${avail_gb} GB (need ~${NEED_GB} GB)"
if [ "${avail_gb:-0}" -lt "$NEED_GB" ]; then
  echo
  echo "REFUSING TO START: not enough free space."
  echo "Free space first, then re-run. See:"
  echo "  ~/InfoSystemic/AI-Server/tuning/deepseek-v4/wd750-variants-DELETED-20260828/README.md"
  exit 1
fi

# Keep every cache off the root filesystem -- / has ~25 GB free and the xet
# staging area will happily fill it.
export HF_HOME="$DEST/.hf-home"
export HF_HUB_CACHE="$DEST/.hf-cache"
export HF_XET_CACHE="$DEST/.xet-cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"

echo "starting $(date -Is)"
# The downloader's own pages land wherever it runs. Interleave them so a later
# tensor-split allocation is not blocked by one starved node -- the exact fault
# that OOM-killed the first GLM-5.3 numa-tensor cutover (see
# GLM53-PERFORMANCE-INVESTIGATION.md section 7).
numactl --interleave=all \
  hf download "$REPO" \
    --include "$SUBDIR/*" \
    --local-dir "$DEST" \
    --max-workers 8
rc=$?
echo "hf download exited rc=$rc at $(date -Is)"

echo
echo "=== verifying shard set ==="
n=$(ls -1 "$DEST/$SUBDIR"/GLM-5.3-UD-Q4_K_XL-*-of-00011.gguf 2>/dev/null | wc -l)
echo "shards present: $n / 11"
du -sh "$DEST/$SUBDIR" 2>/dev/null
ls -la "$DEST/$SUBDIR" 2>/dev/null

if [ "$n" -ne 11 ]; then
  echo "INCOMPLETE -- re-run this script, hf download resumes."
  exit 1
fi
echo "COMPLETE"
