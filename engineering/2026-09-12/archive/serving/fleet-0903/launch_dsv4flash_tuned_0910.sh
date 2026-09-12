#!/usr/bin/env bash
# launch_dsv4flash_tuned_0910.sh - run the existing DeepSeek-V4-Flash-0731 GGUF (arch
# deepseek4, 155 GB, already on /models) on the TUNED glm5n-goal runtime with the meta
# backend + TP4 + x16 kernels. This has never been done: the only prior DeepSeek-V4-Flash
# measurement (2.67 tok/s, 08-30) used a dspark fork that has NO meta backend at all.
# Purpose: predict what DeepSeek-V4.1-Flash could reach before paying for a 510 GB
# download and an Engram port. Same family, same engine, same box.
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
BIN="$BASE/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server"
MODEL=/models/gguf/DeepSeek-V4-Flash-0731-UD-Q4_K_XL/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf
PORT=18132
LOG="$BASE/results/goal-0910-restore/dsv4flash-$(date -u +%H%M%S).log"
SPLIT="${1:-tensor}"     # tensor | none
SPEC="${2:-}"            # e.g. draft-dspark  |  ngram-mod  |  ngram-mod,draft-dspark
DRAFT="${3:-}"           # draft gguf path
NMAX="${4:-2}"
ENVARGS=(); while IFS= read -r l; do [[ -n "$l" ]] && ENVARGS+=("$l"); done < "$BASE/results/goal-0910-restore/env.txt"

ARGS=("$BIN" --host 127.0.0.1 --port $PORT --load-mode mmap --fit off --ctx-size 4096
      --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999
      --jinja --no-webui --metrics --verbosity 2 --no-cache-prompt
      --threads 15 --threads-batch 15 --model "$MODEL" --alias dsv4-flash)
if [[ "$SPLIT" == "tensor" ]]; then
  ARGS+=(--device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1)
fi
if [[ -n "$SPEC" ]]; then
  ARGS+=(--spec-type "$SPEC" --spec-draft-n-max "$NMAX" --spec-draft-p-min 0.0)
  [[ -n "$DRAFT" ]] && ARGS+=(--spec-draft-model "$DRAFT" --spec-draft-ngl all
                              --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3)
fi
prev=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
if [[ -n "${prev:-}" ]]; then echo "stopping previous pid=$prev"; kill "$prev"; for _ in $(seq 1 180); do kill -0 "$prev" 2>/dev/null || break; sleep 1; done; fi
echo "launching split-mode=$SPLIT spec=${SPEC:-none} -> $LOG"
cd "$BASE"
nohup env "${ENVARGS[@]}" "${ARGS[@]}" > "$LOG" 2>&1 &
NEW=$!; echo "pid=$NEW"
for _ in $(seq 1 900); do
  curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "healthy after ${SECONDS}s"; echo $NEW > "$BASE/results/goal-0910-restore/dsv4.pid"; exit 0; }
  kill -0 $NEW 2>/dev/null || { echo "DIED after ${SECONDS}s. tail:"; tail -25 "$LOG"; exit 1; }
  sleep 2
done
echo "timeout"; tail -25 "$LOG"; exit 1
