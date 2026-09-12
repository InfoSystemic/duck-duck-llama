#!/usr/bin/env bash
set -uo pipefail
ENG=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
BASE=/home/user/InfoSystemic/AI-Server/serving/fleet-0903
PORT=18132
DRAFT=/models/deepseek-v4-tuning/DSV4-Flash-DSpark-draft-DS3-IQ3S-protected.gguf
MODEL=/models/gguf/DeepSeek-V4-Flash-0731-UD-Q4_K_XL/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf
arm() {
  local label="$1" tp="$2"
  local prev; prev=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
  if [[ -n "${prev:-}" ]]; then kill "$prev" 2>/dev/null; for _ in $(seq 1 200); do kill -0 "$prev" 2>/dev/null || break; sleep 1; done; fi
  local ENVARGS=(); while IFS= read -r l; do [[ -n "$l" ]] && ENVARGS+=("$l"); done < "$BASE/results/goal-0910-restore/env.txt"
  nohup env "${ENVARGS[@]}" LD_LIBRARY_PATH="$ENG/build-goal/bin" GGML_DSV4_OUTPUT_TP="$tp" \
    "$ENG/build-goal/bin/llama-server" --host 127.0.0.1 --port $PORT --load-mode mmap --fit off \
    --ctx-size 4096 --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999 \
    --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1 \
    --jinja --no-webui --metrics --no-cache-prompt --threads 15 --threads-batch 15 \
    --spec-type ngram-mod,draft-dspark --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
    --spec-draft-model "$DRAFT" --spec-draft-ngl all \
    --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
    --model "$MODEL" --alias dsv4-flash > "$BASE/results/goal-0910-restore/fullab-$label.log" 2>&1 &
  for _ in $(seq 1 450); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "$label FAILED"; tail -12 "$BASE/results/goal-0910-restore/fullab-$label.log"; return 1; }
  python3 "$BASE/full_probe_0910.py" "$label" "$tp"
}
arm "mirrored" 0
arm "split"    1
