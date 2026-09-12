#!/usr/bin/env bash
# A/B GGML_DSV4_OUTPUT_TP on the 8-layer truncated DeepSeek-V4-Flash. Same binary both arms.
set -uo pipefail
ENG=/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904
BASE=/home/kwebb/InfoSystemic/AI-Server/serving/fleet-0903
PORT=18133
arm() {
  local label="$1" tp="$2"
  local prev; prev=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
  if [[ -n "${prev:-}" ]]; then kill "$prev" 2>/dev/null; for _ in $(seq 1 120); do kill -0 "$prev" 2>/dev/null || break; sleep 1; done; fi
  local ENVARGS=(); while IFS= read -r l; do [[ -n "$l" ]] && ENVARGS+=("$l"); done < "$BASE/results/goal-0910-restore/env.txt"
  nohup env "${ENVARGS[@]}" LD_LIBRARY_PATH="$ENG/build-goal/bin" GGML_DSV4_OUTPUT_TP="$tp" \
    "$ENG/build-goal/bin/llama-server" --host 127.0.0.1 --port $PORT --load-mode mmap --fit off \
    --ctx-size 2048 --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999 \
    --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1 \
    --no-webui --metrics --no-cache-prompt --threads 15 --threads-batch 15 \
    --model /dev/shm/dsv4-8L.gguf --alias dsv4-8L > "$BASE/results/goal-0910-restore/ab-$label.log" 2>&1 &
  for _ in $(seq 1 300); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "$label: FAILED TO START"; tail -12 "$BASE/results/goal-0910-restore/ab-$label.log"; return 1
  fi
  python3 "$BASE/ab_probe_0910.py" "$label" "$tp"
}
arm "output-mirrored" 0
arm "output-split"    1
