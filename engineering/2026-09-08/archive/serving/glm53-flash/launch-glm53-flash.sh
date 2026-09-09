#!/usr/bin/env bash
set -euo pipefail

ENGINE="${GLM53_ENGINE:-/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server}"
RAM_MODEL_ROOT=/dev/shm/ai-models/GLM-5.3-Flash
DISK_MODEL_ROOT=/models/gguf/GLM-5.3-Flash
MODEL_ROOT="${GLM53_MODEL_ROOT:-}"
QUANT="${GLM53_QUANT:-}"
PORT="${1:-5830}"
NODE="${GLM53_NUMA_NODE:-0}"
THREADS="${GLM53_THREADS:-32}"
CTX_SIZE="${GLM53_CTX_SIZE:-65536}"

model_complete() {
  local root="$1"
  local quant="$2"
  local file
  for file in \
    "$root/$quant/GLM-5.3-Flash-$quant-00001-of-00004.gguf" \
    "$root/$quant/GLM-5.3-Flash-$quant-00002-of-00004.gguf" \
    "$root/$quant/GLM-5.3-Flash-$quant-00003-of-00004.gguf" \
    "$root/$quant/GLM-5.3-Flash-$quant-00004-of-00004.gguf" \
    "$root/mmproj-F16.gguf"; do
    [[ -r "$file" ]] || return 1
  done
}

case "$QUANT" in
  ""|UD-IQ2_XXS|UD-IQ3_XXS) ;;
  *)
    printf 'Unsupported GLM53_QUANT: %s\n' "$QUANT" >&2
    exit 1
    ;;
esac

select_quant() {
  local root="$1"
  local candidate

  if [[ -n "$QUANT" ]]; then
    model_complete "$root" "$QUANT"
    return
  fi

  for candidate in UD-IQ2_XXS UD-IQ3_XXS; do
    if model_complete "$root" "$candidate"; then
      QUANT="$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -z "$MODEL_ROOT" ]]; then
  if select_quant "$RAM_MODEL_ROOT"; then
    MODEL_ROOT="$RAM_MODEL_ROOT"
  elif disk_source=$(findmnt -n -o SOURCE --target /models 2>/dev/null) &&
      [[ -b "$disk_source" ]] && select_quant "$DISK_MODEL_ROOT"; then
    MODEL_ROOT="$DISK_MODEL_ROOT"
  else
    printf 'GLM-5.3 weights are incomplete in RAM and /models has no live block device.\n' >&2
    exit 1
  fi
else
  if ! select_quant "$MODEL_ROOT"; then
    printf 'Incomplete GLM-5.3 model directory: %s\n' "$MODEL_ROOT" >&2
    exit 1
  fi
fi

MODEL="$MODEL_ROOT/$QUANT/GLM-5.3-Flash-$QUANT-00001-of-00004.gguf"
MMPROJ="$MODEL_ROOT/mmproj-F16.gguf"

for required in "$ENGINE" "$MODEL" "$MMPROJ"; do
  if [[ ! -r "$required" ]]; then
    printf 'Missing required GLM-5.3 file: %s\n' "$required" >&2
    exit 1
  fi
done

# The support PR requires flash attention off for correct GLM5-Next output.
# NVIDIA_TF32_OVERRIDE is harmless on this CPU build and keeps the launcher safe
# if a CUDA build is substituted later.
exec env NVIDIA_TF32_OVERRIDE=0 \
  numactl --cpunodebind="$NODE" --preferred="$NODE" -- \
  "$ENGINE" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --alias glm53-flash,glm-5.3-flash,GLM-5.3-Flash,ox-alpha \
  --gpu-layers 0 --load-mode mmap \
  --threads "$THREADS" --threads-batch "$THREADS" \
  --ctx-size "$CTX_SIZE" --flash-attn off \
  --batch-size 512 --ubatch-size 256 --parallel 1 \
  --jinja --reasoning-format deepseek --reasoning-preserve \
  --predict 131072 --temp 1.0 --top-p 0.95 --presence-penalty 0.0 \
  --timeout 3600 --cors-origins localhost \
  --no-webui --metrics --log-timestamps
