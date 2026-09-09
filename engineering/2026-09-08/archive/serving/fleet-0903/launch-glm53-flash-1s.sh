#!/usr/bin/env bash
# GLM-5.3-Flash (glm5next) on ONE socket, node-local, with optional MTP
# speculative decoding from the draft layer that ships inside the GGUF
# (blk.45.nextn.*, which a normal context logs as "unused tensor ... ignoring").
#
# Single socket is a ~95 GB/s ceiling and this model reads ~8.8 GB per token,
# so raw decode cannot exceed ~10.8 tok/s here.  Speculation is what takes it
# past that, because an accepted draft token costs no extra weight pass.
#
# Tensor-parallel across four sockets would raise the ceiling to ~43, but the
# meta backend currently aborts on this architecture -- see
# FLEET-TUNING-20260903.md for the two blockers.
set -u
PORT=${1:-18096}
BIN=${GLM53F_BIN:-/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server}
ROOT=${GLM53F_ROOT:-/models/gguf/GLM-5.3-Flash}
QUANT=${GLM53F_QUANT:-UD-IQ2_XXS}
MODEL="$ROOT/$QUANT/GLM-5.3-Flash-$QUANT-00001-of-00004.gguf"
CTX=${GLM53F_CTX:-32768}
NODE=${GLM53F_NODE:-0}
THREADS=${GLM53F_THREADS:-32}
ALIAS=${GLM53F_ALIAS:-glm53-flash,glm-5.3-flash,GLM-5.3-Flash}

[ -x "$BIN" ]   || { echo "missing binary: $BIN" >&2; exit 1; }
[ -r "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 1; }

SPEC=()
if [ "${GLM53F_SPEC:-0}" != "0" ]; then
  SPEC=(--spec-type "${GLM53F_SPEC_TYPE:-draft-mtp}")
  case "${GLM53F_SPEC_TYPE:-draft-mtp}" in
    ngram*) SPEC+=(--spec-ngram-mod-n-match "${GLM53F_NGRAM_MATCH:-8}"
                   --spec-ngram-mod-n-max "${GLM53F_NGRAM_MAX:-64}"
                   --spec-ngram-mod-n-min "${GLM53F_NGRAM_MIN:-2}") ;;
    *)      SPEC+=(--spec-draft-n-max "${GLM53F_SPEC_N_MAX:-3}"
                   --spec-draft-p-min "${GLM53F_SPEC_P_MIN:-0.2}"
                   --spec-draft-threads "$THREADS" --spec-draft-threads-batch "$THREADS") ;;
  esac
fi

exec env NVIDIA_TF32_OVERRIDE=0 \
  numactl --cpunodebind="$NODE" --membind="$NODE" -- \
  "$BIN" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --alias "$ALIAS" \
  --gpu-layers 0 --load-mode "${GLM53F_LOAD_MODE:-mmap}" \
  --threads "$THREADS" --threads-batch "$THREADS" \
  --ctx-size "$CTX" --flash-attn "${GLM53F_FA:-off}" \
  --batch-size 512 --ubatch-size 256 --parallel 1 --cache-prompt --cache-reuse 256 \
  "${SPEC[@]}" \
  --jinja --reasoning-format deepseek --reasoning-preserve \
  --predict 131072 --temp 1.0 --top-p 0.95 --presence-penalty 0.0 \
  --timeout 3600 --threads-http 4 --cors-origins localhost --no-webui --metrics --log-timestamps
