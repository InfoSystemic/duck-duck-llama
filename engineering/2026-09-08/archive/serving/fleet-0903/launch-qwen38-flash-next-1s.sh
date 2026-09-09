#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4exp) on ONE socket, node-local, on the fast-stack
# engine: x16 AVX-512-VNNI kernels, repacked IQ2_XS/IQ3_XXS expert paths, MoE
# gate/up fusion, and optional MTP speculative decoding from the sidecar.
#
# Why one socket: measured 2026-09-03, four-socket tensor-parallel gives 7.9
# tok/s vs 6.5 here AND produces garbage (the qwen4exp split rules do not cover
# the SSM state, indexer or hc_* tensors). This model reads only ~4.0 GB per
# token, so it is not bandwidth-bound -- 36 of its 48 layers are linear-attention
# scans and its expert matrices are 2560x640. Extra memory controllers do not
# help; better kernels and speculation do.
#
# HARD CONSTRAINT: f16 KV only. A quantized KV cache trips
#   GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr)
# in qwen4exp.cpp build_attn_qsa, AFTER the weights load cleanly.
#
# --load-mode none is required: mmap'd pages come from the page cache and sit
# wherever they were first read, silently ignoring --membind. That cost 1.79 vs
# 2.90 tok/s on the sibling GLM-5.3-Flash before it was caught.
set -u
PORT=${1:-18095}
BIN=${QWEN4E_BIN:-/dev/shm/q4e-fast/build-fast/bin/llama-server}
ROOT=${QWEN4E_ROOT:-/models/gguf/Qwen3.8-Flash-Next}
MODEL=${QWEN4E_MODEL:-$ROOT/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf}
CTX=${QWEN4E_CTX:-32768}
NODE=${QWEN4E_NODE:-1}
THREADS=${QWEN4E_THREADS:-32}
ALIAS=${QWEN4E_ALIAS:-qwen38-flash-next,qwen3.8-flash-next,flash-next}

[ -x "$BIN" ]   || { echo "missing binary: $BIN" >&2; exit 1; }
[ -r "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 1; }

# Speculation can come from a draft model (draft-mtp) or from the context alone
# (ngram-mod, which needs no draft model at all -- the only option here while the
# published MTP sidecar fails to load against the current qwen4exp definition).
SPEC=()
if [ -n "${QWEN4E_MTP_MODEL:-}" ]; then
  [ -r "$QWEN4E_MTP_MODEL" ] || { echo "unreadable sidecar: $QWEN4E_MTP_MODEL" >&2; exit 1; }
  SPEC=(--model-draft "$QWEN4E_MTP_MODEL"
        --spec-type "${QWEN4E_SPEC_TYPE:-draft-mtp}"
        --spec-draft-n-max "${QWEN4E_SPEC_N_MAX:-3}"
        --spec-draft-p-min "${QWEN4E_SPEC_P_MIN:-0.2}"
        --spec-draft-threads "$THREADS" --spec-draft-threads-batch "$THREADS")
elif [ -n "${QWEN4E_SPEC_TYPE:-}" ]; then
  SPEC=(--spec-type "$QWEN4E_SPEC_TYPE"
        --spec-ngram-mod-n-match "${QWEN4E_NGRAM_MATCH:-8}"
        --spec-ngram-mod-n-max "${QWEN4E_NGRAM_MAX:-64}"
        --spec-ngram-mod-n-min "${QWEN4E_NGRAM_MIN:-2}")
fi

exec env LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  numactl --cpunodebind="$NODE" --membind="$NODE" -- \
  "$BIN" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --alias "$ALIAS" \
  --gpu-layers 0 --load-mode "${QWEN4E_LOAD_MODE:-none}" \
  --threads "$THREADS" --threads-batch "$THREADS" \
  --ctx-size "$CTX" --flash-attn "${QWEN4E_FA:-on}" \
  --batch-size 2048 --ubatch-size 256 --parallel 1 --cache-prompt --cache-reuse 256 \
  "${SPEC[@]}" \
  --jinja --reasoning-format deepseek --reasoning-preserve \
  --predict 32768 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --timeout 3600 --threads-http 4 --cors-origins localhost --no-webui --metrics --log-timestamps
