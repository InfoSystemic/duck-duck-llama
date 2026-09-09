#!/usr/bin/env bash
set -euo pipefail

config_file="${GLM_SR950_CONFIG:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.env}"
if [[ ! -r "$config_file" ]]; then
  echo "GLM configuration is not readable: $config_file" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$config_file"
set +a

port="${1:-${GLM_PORT:-18091}}"
for required_var in GLM_BINARY GLM_MODEL GLM_HOST GLM_CONTEXT_SIZE GLM_BATCH_SIZE GLM_UBATCH_SIZE GLM_THREADS GLM_CACHE_TYPE_K GLM_CACHE_TYPE_V GLM_ALIASES; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Missing required setting: $required_var" >&2
    exit 1
  fi
done
if [[ ! -x "$GLM_BINARY" ]]; then
  echo "GLM server binary is not executable: $GLM_BINARY" >&2
  exit 1
fi
if [[ ! -r "$GLM_MODEL" ]]; then
  echo "GLM model is not readable: $GLM_MODEL" >&2
  exit 1
fi

speculative_args=()
spec_type="${GLM_SPEC_TYPE:-none}"
if [[ -n "$spec_type" && "$spec_type" != "none" ]]; then
  speculative_args=(--spec-type "$spec_type")
  IFS=',' read -r -a spec_parts <<<"$spec_type"
  for spec_part in "${spec_parts[@]}"; do
    case "$spec_part" in
      draft-mtp)
        speculative_args+=(
          --spec-draft-n-max "${GLM_SPEC_DRAFT_N_MAX:-3}"
          --spec-draft-n-min "${GLM_SPEC_DRAFT_N_MIN:-0}"
          --spec-draft-p-min "${GLM_SPEC_DRAFT_P_MIN:-0.0}"
          --spec-draft-type-k "${GLM_SPEC_CACHE_TYPE_K:-q8_0}"
          --spec-draft-type-v "${GLM_SPEC_CACHE_TYPE_V:-q8_0}"
          --spec-draft-threads "${GLM_SPEC_THREADS:-$GLM_THREADS}"
          --spec-draft-threads-batch "${GLM_SPEC_THREADS_BATCH:-$GLM_THREADS}"
        )
        if [[ -n "${GLM_SPEC_DRAFT_N_DEFAULT:-}" ]]; then
          speculative_args+=(--spec-draft-n-default "$GLM_SPEC_DRAFT_N_DEFAULT")
        fi
        if [[ -n "${GLM_MTP_MODEL:-}" ]]; then
          if [[ ! -r "$GLM_MTP_MODEL" ]]; then
            echo "GLM MTP draft model is not readable: $GLM_MTP_MODEL" >&2
            exit 1
          fi
          speculative_args+=(
            --spec-draft-model "$GLM_MTP_MODEL"
            --spec-draft-device "${GLM_MTP_DEVICE:-CPU-NUMA0}"
            --spec-draft-ngl "${GLM_MTP_GPU_LAYERS:-all}"
          )
        fi
        ;;
      ngram-mod)
        speculative_args+=(
          --spec-ngram-mod-n-max "${GLM_SPEC_NGRAM_MOD_N_MAX:-64}"
          --spec-ngram-mod-n-min "${GLM_SPEC_NGRAM_MOD_N_MIN:-2}"
          --spec-ngram-mod-n-match "${GLM_SPEC_NGRAM_MOD_N_MATCH:-8}"
        )
        ;;
      none)
        ;;
      *)
        echo "Unsupported GLM_SPEC_TYPE component: $spec_part" >&2
        exit 1
        ;;
    esac
  done
fi
if [[ -n "${GLM_SPEC_ALIAS_P_MIN:-}" ]]; then
  speculative_args+=(--spec-alias-p-min "$GLM_SPEC_ALIAS_P_MIN")
fi

chat_template_args=()
if [[ -n "${GLM_CHAT_TEMPLATE_FILE:-}" ]]; then
  if [[ ! -r "$GLM_CHAT_TEMPLATE_FILE" ]]; then
    echo "GLM chat template is not readable: $GLM_CHAT_TEMPLATE_FILE" >&2
    exit 1
  fi
  chat_template_args=(--chat-template-file "$GLM_CHAT_TEMPLATE_FILE")
fi

reasoning_args=(--reasoning-format "${GLM_REASONING_FORMAT:-deepseek}")
case "${GLM_REASONING_PRESERVE:-1}" in
  1|true|yes|on)
    reasoning_args+=(--reasoning-preserve)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "GLM_REASONING_PRESERVE must be a boolean: ${GLM_REASONING_PRESERVE}" >&2
    exit 1
    ;;
esac

backend_args=()
numa_prefix=()
case "${GLM_BACKEND_MODE:-numa-tensor}" in
  numa-tensor)
    backend_args=(
      --gpu-layers 999
      --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3
      --split-mode tensor
      --tensor-split 1,1,1,1
    )
    ;;
  cpu-repack)
    backend_args=(
      --gpu-layers 0
    )
    case "${GLM_CPU_REPACK_NUMA_POLICY:-interleave}" in
      interleave)
        if ! command -v numactl >/dev/null 2>&1; then
          echo "GLM_CPU_REPACK_NUMA_POLICY=interleave requires numactl" >&2
          exit 1
        fi
        numa_prefix=(numactl --interleave=all)
        ;;
      distribute)
        backend_args+=(--numa distribute)
        ;;
      none)
        ;;
      *)
        echo "Unsupported GLM_CPU_REPACK_NUMA_POLICY: ${GLM_CPU_REPACK_NUMA_POLICY}" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported GLM_BACKEND_MODE: ${GLM_BACKEND_MODE}" >&2
    exit 1
    ;;
esac

exec "${numa_prefix[@]}" "$GLM_BINARY" \
  --host "$GLM_HOST" \
  --port "$port" \
  --model "$GLM_MODEL" \
  --alias "$GLM_ALIASES" \
  --load-mode mmap \
  --fit off \
  --ctx-size "$GLM_CONTEXT_SIZE" \
  --cache-type-k "$GLM_CACHE_TYPE_K" \
  --cache-type-v "$GLM_CACHE_TYPE_V" \
  --flash-attn on \
  --batch-size "$GLM_BATCH_SIZE" \
  --ubatch-size "$GLM_UBATCH_SIZE" \
  --parallel "${GLM_PARALLEL:-1}" \
  --cache-prompt \
  --cache-reuse 256 \
  --jinja \
  "${chat_template_args[@]}" \
  "${reasoning_args[@]}" \
  --predict 131072 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --timeout 3600 \
  --sse-ping-interval 15 \
  --threads-http 4 \
  --cors-origins localhost \
  --no-webui \
  --metrics \
  --log-timestamps \
  --verbosity "${GLM_LOG_VERBOSITY:-3}" \
  --threads "$GLM_THREADS" \
  --threads-batch "$GLM_THREADS" \
  "${backend_args[@]}" \
  "${speculative_args[@]}"
