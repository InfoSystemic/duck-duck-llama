#!/usr/bin/env bash
set -euo pipefail

# Start a verified non-Flash GLM-5.3 Q2/Q3 model from RAM without modifying or
# deleting the currently installed GLM-5.2 model.

first_shard="${1:-}"
base_config="${GLM53_BASE_CONFIG:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.env}"
stage_config="${GLM53_STAGE_CONFIG:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.glm53-staging.env}"
template_file="${GLM53_CHAT_TEMPLATE:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/chat-template-glm-5.3-llamacpp.jinja}"
inspector="/home/user/InfoSystemic/AI-Server/serving/glm-sr950/inspect-glm-gguf.py"
service_name="${GLM53_SERVICE:-glm53-sr950.service}"
port="${GLM53_PORT:-18091}"
health_timeout="${GLM53_HEALTH_TIMEOUT:-2400}"
smoke_timeout="${GLM53_SMOKE_TIMEOUT:-600}"
min_model_bytes="${GLM53_MIN_TOTAL_BYTES:-200000000000}"
min_available_bytes="${GLM53_MIN_AVAILABLE_MEMORY:-103079215104}"
state_dir="${GLM53_STATE_DIR:-/home/user/.local/state/glm-sr950}"
state_file="$state_dir/glm53-runtime.json"

usage() {
  printf 'Usage: %s FIRST_GGUF_SHARD\n' "${0##*/}" >&2
  exit 2
}

[[ -n "$first_shard" ]] || usage
[[ -r "$base_config" ]] || {
  printf 'Base GLM config is not readable: %s\n' "$base_config" >&2
  exit 2
}
[[ -r "$template_file" ]] || {
  printf 'GLM-5.3 compatibility template is not readable: %s\n' "$template_file" >&2
  exit 2
}
[[ -x "$inspector" ]] || {
  printf 'GGUF inspector is unavailable: %s\n' "$inspector" >&2
  exit 2
}
[[ "$port" =~ ^[0-9]+$ ]] && (( port > 0 && port < 65536 )) || {
  printf 'Invalid GLM53_PORT: %s\n' "$port" >&2
  exit 2
}

first_shard="$(realpath -e -- "$first_shard")"
case "${first_shard,,}" in
  *flash*)
    printf 'Refusing a Flash model: %s\n' "$first_shard" >&2
    exit 2
    ;;
esac

model_dir="$(dirname -- "$first_shard")"
model_name="$(basename -- "$first_shard")"
if [[ "$model_name" =~ ^GLM-5[.]3-(UD-Q[23]_K_XL)-00001-of-([0-9]{5})[.]gguf$ ]]; then
  quant="${BASH_REMATCH[1]}"
  shard_count=$((10#${BASH_REMATCH[2]}))
else
  printf 'Refusing unexpected model filename: %s\n' "$model_name" >&2
  exit 2
fi

shards=()
for ((index = 1; index <= shard_count; index++)); do
  shard="$(printf '%s/GLM-5.3-%s-%05d-of-%05d.gguf' "$model_dir" "$quant" "$index" "$shard_count")"
  [[ -r "$shard" ]] || {
    printf 'Missing or unreadable GGUF shard: %s\n' "$shard" >&2
    exit 2
  }
  shards+=("$shard")
done

total_bytes="$(stat -Lc '%s' -- "${shards[@]}" | awk '{ total += $1 } END { printf "%.0f", total }')"
if (( total_bytes < min_model_bytes )); then
  printf 'Refusing incomplete model: %s bytes is below %s\n' "$total_bytes" "$min_model_bytes" >&2
  exit 2
fi

mkdir -p -- "$state_dir"
work_dir="$(mktemp -d "$state_dir/.glm53-stage.XXXXXX")"
preflight_file="$work_dir/preflight.json"
models_file="$work_dir/models.json"
smoke_file="$work_dir/smoke.json"
tool_smoke_file="$work_dir/tool-smoke.json"
stage_ok=false

write_state() {
  local status="$1"
  local detail="$2"
  local temp_state
  temp_state="$(mktemp "$state_dir/.glm53-runtime-state.XXXXXX")"
  jq -n \
    --arg checked_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "$status" \
    --arg detail "$detail" \
    --arg model "$first_shard" \
    --arg service "$service_name" \
    --arg endpoint "http://127.0.0.1:$port" \
    --arg config "$stage_config" \
    --argjson total_bytes "$total_bytes" \
    '{checkedAt:$checked_at,status:$status,detail:$detail,model:$model,service:$service,endpoint:$endpoint,config:$config,totalBytes:$total_bytes}' \
    >"$temp_state"
  mv -f -- "$temp_state" "$state_file"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  rm -rf -- "$work_dir"
  if (( status != 0 )) && [[ "$stage_ok" != true ]]; then
    write_state failed "GLM-5.3 staged start or API validation failed"
    systemctl --user stop "$service_name" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_state preflight "Inspecting the verified GLM-5.3 shard set"
"$inspector" --json "$first_shard" >"$preflight_file"
jq -e '
  .metadata["general.architecture"] == "glm-dsa"
  and (.metadata["general.name"] | ascii_downcase | contains("5.3"))
  and .shard_count >= 2
  and .totals.elements > 700000000000
  and .totals.bytes > 200000000000
' "$preflight_file" >/dev/null || {
  printf 'GGUF preflight does not identify the expected full GLM-5.3 architecture\n' >&2
  exit 1
}

available_bytes="$(awk '/^MemAvailable:/ { printf "%.0f", $2 * 1024 }' /proc/meminfo)"
if (( available_bytes < min_available_bytes )); then
  printf 'Insufficient available RAM: need at least %s bytes, have %s\n' "$min_available_bytes" "$available_bytes" >&2
  exit 1
fi

temp_config="$(mktemp "${stage_config}.XXXXXX")"
awk \
  -v model="$first_shard" \
  -v port="$port" \
  -v template="$template_file" '
  BEGIN {
    saw_template = 0
    saw_reasoning = 0
  }
  /^GLM_MODEL=/                 { print "GLM_MODEL=" model; next }
  /^GLM_PORT=/                  { print "GLM_PORT=" port; next }
  /^GLM_CONTEXT_SIZE=/          { print "GLM_CONTEXT_SIZE=32768"; next }
  /^GLM_BATCH_SIZE=/            { print "GLM_BATCH_SIZE=256"; next }
  /^GLM_UBATCH_SIZE=/           { print "GLM_UBATCH_SIZE=128"; next }
  /^GLM_THREADS=/               { print "GLM_THREADS=64"; next }
  /^GLM_ALIASES=/               { print "GLM_ALIASES=glm-sr950,glm-sr950-agentic,glm-5.3-sr950,glm-5.3,glm53,GLM-5.3"; next }
  /^GLM_BACKEND_MODE=/          { print "GLM_BACKEND_MODE=cpu-repack"; next }
  /^GLM_CPU_REPACK_NUMA_POLICY=/{ print "GLM_CPU_REPACK_NUMA_POLICY=interleave"; saw_policy = 1; next }
  /^GLM_LOG_VERBOSITY=/         { print "GLM_LOG_VERBOSITY=2"; next }
  /^GLM_SPEC_TYPE=/             { print "GLM_SPEC_TYPE=none"; next }
  /^GLM_SPEC_ALIAS_P_MIN=/      { print "GLM_SPEC_ALIAS_P_MIN="; next }
  /^GGML_CPU_NUMA_REPACK=/      { print "GGML_CPU_NUMA_REPACK=0"; next }
  /^GGML_CPU_NUMA_DEVICES=/     { print "GGML_CPU_NUMA_DEVICES=0"; next }
  /^GGML_CPU_NUMA_DIRECT_ALLREDUCE=/ { print "GGML_CPU_NUMA_DIRECT_ALLREDUCE=0"; next }
  /^GGML_GLM_ATTN_TP=/          { print "GGML_GLM_ATTN_TP=0"; next }
  /^GGML_CPU_IQ2_XS_REPACK=/    { print "GGML_CPU_IQ2_XS_REPACK=0"; next }
  /^GGML_CPU_IQ3_XXS_REPACK=/   { print "GGML_CPU_IQ3_XXS_REPACK=0"; next }
  /^GGML_CPU_Q5_K_REPACK=/      { print "GGML_CPU_Q5_K_REPACK=0"; next }
  /^GLM_CHAT_TEMPLATE_FILE=/    { print "GLM_CHAT_TEMPLATE_FILE=" template; saw_template = 1; next }
  /^GLM_REASONING_PRESERVE=/    { print "GLM_REASONING_PRESERVE=1"; saw_reasoning = 1; next }
  { print }
  END {
    if (!saw_template)  print "GLM_CHAT_TEMPLATE_FILE=" template
    if (!saw_reasoning) print "GLM_REASONING_PRESERVE=1"
    if (!saw_policy)    print "GLM_CPU_REPACK_NUMA_POLICY=interleave"
  }
' "$base_config" >"$temp_config"
chmod 0644 -- "$temp_config"
mv -f -- "$temp_config" "$stage_config"

write_state starting "Starting GLM-5.3 at 32K context without speculative decoding or weight repacking"
systemctl --user daemon-reload
systemctl --user stop glm-sr950.service
systemctl --user restart "$service_name"

deadline=$((SECONDS + health_timeout))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    break
  fi
  if ! systemctl --user is-active --quiet "$service_name"; then
    printf 'GLM-5.3 service exited while loading\n' >&2
    journalctl --user -u "$service_name" -n 120 --no-pager >&2 || true
    exit 1
  fi
  sleep 5
done
curl -fsS --max-time 10 "http://127.0.0.1:$port/health" >/dev/null || {
  printf 'GLM-5.3 did not become healthy within %s seconds\n' "$health_timeout" >&2
  exit 1
}

curl -fsS --max-time 30 "http://127.0.0.1:$port/v1/models" >"$models_file"
jq -e '.data | any(.id == "glm-sr950" or ((.aliases // []) | index("glm-sr950") != null))' "$models_file" >/dev/null || {
  printf 'Stable glm-sr950 alias is absent from /v1/models\n' >&2
  exit 1
}

smoke_request='{"model":"glm-sr950","messages":[{"role":"user","content":"Reply with READY."}],"reasoning_effort":"low","max_tokens":1,"temperature":1.0,"top_p":0.95,"stream":false}'
curl -fsS --max-time "$smoke_timeout" \
  -H 'Content-Type: application/json' \
  -d "$smoke_request" \
  "http://127.0.0.1:$port/v1/chat/completions" >"$smoke_file"
jq -e '.choices | length > 0' "$smoke_file" >/dev/null || {
  printf 'GLM-5.3 basic completion smoke test failed\n' >&2
  exit 1
}

tool_smoke_request='{"model":"glm-sr950-agentic","messages":[{"role":"user","content":"Look up Denver weather."},{"role":"assistant","content":"","tool_calls":[{"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":{"location":"Denver"}}}]},{"role":"tool","tool_call_id":"call_weather","content":"Sunny"},{"role":"user","content":"Acknowledge the result."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"reasoning_effort":"low","max_tokens":1,"temperature":1.0,"top_p":0.95,"stream":false}'
curl -fsS --max-time "$smoke_timeout" \
  -H 'Content-Type: application/json' \
  -d "$tool_smoke_request" \
  "http://127.0.0.1:$port/v1/chat/completions" >"$tool_smoke_file"
jq -e '.choices | length > 0' "$tool_smoke_file" >/dev/null || {
  printf 'GLM-5.3 tool-history/API smoke test failed\n' >&2
  exit 1
}

stage_ok=true
write_state running "Health, model alias, completion, and tool-history API checks passed"
trap - EXIT INT TERM
rm -rf -- "$work_dir"
printf 'GLM-5.3 is running at http://127.0.0.1:%s as glm-sr950 and glm-5.3\n' "$port"
