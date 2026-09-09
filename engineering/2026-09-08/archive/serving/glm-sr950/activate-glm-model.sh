#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: activate-glm-model.sh FIRST_GGUF_SHARD VERSION

Example:
  activate-glm-model.sh /models/GLM-5.3-UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf 5.3

The script verifies the complete shard set, switches model.env atomically,
restarts glm-sr950.service, runs health and API smoke tests, and rolls back on
any failure. It never deletes the previous model.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage
requested_model="$1"
version="$2"
config_file="${GLM_SR950_CONFIG:-/home/kwebb/InfoSystemic/AI-Server/serving/glm-sr950/model.env}"
service_name="${GLM_SR950_SERVICE:-glm-sr950.service}"
health_timeout="${GLM_HEALTH_TIMEOUT:-1200}"
smoke_timeout="${GLM_SMOKE_TIMEOUT:-300}"
min_model_bytes="${GLM_MIN_MODEL_BYTES:-100000000000}"

if [[ ! "$version" =~ ^[0-9]+([.][0-9]+)*$ ]]; then
  echo "Invalid GLM version: $version" >&2
  exit 2
fi
if [[ ! -r "$requested_model" ]]; then
  echo "First GGUF shard is not readable: $requested_model" >&2
  exit 2
fi

new_model="$(realpath -e -- "$requested_model")"
if [[ ! "$new_model" =~ ^[-A-Za-z0-9_./:+,@=]+$ ]]; then
  echo "Model path contains characters unsupported by model.env: $new_model" >&2
  exit 2
fi
if [[ ! -r "$config_file" ]]; then
  echo "GLM configuration is not readable: $config_file" >&2
  exit 2
fi

model_dir="$(dirname -- "$new_model")"
model_name="$(basename -- "$new_model")"
shards=()
if [[ "$model_name" =~ ^(.+)-00001-of-([0-9]{5})[.]gguf$ ]]; then
  shard_prefix="${BASH_REMATCH[1]}"
  shard_count=$((10#${BASH_REMATCH[2]}))
  for ((i = 1; i <= shard_count; i++)); do
    shard="$(printf '%s/%s-%05d-of-%05d.gguf' "$model_dir" "$shard_prefix" "$i" "$shard_count")"
    if [[ ! -r "$shard" ]]; then
      echo "Missing or unreadable GGUF shard: $shard" >&2
      exit 2
    fi
    shards+=("$shard")
  done
else
  shards+=("$new_model")
fi

total_bytes="$(stat -Lc '%s' -- "${shards[@]}" | awk '{ total += $1 } END { printf "%.0f", total }')"
if (( total_bytes < min_model_bytes )); then
  echo "Shard set is only $total_bytes bytes; refusing a likely incomplete GLM model" >&2
  exit 2
fi

aliases="glm-sr950,glm-sr950-agentic,glm-$version-sr950,glm-$version,glm${version//./},GLM-$version"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_file="$config_file.before-glm-$version-$timestamp"
temp_file="$(mktemp "${config_file}.XXXXXX")"
lock_file="${config_file}.lock"
changed=false
committed=false

wait_for_health() {
  local timeout="$1"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 http://127.0.0.1:18091/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

rollback_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  rm -f -- "$temp_file"
  if [[ "$changed" == true && "$committed" != true ]]; then
    echo "Activation failed; restoring $backup_file" >&2
    cp -a -- "$backup_file" "$config_file"
    systemctl --user restart "$service_name" || true
    wait_for_health "$health_timeout" || true
  fi
  exit "$status"
}
trap rollback_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$lock_file"
flock -n 9 || {
  echo "Another GLM activation is already running" >&2
  exit 1
}

cp -a -- "$config_file" "$backup_file"
awk -v model="$new_model" -v aliases="$aliases" '
  BEGIN { saw_model = 0; saw_aliases = 0 }
  /^GLM_MODEL=/   { print "GLM_MODEL=" model; saw_model = 1; next }
  /^GLM_ALIASES=/ { print "GLM_ALIASES=" aliases; saw_aliases = 1; next }
  { print }
  END {
    if (!saw_model)   print "GLM_MODEL=" model
    if (!saw_aliases) print "GLM_ALIASES=" aliases
  }
' "$config_file" >"$temp_file"
chmod --reference="$config_file" "$temp_file"
mv -f -- "$temp_file" "$config_file"
changed=true

systemctl --user restart "$service_name"
if ! wait_for_health "$health_timeout"; then
  echo "New GLM service did not become healthy within ${health_timeout}s" >&2
  exit 1
fi

if ! curl -fsS --max-time 30 http://127.0.0.1:18091/v1/models \
    | jq -e --arg model glm-sr950 \
        '.data | any(.id == $model or ((.aliases // []) | index($model) != null))' >/dev/null; then
  echo "Stable glm-sr950 alias is absent from /v1/models" >&2
  exit 1
fi
if ! curl -fsS --max-time 30 http://127.0.0.1:18091/v1/models \
    | jq -e --arg model glm-sr950-agentic \
        '.data | any(.id == $model or ((.aliases // []) | index($model) != null))' >/dev/null; then
  echo "Stable glm-sr950-agentic alias is absent from /v1/models" >&2
  exit 1
fi

smoke_request='{"model":"glm-sr950","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":1,"temperature":0}'
if ! curl -fsS --max-time "$smoke_timeout" \
    -H 'Content-Type: application/json' \
    -d "$smoke_request" \
    http://127.0.0.1:18091/v1/chat/completions \
    | jq -e '.choices | length > 0' >/dev/null; then
  echo "New GLM service failed the completion smoke test" >&2
  exit 1
fi

agentic_smoke_request='{"model":"glm-sr950-agentic","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":2,"temperature":0,"verbose":true}'
if ! curl -fsS --max-time "$smoke_timeout" \
    -H 'Content-Type: application/json' \
    -d "$agentic_smoke_request" \
    http://127.0.0.1:18091/v1/chat/completions \
    | jq -e '.__verbose.generation_settings["speculative.p_min"] > 0.89' >/dev/null; then
  echo "New GLM service failed the OpenAI agentic-profile smoke test" >&2
  exit 1
fi

if ! curl -fsS --max-time "$smoke_timeout" \
    -H 'Content-Type: application/json' \
    -d "$agentic_smoke_request" \
    http://127.0.0.1:18091/v1/messages \
    | jq -e '.__verbose.generation_settings["speculative.p_min"] > 0.89' >/dev/null; then
  echo "New GLM service failed the Anthropic agentic-profile smoke test" >&2
  exit 1
fi

committed=true
trap - EXIT INT TERM
echo "Activated GLM-$version from $new_model"
echo "Rollback configuration retained at $backup_file"
