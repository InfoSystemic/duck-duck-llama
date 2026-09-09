#!/usr/bin/env bash
set -euo pipefail

# Wait for a complete reviewed non-Flash GLM-5.3 CPU GGUF set, pin the exact
# publisher revision, download it to RAM without touching existing models, and
# verify every file against its Hugging Face LFS SHA-256.

repo="${GLM53_REPO:-unsloth/GLM-5.3-GGUF}"
quant="${GLM53_QUANT:-UD-Q3_K_XL}"
destination="${GLM53_DEST:-/dev/shm/ai-models/GLM-5.3-GGUF}"
poll_seconds="${GLM53_POLL_SECONDS:-120}"
min_total_bytes="${GLM53_MIN_TOTAL_BYTES:-200000000000}"
reserve_bytes="${GLM53_RESERVE_BYTES:-10737418240}"
state_dir="${GLM53_STATE_DIR:-/home/user/.local/state/glm-sr950}"
state_file="$state_dir/glm53-download.json"
inspector="/home/user/InfoSystemic/AI-Server/serving/glm-sr950/inspect-glm-gguf.py"
starter="${GLM53_STARTER:-/home/user/InfoSystemic/AI-Server/serving/glm-sr950/start-glm53-staged.sh}"
auto_start="${GLM53_AUTO_START:-1}"

case "$repo" in
  */GLM-5.3-GGUF) ;;
  *)
    printf 'Refusing unexpected repository (must be non-Flash GLM-5.3-GGUF): %s\n' "$repo" >&2
    exit 2
    ;;
esac
if [[ "$repo" == *Flash* || "$repo" == *flash* ]]; then
  printf 'Refusing Flash repository: %s\n' "$repo" >&2
  exit 2
fi
case "$quant" in
  UD-Q2_K_XL|UD-Q3_K_XL) ;;
  *)
    printf 'Refusing unreviewed quantization: %s\n' "$quant" >&2
    exit 2
    ;;
esac
if ! [[ "$poll_seconds" =~ ^[0-9]+$ ]] || (( poll_seconds < 30 )); then
  printf 'GLM53_POLL_SECONDS must be an integer of at least 30\n' >&2
  exit 2
fi
if [[ "$auto_start" != "0" && "$auto_start" != "1" ]]; then
  printf 'GLM53_AUTO_START must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$auto_start" == "1" && ! -x "$starter" ]]; then
  printf 'GLM-5.3 staged starter is unavailable: %s\n' "$starter" >&2
  exit 2
fi

command -v curl >/dev/null
command -v hf >/dev/null
command -v jq >/dev/null
command -v sha256sum >/dev/null
[[ -x "$inspector" ]] || {
  printf 'Missing GGUF inspector: %s\n' "$inspector" >&2
  exit 2
}

mkdir -p -- "$state_dir" "$destination"
work_dir="$(mktemp -d "$state_dir/.glm53-download.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT INT TERM
model_json="$work_dir/model.json"
tree_json="$work_dir/tree.json"
files_json="$work_dir/files.json"

write_state() {
  local status="$1"
  local detail="$2"
  local revision="${3:-}"
  local total_bytes="${4:-0}"
  local temp_state
  temp_state="$(mktemp "$state_dir/.glm53-state.XXXXXX")"
  jq -n \
    --arg checked_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "$status" \
    --arg detail "$detail" \
    --arg repo "$repo" \
    --arg quant "$quant" \
    --arg destination "$destination" \
    --arg revision "$revision" \
    --argjson total_bytes "$total_bytes" \
    '{checkedAt:$checked_at,status:$status,detail:$detail,repo:$repo,quant:$quant,destination:$destination,revision:$revision,totalBytes:$total_bytes}' \
    >"$temp_state"
  mv -f -- "$temp_state" "$state_file"
}

discover_complete_set() {
  curl -fsS --retry 3 --retry-delay 5 \
    "https://huggingface.co/api/models/$repo" >"$model_json"
  revision="$(jq -er '.sha' "$model_json")"
  curl -fsS --retry 3 --retry-delay 5 \
    "https://huggingface.co/api/models/$repo/tree/$revision?recursive=true&expand=false&limit=1000" \
    >"$tree_json"

  jq -c --arg prefix "$quant/" \
    '[.[] | select(.type == "file" and (.path | startswith($prefix)))]' \
    "$tree_json" >"$files_json"

  shard_count="$(
    jq -r --arg pattern "^$quant/GLM-5[.]3-$quant-00001-of-(?<count>[0-9]{5})[.]gguf$" '
      [.[].path | capture($pattern).count?] | first // empty
    ' "$files_json"
  )"
  [[ -n "$shard_count" ]] || return 1
  shard_count=$((10#$shard_count))

  local index expected present
  for ((index = 1; index <= shard_count; index++)); do
    expected="$(printf '%s/GLM-5.3-%s-%05d-of-%05d.gguf' "$quant" "$quant" "$index" "$shard_count")"
    present="$(jq -r --arg path "$expected" '[.[] | select(.path == $path and (.size // 0) > 0 and (.lfs.oid // "") != "")] | length' "$files_json")"
    (( present == 1 )) || return 1
  done

  total_bytes="$(jq '[.[].size // 0] | add // 0' "$files_json")"
  (( total_bytes >= min_total_bytes )) || return 1
  return 0
}

while ! discover_complete_set; do
  write_state waiting "Publisher has not exposed a complete $quant shard set yet" "${revision:-}"
  printf '%s waiting for complete %s/%s at %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo" "$quant" "$revision"
  sleep "$poll_seconds"
done

available_bytes="$(df -B1 --output=avail "$destination" | awk 'NR == 2 { print $1 }')"
required_bytes=$((total_bytes + reserve_bytes))
if (( available_bytes < required_bytes )); then
  write_state blocked "Insufficient destination space: need $required_bytes bytes, have $available_bytes" "$revision" "$total_bytes"
  printf 'Insufficient free space at %s: need %s bytes including reserve, have %s\n' \
    "$destination" "$required_bytes" "$available_bytes" >&2
  exit 1
fi

write_state downloading "Complete shard set discovered; download started" "$revision" "$total_bytes"
printf 'Downloading pinned %s revision %s (%s bytes) to %s\n' \
  "$repo" "$revision" "$total_bytes" "$destination"

# Keep all transient Hub/Xet data on the RAM filesystem. Disabling the Xet
# chunk cache avoids a second large copy while local-dir downloads are active.
HF_HOME="$destination/.hf-home" \
HF_XET_CACHE="$destination/.xet-cache" \
HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
hf download "$repo" \
  --revision "$revision" \
  --include "$quant/*" \
  --local-dir "$destination" \
  --max-workers "${GLM53_DOWNLOAD_WORKERS:-8}"

manifest="$destination/MANIFEST.$revision.tsv"
: >"$manifest"
while IFS=$'\t' read -r path size oid; do
  file="$destination/$path"
  [[ -f "$file" ]] || {
    write_state failed "Downloaded set is missing $path" "$revision" "$total_bytes"
    printf 'Missing downloaded file: %s\n' "$file" >&2
    exit 1
  }
  actual_size="$(stat -Lc '%s' -- "$file")"
  [[ "$actual_size" == "$size" ]] || {
    write_state failed "Size mismatch for $path" "$revision" "$total_bytes"
    printf 'Size mismatch for %s: expected %s, got %s\n' "$file" "$size" "$actual_size" >&2
    exit 1
  }
  actual_oid="$(sha256sum -- "$file" | awk '{ print $1 }')"
  [[ "$actual_oid" == "$oid" ]] || {
    write_state failed "SHA-256 mismatch for $path" "$revision" "$total_bytes"
    printf 'SHA-256 mismatch for %s\n' "$file" >&2
    exit 1
  }
  printf '%s\t%s\t%s\n' "$oid" "$size" "$path" >>"$manifest"
done < <(jq -r '.[] | [.path, (.size | tostring), .lfs.oid] | @tsv' "$files_json")

first_shard="$(printf '%s/%s/GLM-5.3-%s-00001-of-%05d.gguf' "$destination" "$quant" "$quant" "$shard_count")"
"$inspector" --json "$first_shard" >"$destination/sr950-preflight.json"
chmod 0444 -- "$destination/$quant"/*.gguf "$manifest" "$destination/sr950-preflight.json"
write_state verified "Download, size, SHA-256, and GGUF preflight checks passed" "$revision" "$total_bytes"
printf 'Verified non-Flash GLM-5.3 at %s (revision %s)\n' "$destination" "$revision"

if [[ "$auto_start" == "1" ]]; then
  write_state starting "Verified model is being loaded and API-tested" "$revision" "$total_bytes"
  if ! "$starter" "$first_shard"; then
    write_state failed "Verified model failed staged start or API validation" "$revision" "$total_bytes"
    exit 1
  fi
  write_state running "Verified model is healthy and passed API smoke tests" "$revision" "$total_bytes"
fi
