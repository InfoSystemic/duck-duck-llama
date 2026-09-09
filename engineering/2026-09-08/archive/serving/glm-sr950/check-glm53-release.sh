#!/usr/bin/env bash
set -euo pipefail

state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/glm-sr950"
state_file="$state_dir/glm53-release.json"
mkdir -p "$state_dir"

repos=(
  "zai-org/GLM-5.3"
  "zai-org/GLM-5.3-FP8"
  "zai-org/GLM-5.3-GGUF"
  "unsloth/GLM-5.3-GGUF"
)
search_authors=(
  "zai-org"
  "unsloth"
)
watched_repos=(
  "zai-org/GLM-5"
  "unsloth/GLM-5-GGUF"
)

checked_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
previous_state='{}'
if [[ -r "$state_file" ]] && jq -e 'type == "object"' "$state_file" >/dev/null 2>&1; then
  previous_state="$(<"$state_file")"
fi
was_available=false
if jq -e '.available == true' <<<"$previous_state" >/dev/null 2>&1; then
  was_available=true
fi

available=false
had_network_error=false
repositories='[]'
for repo in "${repos[@]}"; do
  api_url="https://huggingface.co/api/models/$repo"
  if ! http_code="$(curl -sS -L -o /dev/null -w '%{http_code}' \
      --connect-timeout 10 --max-time 30 "$api_url")"; then
    http_code=000
    had_network_error=true
  fi

  case "$http_code" in
    200)
      status=available
      available=true
      ;;
    401)
      # Hugging Face returns 401 for both private and nonexistent model IDs.
      status=not_public
      ;;
    403)
      status=restricted
      ;;
    404)
      status=not_found
      ;;
    000)
      status=network_error
      ;;
    *)
      status="http_$http_code"
      ;;
  esac

  repositories="$(jq -cn \
    --argjson existing "$repositories" \
    --arg repo "$repo" \
    --arg url "https://huggingface.co/$repo" \
    --arg status "$status" \
    --arg httpCode "$http_code" \
    '$existing + [{repo: $repo, url: $url, status: $status, httpCode: $httpCode}]')"
done

discovered_repositories='[]'
searches='[]'
for author in "${search_authors[@]}"; do
  api_url="https://huggingface.co/api/models?author=$author&search=GLM-5.3&limit=100"
  search_status=ok
  search_results='[]'
  if ! response="$(curl -fsS -L --connect-timeout 10 --max-time 30 "$api_url")"; then
    search_status=network_error
    had_network_error=true
  elif ! jq -e 'type == "array"' <<<"$response" >/dev/null; then
    search_status=invalid_response
  else
    search_results="$(jq -c '[.[]
      | select(.id | test("GLM[-_.]?5[.]3"; "i"))
      | {repo: .id, url: ("https://huggingface.co/" + .id), sha, lastModified}]' <<<"$response")"
    discovered_repositories="$(jq -cn \
      --argjson existing "$discovered_repositories" \
      --argjson found "$search_results" \
      '$existing + $found | unique_by(.repo)')"
  fi
  searches="$(jq -cn \
    --argjson existing "$searches" \
    --arg author "$author" \
    --arg status "$search_status" \
    --argjson results "$search_results" \
    '$existing + [{author: $author, status: $status, results: $results}]')"
done
if [[ "$(jq 'length' <<<"$discovered_repositories")" -gt 0 ]]; then
  available=true
fi

watched_repositories='[]'
watched_repository_changed=false
for repo in "${watched_repos[@]}"; do
  api_url="https://huggingface.co/api/models/$repo"
  watch_status=ok
  sha=''
  last_modified=''
  changed_since_previous=false
  if ! response="$(curl -fsS -L --connect-timeout 10 --max-time 30 "$api_url")"; then
    watch_status=network_error
    had_network_error=true
  elif ! jq -e 'type == "object" and (.sha | type == "string")' <<<"$response" >/dev/null; then
    watch_status=invalid_response
  else
    sha="$(jq -r '.sha' <<<"$response")"
    last_modified="$(jq -r '.lastModified // ""' <<<"$response")"
    previous_sha="$(jq -r --arg repo "$repo" \
      '.watchedRepositories[]? | select(.repo == $repo) | .sha // empty' \
      <<<"$previous_state" | head -n 1)"
    if [[ -n "$previous_sha" && "$sha" != "$previous_sha" ]]; then
      changed_since_previous=true
      watched_repository_changed=true
    fi
  fi
  watched_repositories="$(jq -cn \
    --argjson existing "$watched_repositories" \
    --arg repo "$repo" \
    --arg url "https://huggingface.co/$repo" \
    --arg status "$watch_status" \
    --arg sha "$sha" \
    --arg lastModified "$last_modified" \
    --argjson changedSincePrevious "$changed_since_previous" \
    '$existing + [{repo: $repo, url: $url, status: $status, sha: $sha,
      lastModified: $lastModified, changedSincePrevious: $changedSincePrevious}]')"
done

tmp_file="$(mktemp "$state_dir/.glm53-release.XXXXXX")"
trap 'rm -f -- "$tmp_file"' EXIT
jq -n \
  --arg checkedAt "$checked_at" \
  --argjson available "$available" \
  --argjson repositories "$repositories" \
  --argjson discoveredRepositories "$discovered_repositories" \
  --argjson searches "$searches" \
  --argjson watchedRepositories "$watched_repositories" \
  --argjson watchedRepositoryChanged "$watched_repository_changed" \
  '{checkedAt: $checkedAt, available: $available,
    repositories: $repositories,
    discoveredRepositories: $discoveredRepositories,
    searches: $searches,
    watchedRepositories: $watchedRepositories,
    watchedRepositoryChanged: $watchedRepositoryChanged}' \
  >"$tmp_file"
mv -f -- "$tmp_file" "$state_file"
trap - EXIT

if [[ "$available" == true ]]; then
  if [[ "$was_available" != true ]]; then
    logger -t glm53-watch "GLM-5.3 is now available on Hugging Face; inspect $state_file"
  fi
  jq . "$state_file"
  exit 0
fi

if [[ "$watched_repository_changed" == true ]]; then
  logger -t glm53-watch "An existing official GLM-5 repository changed; inspect $state_file"
fi

jq . "$state_file"
if [[ "$had_network_error" == true ]]; then
  exit 1
fi
