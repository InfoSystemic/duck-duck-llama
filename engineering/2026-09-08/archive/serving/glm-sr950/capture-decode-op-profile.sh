#!/usr/bin/env bash
set -euo pipefail

url="${GLM_PROFILE_URL:-http://127.0.0.1:18091/v1/chat/completions}"
model="${GLM_PROFILE_MODEL:-glm-sr950}"
arm_file="${GLM_PROFILE_ARM_FILE:-/tmp/glm-sr950-op-profile.arm}"
max_tokens="${GLM_PROFILE_MAX_TOKENS:-128}"
spec_n_max="${GLM_PROFILE_SPEC_N_MAX:-0}"
spec_p_min="${GLM_PROFILE_SPEC_P_MIN:-}"

if [[ ! "$spec_n_max" =~ ^[0-9]+$ ]]; then
    printf 'GLM_PROFILE_SPEC_N_MAX must be a non-negative integer\n' >&2
    exit 2
fi
if [[ -n "$spec_p_min" ]] && ! jq -en --arg value "$spec_p_min" \
    '$value | tonumber | . >= 0 and . <= 1' >/dev/null 2>&1; then
    printf 'GLM_PROFILE_SPEC_P_MIN must be a number between 0 and 1\n' >&2
    exit 2
fi

cleanup() {
    rm -f -- "$arm_file"
}
trap cleanup EXIT
cleanup

payload="$(jq -nc \
    --arg model "$model" \
    --argjson max_tokens "$max_tokens" \
    --argjson spec_n_max "$spec_n_max" \
    --argjson spec_p_min "${spec_p_min:-null}" \
    '{
        model: $model,
        messages: [{
            role: "user",
            content: "Write a compact technical explanation of why NUMA-local memory placement matters for CPU language-model decoding. Use several paragraphs."
        }],
        max_tokens: $max_tokens,
        temperature: 0,
        seed: 260826,
        reasoning_effort: "low",
        cache_prompt: false,
        stream: true,
        "speculative.n_max": $spec_n_max
    }
    | if $spec_p_min == null then . else .["speculative.p_min"] = $spec_p_min end')"

armed=0
while IFS= read -r line; do
    if [[ "$armed" -eq 0 && "$line" == data:* && "$line" != "data: [DONE]" ]]; then
        touch -- "$arm_file"
        armed=1
        printf 'decode profiler armed after first SSE event\n' >&2
    fi
    printf '%s\n' "$line"
done < <(
    curl -fsSN --max-time 900 \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        "$url"
)

if [[ "$armed" -ne 1 ]]; then
    printf 'request completed without an SSE event; profiler was not armed\n' >&2
    exit 1
fi
