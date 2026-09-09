#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
window_guard="${GLM_BENCH_WINDOW_GUARD:-$script_dir/assert-benchmark-window.sh}"
benchmark_lock="${GLM_BENCH_LOCK:-/tmp/glm-sr950-benchmark.lock}"
exec 9>"$benchmark_lock"
if ! flock -n 9; then
    echo "Another GLM benchmark holds $benchmark_lock" >&2
    exit 75
fi
"$window_guard"

label="${1:-service-default}"
url="${GLM_BENCH_URL:-http://127.0.0.1:18091/v1/chat/completions}"
model="${GLM_BENCH_MODEL:-glm-sr950}"
max_tokens="${GLM_BENCH_TOKENS:-96}"
temperature="${GLM_BENCH_TEMPERATURE:-0.7}"
spec_n_max="${GLM_BENCH_SPEC_N_MAX:-}"
spec_p_min="${GLM_BENCH_SPEC_P_MIN:-}"
output="${GLM_WORKLOAD_OUT:-$(mktemp /tmp/glm-workloads.XXXXXX)}"
cleanup_output=false
if [[ -z "${GLM_WORKLOAD_OUT:-}" ]]; then
    cleanup_output=true
fi
if [[ -n "$spec_p_min" ]] && ! jq -en --arg value "$spec_p_min" \
    '$value | tonumber | . >= 0 and . <= 1' >/dev/null 2>&1; then
    echo "GLM_BENCH_SPEC_P_MIN must be a number between 0 and 1" >&2
    exit 2
fi
trap 'if [[ "$cleanup_output" == true ]]; then rm -f -- "$output"; fi' EXIT

if [[ ! "$max_tokens" =~ ^([2-9]|[1-9][0-9]+)$ ]]; then
    echo "GLM_BENCH_TOKENS must be an integer of at least 2" >&2
    exit 2
fi
if ! jq -en --arg value "$temperature" '$value | tonumber | . >= 0' >/dev/null; then
    echo "GLM_BENCH_TEMPERATURE must be a non-negative number" >&2
    exit 2
fi
if [[ -n "$spec_n_max" && ! "$spec_n_max" =~ ^[0-9]+$ ]]; then
    echo "GLM_BENCH_SPEC_N_MAX must be a non-negative integer" >&2
    exit 2
fi

prompts=(
    "Explain how NUMA placement affects mixture-of-experts inference on a four-socket CPU server. Be technical and continue until the output limit."
    "Write a robust Python function that merges overlapping half-open integer intervals, including validation and a short complexity explanation. Continue until the output limit."
    "Review a C++ lock-free queue design for memory-ordering bugs and propose a safe correction. Continue until the output limit."
    "Design a rollback-safe PostgreSQL migration that replaces a populated text status column with an enum under live traffic. Continue until the output limit."
    "Diagnose a systemd service that is healthy locally but intermittently times out through a reverse proxy and encrypted tunnel. Continue until the output limit."
    "Threat-model a private AI gateway that exposes OpenAI and Anthropic-compatible endpoints to one remote collaborator. Continue until the output limit."
    "Derive the asymptotic time and space complexity of Dijkstra's algorithm with a binary heap, then discuss disconnected graphs. Continue until the output limit."
    "Draft a concise incident report for a production inference slowdown caused by an incorrect NUMA policy, including impact, cause, correction, and prevention. Continue until the output limit."
)

: >"$output"
for i in "${!prompts[@]}"; do
    "$window_guard"
    seed=$((4200 + i))
    payload="$(jq -cn \
        --arg model "$model" \
        --arg prompt "${prompts[$i]}" \
        --argjson max_tokens "$max_tokens" \
        --argjson temperature "$temperature" \
        --argjson spec_n_max "${spec_n_max:-null}" \
        --argjson spec_p_min "${spec_p_min:-null}" \
        --argjson seed "$seed" \
        '{
            model: $model,
            messages: [{role: "user", content: $prompt}],
            max_tokens: $max_tokens,
            temperature: $temperature,
            seed: $seed,
            cache_prompt: false,
            stream: false
        }
        | if $spec_n_max == null then . else .["speculative.n_max"] = $spec_n_max end
        | if $spec_p_min == null then . else .["speculative.p_min"] = $spec_p_min end')"

    response="$(curl -fsS --max-time 600 \
        -H 'Content-Type: application/json' \
        --data-binary "$payload" \
        "$url")"

    result="$(jq -ce \
        --arg label "$label" \
        --argjson workload "$((i + 1))" \
        '{
            label: $label,
            workload: $workload,
            prompt_tokens: .usage.prompt_tokens,
            output_tokens: .usage.completion_tokens,
            prompt_ms: .timings.prompt_ms,
            predicted_ms: .timings.predicted_ms,
            prompt_tok_s: .timings.prompt_per_second,
            decode_tok_s: .timings.predicted_per_second,
            draft_tokens: (.timings.draft_n // 0),
            accepted_tokens: (.timings.draft_n_accepted // 0)
        }
        | select(.output_tokens > 0 and .decode_tok_s > 0)' <<<"$response")"

    printf '%s\n' "$result" | tee -a "$output"
done

jq -s '
    {
        label: .[0].label,
        workloads: length,
        output_tokens: ([.[].output_tokens] | add),
        aggregate_decode_tok_s: (([.[].output_tokens] | add) * 1000 / ([.[].predicted_ms] | add)),
        mean_decode_tok_s: (([.[].decode_tok_s] | add) / length),
        min_decode_tok_s: ([.[].decode_tok_s] | min),
        max_decode_tok_s: ([.[].decode_tok_s] | max),
        aggregate_prompt_tok_s: (([.[].prompt_tokens] | add) * 1000 / ([.[].prompt_ms] | add)),
        draft_tokens: ([.[].draft_tokens] | add),
        accepted_tokens: ([.[].accepted_tokens] | add),
        draft_acceptance: (
            ([.[].draft_tokens] | add) as $drafts
            | if $drafts > 0 then ([.[].accepted_tokens] | add) / $drafts else 0 end
        )
    }
' "$output"

if [[ "$cleanup_output" != true ]]; then
    echo "Detailed results: $output" >&2
fi
