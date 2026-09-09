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

url="${GLM_BENCH_URL:-http://127.0.0.1:18091/v1/chat/completions}"
model="${GLM_BENCH_MODEL:-glm-sr950}"
samples="${GLM_BENCH_SAMPLES:-3}"
max_tokens="${GLM_BENCH_TOKENS:-128}"
spec_p_min="${GLM_BENCH_SPEC_P_MIN:-}"
temperature="${GLM_BENCH_TEMPERATURE:-0.7}"

if (( $# > 0 )); then
    draft_lengths=("$@")
else
    draft_lengths=(0 1 2)
fi

if [[ ! "$samples" =~ ^[1-9][0-9]*$ || ! "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    echo "GLM_BENCH_SAMPLES and GLM_BENCH_TOKENS must be positive integers" >&2
    exit 2
fi
if [[ -n "$spec_p_min" ]] && ! jq -en --arg value "$spec_p_min" \
    '$value | tonumber | . >= 0 and . <= 1' >/dev/null 2>&1; then
    echo "GLM_BENCH_SPEC_P_MIN must be a number between 0 and 1" >&2
    exit 2
fi
if ! jq -en --arg value "$temperature" \
    '$value | tonumber | . >= 0 and . <= 2' >/dev/null 2>&1; then
    echo "GLM_BENCH_TEMPERATURE must be a number between 0 and 2" >&2
    exit 2
fi
for n_max in "${draft_lengths[@]}"; do
    if [[ ! "$n_max" =~ ^[0-9]+$ ]]; then
        echo "Draft lengths must be non-negative integers: $n_max" >&2
        exit 2
    fi
done

results="$(mktemp /tmp/glm-mtp-benchmark.XXXXXX)"
trap 'rm -f -- "$results"' EXIT

for n_max in "${draft_lengths[@]}"; do
    for ((sample = 1; sample <= samples; sample++)); do
        "$window_guard"
        payload="$(jq -cn \
            --arg model "$model" \
            --argjson max_tokens "$max_tokens" \
            --argjson n_max "$n_max" \
            --argjson spec_p_min "${spec_p_min:-null}" \
            --argjson temperature "$temperature" \
            '{
                model: $model,
                messages: [{
                    role: "user",
                    content: "Write a compact technical explanation of NUMA-aware mixture-of-experts inference. Continue until the output limit."
                }],
                max_tokens: $max_tokens,
                temperature: $temperature,
                seed: 42,
                cache_prompt: false,
                stream: false,
                "speculative.n_max": $n_max
            }
            | if $spec_p_min == null then . else .["speculative.p_min"] = $spec_p_min end')"

        response="$(curl -fsS --max-time 600 \
            -H 'Content-Type: application/json' \
            --data-binary "$payload" \
            "$url")"

        result="$(jq -ce \
            --argjson n_max "$n_max" \
            --argjson sample "$sample" \
            --argjson p_min "${spec_p_min:-null}" \
            --argjson temperature "$temperature" \
            '{
                n_max: $n_max,
                p_min: $p_min,
                temperature: $temperature,
                sample: $sample,
                prompt_tokens: .usage.prompt_tokens,
                output_tokens: .usage.completion_tokens,
                prompt_tok_s: .timings.prompt_per_second,
                decode_tok_s: .timings.predicted_per_second,
                draft_tokens: (.timings.draft_n // 0),
                accepted_tokens: (.timings.draft_n_accepted // 0)
            }
            | select(.output_tokens > 0 and .decode_tok_s > 0)' <<<"$response")"

        printf '%s\n' "$result" | tee -a "$results"
    done
done

jq -s '
    def mean: add / length;
    sort_by(.n_max)
    | group_by(.n_max)
    | map({
        n_max: .[0].n_max,
        p_min: .[0].p_min,
        temperature: .[0].temperature,
        samples: length,
        decode_mean: ([.[].decode_tok_s] | mean),
        decode_min: ([.[].decode_tok_s] | min),
        decode_max: ([.[].decode_tok_s] | max),
        prompt_mean: ([.[].prompt_tok_s] | mean),
        draft_tokens: ([.[].draft_tokens] | add),
        accepted_tokens: ([.[].accepted_tokens] | add),
        acceptance: (
            ([.[].draft_tokens] | add) as $drafts
            | if $drafts > 0 then ([.[].accepted_tokens] | add) / $drafts else 0 end
        )
    })
' "$results"
