#!/usr/bin/env bash
# Measure useful concurrent decode throughput. Aggregate decode tok/s is the
# sum of all generated tokens divided by the longest overlapping decode time;
# wall tok/s also includes prompt ingestion and request setup.
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

label="${1:-concurrent}"
url="${GLM_BENCH_URL:-http://127.0.0.1:18091/v1/chat/completions}"
model="${GLM_BENCH_MODEL:-glm-sr950}"
streams="${GLM_BENCH_STREAMS:-4}"
max_tokens="${GLM_BENCH_TOKENS:-400}"
spec_n_max="${GLM_BENCH_SPEC_N_MAX:-0}"
spec_p_min="${GLM_BENCH_SPEC_P_MIN:-0.8}"
out="${GLM_CONCURRENT_OUT:-$(mktemp /tmp/glm-concurrent.XXXXXX.jsonl)}"

if [[ ! "$streams" =~ ^[1-8]$ ]]; then
    echo "GLM_BENCH_STREAMS must be between 1 and 8" >&2
    exit 2
fi

read -r -d '' source_file <<'SRCEOF' || true
def load_config(path):
    with open(path) as f:
        raw = f.read()
    cfg = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("bad line: %s" % line)
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg

def validate_config(cfg, required):
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError("missing keys: %s" % ", ".join(sorted(missing)))
    return True

def merge_config(base, override):
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged
SRCEOF

tasks=(
    "Rename load_config to read_config everywhere."
    'Add the one-line docstring "Load a configuration file." to load_config.'
    "Change the ValueError raised by load_config to ConfigError."
    "Add a return type annotation of dict to load_config."
    'Add the one-line docstring "Validate required configuration keys." to validate_config.'
    'Add the one-line docstring "Merge two configuration mappings." to merge_config.'
    "Add a return type annotation of bool to validate_config."
    "Add a return type annotation of dict to merge_config."
)

tmp_dir="$(mktemp -d /tmp/glm-concurrent-run.XXXXXX)"
trap 'rm -rf -- "$tmp_dir"' EXIT
: >"$out"
start_ns="$(date +%s%N)"

for ((i = 0; i < streams; i++)); do
    prompt="Here is a Python file:

\`\`\`python
$source_file
\`\`\`

${tasks[$i]} Output the COMPLETE modified file and nothing else except one Python code block."
    jq -cn \
        --arg model "$model" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$max_tokens" \
        --argjson seed "$((8200 + i))" \
        --argjson spec_n_max "$spec_n_max" \
        --argjson spec_p_min "$spec_p_min" \
        '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,
          temperature:0,seed:$seed,reasoning_effort:"low",cache_prompt:false,stream:false,
          "speculative.n_max":$spec_n_max,"speculative.p_min":$spec_p_min}' \
        >"$tmp_dir/request-$i.json"
    (
        curl -fsS --max-time 900 -H 'Content-Type: application/json' \
            --data-binary "@$tmp_dir/request-$i.json" "$url" \
            >"$tmp_dir/response-$i.json"
    ) &
done

status=0
for job in $(jobs -p); do
    wait "$job" || status=1
done
end_ns="$(date +%s%N)"
if (( status != 0 )); then
    echo "One or more concurrent requests failed" >&2
    exit 1
fi

for ((i = 0; i < streams; i++)); do
    jq -ce --arg label "$label" --argjson stream "$((i + 1))" '
      (.choices[0].message.content // "") as $content |
      {label:$label,stream:$stream,
       prompt_tokens:.usage.prompt_tokens,output_tokens:.usage.completion_tokens,
       prompt_tok_s:.timings.prompt_per_second,decode_tok_s:.timings.predicted_per_second,
       prompt_ms:.timings.prompt_ms,predicted_ms:.timings.predicted_ms,
       draft_tokens:(.timings.draft_n // 0),accepted_tokens:(.timings.draft_n_accepted // 0),
       finish_reason:.choices[0].finish_reason,content_chars:($content|length),
       correct:(($content|contains("def validate_config")) and
                ($content|contains("def merge_config")) and
                ($content|contains("return merged")))}' \
       "$tmp_dir/response-$i.json" | tee -a "$out"
done

wall_ms="$(( (end_ns - start_ns) / 1000000 ))"
jq -s --argjson wall_ms "$wall_ms" '
  {label:.[0].label,streams:length,
   prompt_tokens:([.[].prompt_tokens]|add),output_tokens:([.[].output_tokens]|add),
   wall_ms:$wall_ms,
   wall_output_tok_s:(([.[].output_tokens]|add)*1000/$wall_ms),
   concurrent_decode_tok_s:(([.[].output_tokens]|add)*1000/([.[].predicted_ms]|max)),
   mean_stream_decode_tok_s:([.[].decode_tok_s]|add/length),
   min_stream_decode_tok_s:([.[].decode_tok_s]|min),
   max_stream_decode_tok_s:([.[].decode_tok_s]|max),
   all_correct:all(.[];.correct),
   draft_tokens:([.[].draft_tokens]|add),accepted_tokens:([.[].accepted_tokens]|add),
   draft_acceptance:(([.[].draft_tokens]|add) as $d |
     if $d > 0 then ([.[].accepted_tokens]|add)/$d else 0 end)}' "$out"
echo "Detailed: $out" >&2
