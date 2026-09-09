#!/usr/bin/env bash
# Replay-heavy benchmark: the agentic-coding pattern (re-emit a file with edits).
# This is the workload n-gram speculative decoding targets. Novel-prose suites
# (benchmark-workloads.sh) will not show its effect.
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

label="${1:-replay}"
url="${GLM_BENCH_URL:-http://127.0.0.1:18091/v1/chat/completions}"
model="${GLM_BENCH_MODEL:-glm-sr950}"
max_tokens="${GLM_BENCH_TOKENS:-400}"
spec_n_max="${GLM_BENCH_SPEC_N_MAX:-}"
spec_p_min="${GLM_BENCH_SPEC_P_MIN:-}"
out="${GLM_REPLAY_OUT:-$(mktemp /tmp/glm-replay.XXXXXX)}"

if [[ -n "$spec_n_max" && ! "$spec_n_max" =~ ^[0-9]+$ ]]; then
  echo "GLM_BENCH_SPEC_N_MAX must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "$spec_p_min" ]] && ! jq -en --arg value "$spec_p_min" \
    '$value | tonumber | . >= 0 and . <= 1' >/dev/null 2>&1; then
  echo "GLM_BENCH_SPEC_P_MIN must be a number between 0 and 1" >&2
  exit 2
fi

read -r -d '' SRC <<'SRCEOF' || true
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
  "Rename the function load_config to read_config everywhere. Output the COMPLETE modified file, unchanged except for that rename."
  'Add these exact one-line docstrings: load_config -> "Load a configuration file."; validate_config -> "Validate required configuration keys."; merge_config -> "Merge two configuration mappings." Output the COMPLETE modified file with everything else byte-identical.'
  "Change the exception in load_config from ValueError to ConfigError. Output the COMPLETE modified file, otherwise unchanged."
)

: >"$out"
for i in "${!tasks[@]}"; do
  "$window_guard"
  prompt="Here is a Python file:

\`\`\`python
$SRC
\`\`\`

${tasks[$i]}

Do not explain. Return exactly one Python code block."
  payload="$(jq -cn --arg m "$model" --arg p "$prompt" --argjson mt "$max_tokens" \
    --argjson spec_n_max "${spec_n_max:-null}" \
    --argjson spec_p_min "${spec_p_min:-null}" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,
      seed:(7100+0),reasoning_effort:"low",cache_prompt:false,stream:false}
      | if $spec_n_max == null then . else .["speculative.n_max"] = $spec_n_max end
      | if $spec_p_min == null then . else .["speculative.p_min"] = $spec_p_min end')"
  resp="$(curl -fsS --max-time 900 -H 'Content-Type: application/json' --data-binary "$payload" "$url")"
  jq -ce --arg label "$label" --argjson w "$((i+1))" '
   (.choices[0].message.content // "") as $content |
   {
     label:$label, workload:$w,
     prompt_tokens:.usage.prompt_tokens, output_tokens:.usage.completion_tokens,
     prompt_tok_s:.timings.prompt_per_second, decode_tok_s:.timings.predicted_per_second,
     predicted_ms:.timings.predicted_ms,
     draft_tokens:(.timings.draft_n // 0), accepted_tokens:(.timings.draft_n_accepted // 0),
     finish_reason:.choices[0].finish_reason,
     content_chars:($content | length),
     correct:(
       ($content | contains("def validate_config")) and
       ($content | contains("def merge_config")) and
       ($content | contains("return merged")) and
       (if $w == 1 then
          ($content | contains("def read_config")) and
          (($content | contains("def load_config")) | not)
        elif $w == 2 then
          ($content | contains("def load_config")) and
          ($content | contains("\"\"\"Load a configuration file.\"\"\"")) and
          ($content | contains("\"\"\"Validate required configuration keys.\"\"\"")) and
          ($content | contains("\"\"\"Merge two configuration mappings.\"\"\""))
        else
          ($content | contains("raise ConfigError")) and
          (($content | contains("raise ValueError")) | not)
        end)
     )
   }' <<<"$resp" | tee -a "$out"
done

jq -s '{label:.[0].label, workloads:length,
        output_tokens:([.[].output_tokens]|add),
        aggregate_decode_tok_s:(([.[].output_tokens]|add)*1000/([.[].predicted_ms]|add)),
        mean_decode_tok_s:(([.[].decode_tok_s]|add)/length),
        min_decode_tok_s:([.[].decode_tok_s]|min),
        max_decode_tok_s:([.[].decode_tok_s]|max),
        correct_workloads:([.[] | select(.correct)] | length),
        all_correct:all(.[]; .correct),
        draft_tokens:([.[].draft_tokens]|add),
        accepted_tokens:([.[].accepted_tokens]|add),
        draft_acceptance:(([.[].draft_tokens]|add) as $d | if $d>0 then ([.[].accepted_tokens]|add)/$d else 0 end)}' "$out"
echo "Detailed: $out" >&2

if ! jq -se 'all(.[]; .correct)' "$out" >/dev/null; then
  echo "Replay benchmark failed one or more correctness checks" >&2
  exit 1
fi
