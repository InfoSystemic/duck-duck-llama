#!/usr/bin/env bash
# sweep-requant.sh -- unattended attention-requant sweep for MiMo-V2.6-Pro at MTP depth 3.
#
# Why a driver rather than three hand-run launches: one tensor-parallel load of this model is 518 GiB off /models and takes
# 20-30 minutes with /health returning 503 the whole way, and teardown needs several more minutes to release the memory
# before the next load can start. Three configurations is therefore ~90 minutes of babysitting that nobody should do by hand.
#
# What it measures, per configuration:
#   golden.py     greedy completions + top-5 logprobs, compared against the Q8_0 run -- requant is NOT bit-exact, so a
#                 speed number without this gate means nothing
#   spec-detail   decode rate split into verify-cycle rate and tokens accepted per cycle, which says whether a
#                 disappointing result is bandwidth or acceptance
#   smoke         the standard --speed-only record in results/
#
# Servers are stopped by LISTENING PORT, never by process name: other sessions run llama-servers on this box.
#
#   ./sweep-requant.sh                 q8_0 (as shipped, the golden baseline), then q6_K, then q5_K
#   ./sweep-requant.sh q6_K q5_K q4_K  an explicit list
#   SKIP_BASELINE=1 ./sweep-requant.sh q6_K     reuse an existing results/golden-mtp3-q8_0.json
set -uo pipefail

cd "$(dirname "$0")"
PORT=${PORT:-18190}
NMAX=${NMAX:-3}
NPRED=${NPRED:-192}
GOLD_N=${GOLD_N:-48}
RES=results
SUMMARY=$RES/requant-sweep-mtp$NMAX.md
BASE_TAG=mtp$NMAX-q8_0
DEPTH_FILE=${DEPTH_FILE:-/tmp/mimo-draft-n}
mkdir -p "$RES"

# GiB read per verify cycle at depth 3, from the tensor table in BANDWIDTH-CEILING-20260921.md: 59.3 with attention at
# Q8_0 (19.3 GiB of it), less what the requant saves on those two tensors per layer.
gib_for() { case "$1" in q8_0) echo 59.3;; q6_K) echo 54.9;; q5_K) echo 52.5;; q4_K) echo 50.2;; *) echo 59.3;; esac; }

server_pid() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

stop_server() {
  local pid; pid=$(server_pid)
  [[ -z "${pid:-}" ]] && return 0
  echo "  stopping pid $pid (by listening port $PORT)"
  kill "$pid" 2>/dev/null
  # the port frees at once but releasing 518 GiB takes minutes; the next load fights for memory if we do not wait
  for _ in $(seq 1 600); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  for _ in $(seq 1 300); do
    local avail; avail=$(free -g | awk 'NR==2 {print $7}')
    (( avail > 600 )) && { echo "  memory released (${avail} GiB available)"; return 0; }
    sleep 5
  done
  echo "  WARNING: memory never came back above 600 GiB available; continuing anyway"
}

# The load is invisible to /health (503 throughout) but perfectly visible in the process's read_bytes, so report progress
# from that and fail fast if the process dies instead of waiting out a timeout.
wait_healthy() {
  local pid=$1 log=$2 last=0
  for i in $(seq 1 240); do
    if [[ "$(curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" == *'"ok"'* ]]; then
      echo "  healthy after $(( i * 15 / 60 )) min"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  DIED during load; last errors:"
      grep -a -iE "error|assert|abort|failed" "$log" | grep -v "unused tensor" | tail -5 | cut -c1-165
      return 1
    fi
    local gib; gib=$(( $(awk '/^read_bytes/{print $2}' "/proc/$pid/io" 2>/dev/null || echo 0) / 1073741824 ))
    (( gib - last >= 50 )) && { echo "  loading: ${gib} GiB read"; last=$gib; }
    sleep 15
  done
  echo "  TIMEOUT waiting for health"; return 1
}

run_one() {
  local rq=$1; local tag=mtp$NMAX-$rq; local log=/models/mimo-$rq-mtp$NMAX-launch.log
  echo "=== $tag ==="
  stop_server
  # LLAMA_MTP_DRAFT_N_FILE is read on EVERY draft call and clamps n_max to min(configured, file) -- so launching at the
  # deepest setting and lowering it between measurements gives the whole depth curve from ONE 25-minute load. A change
  # costs one graph rebuild. (common/speculative.cpp: the getenv is a function-static, so it must be set at launch.)
  echo "$NMAX" > "$DEPTH_FILE"
  # BUILD must be the tree that carries the h_nextn post-norm fix, or the draft head reads a mis-scaled hidden state
  # and every number below measures a broken drafter instead of the requant.
  local env_args=(SPEC=mtp "NMAX=$NMAX" "LLAMA_MTP_DRAFT_N_FILE=$DEPTH_FILE" "BUILD=${BUILD:-build-mtpfix}")
  [[ "$rq" != q8_0 ]] && env_args+=("GGML_CPU_ATTN_REQUANT=$rq")
  # OUT_REQUANT stacks the output head on top: at depth 3 the LM head is read once per draft plus once in the verify,
  # 3.7 GiB of a 59.3 GiB cycle, so it is the next-largest block of bytes after attention and the experts.
  [[ -n "${OUT_REQUANT:-}" ]] && { env_args+=("GGML_CPU_OUTPUT_REQUANT=$OUT_REQUANT"); tag="$tag-out$OUT_REQUANT"; log=/models/mimo-$rq-out$OUT_REQUANT-mtp$NMAX-launch.log; }
  echo "  launching: ${env_args[*]}"
  nohup env "${env_args[@]}" ./launch-mimo-tp.sh > "$log" 2>&1 &
  local pid=$!
  sleep 20
  # launch-mimo-tp.sh execs the binary, so the background pid IS the server once it is up
  wait_healthy "$pid" "$log" || { echo "| $rq | - | LOAD FAILED | | | |" >> "$SUMMARY"; return 1; }

  ./numa-placement.sh "$pid" 2>&1 | tail -3
  "$PY" golden.py "$PORT" "$RES/golden-$tag.json" "$GOLD_N" 2>&1 | tail -4
  local parity="baseline"
  if [[ "$rq" != q8_0 ]]; then
    parity=$("$PY" golden.py --compare "$RES/golden-$BASE_TAG.json" "$RES/golden-$tag.json" 2>&1 | tr '\n' '; ')
    echo "  parity vs $BASE_TAG: $parity"
  fi

  for d in $(seq "$NMAX" -1 "${DEPTH_MIN:-2}"); do
    echo "$d" > "$DEPTH_FILE"
    # one warm request so the graph rebuild is not inside the timed one
    curl -s -m 600 -o /dev/null -X POST "http://127.0.0.1:$PORT/completion" \
      -H 'Content-Type: application/json' -d '{"prompt":"hi","n_predict":8,"temperature":0,"cache_prompt":false}'
    local detail; detail=$("$PY" spec-detail.py "$PORT" "$NPRED" "$(gib_for "$rq")" 2>&1)
    echo "  -- depth $d --"; echo "$detail" | sed 's/^/  /'
    echo "$detail" > "$RES/spec-detail-$tag-d$d.txt"
    local rate tpc bw
    rate=$(echo "$detail" | awk '/^decode/ {print $2}')
    tpc=$(echo "$detail"  | awk '/^tokens\/cycle/ {print $2}')
    bw=$(echo "$detail"   | awk '/^implied bandwidth/ {print $3, $4}')
    printf '| %s | %s | %s | %s | %s | %s |\n' "$rq" "$d" "$rate" "$tpc" "$bw" "$parity" >> "$SUMMARY"
  done

  echo "$NMAX" > "$DEPTH_FILE"
  "$PY" smoke-mimo.py "$PORT" "$tag" --speed-only 2>&1 | tail -4
}

PY=$(command -v python3)
TARGETS=("$@")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(q8_0 q6_K q5_K)
[[ -n "${SKIP_BASELINE:-}" ]] && TARGETS=("${TARGETS[@]/q8_0/}")

{ echo; echo "## attention requant x MTP depth -- $(date '+%Y-%m-%d %H:%M')"; echo
  echo "| attention | depth | tok/s | tokens/cycle | implied bandwidth | parity vs $BASE_TAG |"
  echo "|---|---:|---:|---:|---|---|"; } >> "$SUMMARY"

for rq in "${TARGETS[@]}"; do [[ -n "$rq" ]] && run_one "$rq"; done
stop_server
echo; echo "=== sweep done; summary in $SUMMARY ==="
cat "$SUMMARY"
