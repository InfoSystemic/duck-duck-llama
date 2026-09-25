#!/usr/bin/env bash
# sweep-dflash.sh [attn_requant ...] -- one load, the whole DFlash draft-length curve, optionally per attention requant.
#
# Why the curve matters rather than just "use the biggest block": every extra verify row costs expert bandwidth, because
# 8 more of the 384 experts light up per layer per row, while the attention, router and output head are read once per
# cycle no matter how many rows there are. So the bytes per cycle grow with the block while the tokens per cycle only grow
# as far as acceptance holds. There is an optimum somewhere below block_size, and SGLang's own MiMo recipe runs 4 draft
# tokens, not 8 -- so measure, do not assume the maximum is best.
#
# Requires the build that reads LLAMA_SPEC_DRAFT_N_FILE on every draft call (patches/dflash-runtime-draft-n.patch,
# compiled into engines/llama.cpp-mimo-tp/build-dflashn). Without it every draft length is a separate 30-minute load.
#
#   ./sweep-dflash.sh              draft lengths 7..2 on the shipped Q8_0 attention
#   ./sweep-dflash.sh q8_0 q4_K    the same curve twice, second time with attention requantised (gated on golden parity)
set -uo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-18190}
NMAX=${NMAX:-7}
NPRED=${NPRED:-192}
RES=results
SUMMARY=$RES/dflash-sweep.md
NFILE=${NFILE:-/tmp/dflash-draft-n}
BUILD=${BUILD:-build-dflashn}
BASE_GOLDEN=$RES/golden-mtp3-q8_0.json     # non-speculative reference text; speculation must not change it
mkdir -p "$RES"

# GiB read per verify cycle, from the model's tensor table: attention 19.3 + router/head/dense 3.0 + a DFlash forward 2.7,
# all read once per cycle, plus expert bytes that grow with the row count as 384*(1-(1-8/384)^rows)/8 * 10.3 GiB.
gib_for() { case "$1" in 1) echo 45.4;; 2) echo 55.3;; 3) echo 64.9;; 4) echo 74.3;; 5) echo 83.6;; 6) echo 92.7;; 7) echo 101.5;; *) echo 74.3;; esac; }

server_pid() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

stop_server() {
  local pid; pid=$(server_pid)
  [[ -z "${pid:-}" ]] && return 0
  echo "  stopping pid $pid (by listening port $PORT)"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 600); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  for _ in $(seq 1 300); do
    local avail; avail=$(free -g | awk 'NR==2 {print $7}')
    (( avail > 600 )) && { echo "  memory released (${avail} GiB available)"; return 0; }
    sleep 5
  done
  echo "  WARNING: memory never came back above 600 GiB available; continuing anyway"
}

wait_healthy() {
  local pid=$1 log=$2 last=0
  for i in $(seq 1 240); do
    [[ "$(curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" == *'"ok"'* ]] && { echo "  healthy after ~$(( i / 4 )) min"; return 0; }
    kill -0 "$pid" 2>/dev/null || {
      echo "  DIED during load:"; grep -a -iE "error|assert|abort|failed" "$log" | grep -v "unused tensor" | tail -6 | cut -c1-170; return 1; }
    local gib; gib=$(( $(awk '/^read_bytes/{print $2}' "/proc/$pid/io" 2>/dev/null || echo 0) / 1073741824 ))
    (( gib - last >= 100 )) && { echo "  loading: ${gib} GiB"; last=$gib; }
    sleep 15
  done
  echo "  TIMEOUT"; return 1
}

run_one() {
  local rq=$1 tag=dflash-$rq log=/models/mimo-dflash-$rq-launch.log
  echo "=== $tag ==="
  stop_server
  echo "$NMAX" > "$NFILE"
  local env_args=(SPEC=dflash "NMAX=$NMAX" "BUILD=$BUILD" "LLAMA_SPEC_DRAFT_N_FILE=$NFILE")
  [[ "$rq" != q8_0 ]] && env_args+=("GGML_CPU_ATTN_REQUANT=$rq")
  echo "  launching: ${env_args[*]}"
  nohup env "${env_args[@]}" ./launch-mimo-tp.sh > "$log" 2>&1 &
  local pid=$!
  sleep 20
  wait_healthy "$pid" "$log" || { echo "| $rq | LOAD FAILED | | | |" >> "$SUMMARY"; return 1; }

  # canary: /health only proves the weights loaded. A graph that cannot be built aborts on the FIRST decode, and finding
  # that out through a measurement script means watching every request fail against a dead socket.
  curl -s -m 300 -o /dev/null -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}'
  kill -0 "$pid" 2>/dev/null || { echo "  DIED on the first decode:"; grep -a -iE "assert|abort" "$log" | tail -3 | cut -c1-180
    echo "| $rq | FIRST DECODE ABORTED | | | |" >> "$SUMMARY"; return 1; }

  python3 golden.py "$PORT" "$RES/golden-$tag.json" 48 2>&1 | tail -3
  local parity
  parity=$(python3 golden.py --compare "$BASE_GOLDEN" "$RES/golden-$tag.json" 2>&1 | tr '\n' '; ')
  echo "  parity vs the non-speculative golden: $parity"

  for d in $(seq "$NMAX" -1 "${NMIN:-2}"); do
    echo "$d" > "$NFILE"
    curl -s -m 900 -o /dev/null -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"hi","n_predict":8,"temperature":0,"cache_prompt":false}'
    local detail; detail=$(python3 spec-detail.py "$PORT" "$NPRED" "$(gib_for "$d")" 2>&1)
    echo "  -- draft n=$d --"; echo "$detail" | head -7 | sed 's/^/  /'
    echo "$detail" > "$RES/spec-detail-$tag-n$d.txt"
    printf '| %s | %s | %s | %s | %s | %s |\n' "$rq" "$d" \
      "$(echo "$detail" | awk '/^decode/ {print $2}')" \
      "$(echo "$detail" | awk '/^tokens\/cycle/ {print $2}')" \
      "$(echo "$detail" | awk '/^draft acceptance/ {print $4}')" \
      "$parity" >> "$SUMMARY"
  done
  echo "$NMAX" > "$NFILE"
}

TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(q8_0)
{ echo; echo "## DFlash draft length x attention requant -- $(date '+%Y-%m-%d %H:%M')"; echo
  echo "| attention | draft n | tok/s | tokens/cycle | acceptance | parity vs no-spec |"
  echo "|---|---:|---:|---:|---:|---|"; } >> "$SUMMARY"
for rq in "${TARGETS[@]}"; do run_one "$rq"; done
echo; echo "=== done; summary in $SUMMARY ==="; cat "$SUMMARY"
