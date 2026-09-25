#!/usr/bin/env bash
# One load, the whole p_min x draft-length surface, on BOTH an easy and a hard prompt.
#
# Why p_min is the lever that matters for a block drafter: a rejected row still costs a full row of expert bandwidth in
# the verify pass (measured: 36.8 ms per extra expert set on this host). On predictable text DFlash accepts the whole
# 8-token block and runs at 19-20 tok/s; on open prose it accepts 16-35% and the wasted rows drag it BELOW the
# no-speculation baseline. Truncating the block at the first low-confidence position should keep the first behaviour and
# remove the second, adaptively, per request.
set -uo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-18190}
NFILE=/tmp/dflash-draft-n
PFILE=/tmp/dflash-draft-pmin
SUMMARY=results/dflash-pmin-sweep.md
gib_for() { case "$1" in 1) echo 45.4;; 2) echo 55.3;; 3) echo 64.9;; 4) echo 74.3;; 5) echo 83.6;; 6) echo 92.7;; 7) echo 101.5;; *) echo 74.3;; esac; }

{ echo; echo "## DFlash p_min x draft length -- $(date '+%Y-%m-%d %H:%M')"; echo
  echo "| prompt | n | p_min | tok/s | tokens/cycle | acceptance | ms/cycle |"
  echo "|---|---:|---:|---:|---:|---:|---:|"; } >> "$SUMMARY"

probe() {  # probe <label> <n> <pmin> <request-json-file>
  local label=$1 n=$2 pm=$3 pf=$4
  echo "$n" > "$NFILE"; echo "$pm" > "$PFILE"
  # one throwaway request so any graph rebuild from the new draft length is not inside the timed one
  curl -s -m 900 -o /dev/null -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"hi","n_predict":4,"temperature":0,"cache_prompt":false}'
  curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @"$pf" \
  | LABEL="$label" N="$n" PM="$pm" python3 -c "
import json, os, sys
label, n, pm = os.environ['LABEL'], os.environ['N'], os.environ['PM']
try:
    t = json.load(sys.stdin)['timings']
except Exception:
    print(f'| {label} | {n} | {pm} | ERR | | | |'); raise SystemExit
tp = int(t.get('predicted_n', 0)); a = int(t.get('draft_n_accepted', 0) or 0); d = int(t.get('draft_n', 0) or 0)
cyc = max(1, tp - a)
print(f\"| {label} | {n} | {pm} | {t.get('predicted_per_second', 0):.2f} | {tp/cyc:.3f} | {(a/d if d else 0):.3f} | {t.get('predicted_ms', 0)/cyc:.1f} |\")
" >> "$SUMMARY"
  tail -1 "$SUMMARY"
}

cat > /tmp/p_hard.json <<'J'
{"prompt":"Explain in about 150 words why a speculative decoder's throughput is the product of its verify-cycle rate and the number of tokens accepted per cycle, and which of the two a memory-bandwidth limit constrains.","n_predict":160,"temperature":0,"seed":42,"cache_prompt":false,"stream":false}
J
cat > /tmp/p_code.json <<'J'
{"prompt":"def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\ndef factorial(n):\n","n_predict":160,"temperature":0,"seed":42,"cache_prompt":false,"stream":false}
J

for pm in 0.0 0.3 0.5 0.7 0.9; do
  for n in 7 4; do
    probe hard "$n" "$pm" /tmp/p_hard.json
    probe code "$n" "$pm" /tmp/p_code.json
  done
done
echo 7 > "$NFILE"; echo 0.0 > "$PFILE"
echo; echo "=== surface ==="; cat "$SUMMARY"
