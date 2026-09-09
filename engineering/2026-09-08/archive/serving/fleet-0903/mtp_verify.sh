#!/usr/bin/env bash
# Correctness harness for the new glm5next NextN graph.
#
# Speculative decoding is exactness-preserving at temperature 0: a drafted token
# is only emitted if it matches what the trunk would have produced. So the test
# is not "does it look right" -- it is "is the output byte-identical to the
# non-speculative run". The truncated model emits gibberish, which is *ideal*
# here: gibberish is still deterministic, and a graph bug shows up as a diff.
#
# A wrong NextN graph cannot corrupt output; it shows up as acceptance near zero.
# So this reports both: identical-or-not, and the accepted/drafted ratio.
set -u
BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm53-flash/build-sr950/bin/llama-server
MODEL=${MODEL:-/dev/shm/glm53f-8L-mtp.gguf}
PORT=18099
NODE=${NODE:-2}
PROMPT='The memory bandwidth of a four socket server is'
N=${N:-48}

start() { # $1 = label, rest = extra server args
  local label=$1; shift
  for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  ( setsid env NVIDIA_TF32_OVERRIDE=0 GGML_CPU_REPACK_LOAD_THREADS=16 \
      numactl --cpunodebind=$NODE --membind=$NODE -- \
      "$BIN" --host 127.0.0.1 --port $PORT --model "$MODEL" --alias trunc \
        --gpu-layers 0 --load-mode none --threads 32 --threads-batch 32 \
        --ctx-size 4096 --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 \
        --predict 4096 --temp 0 --timeout 600 --no-webui --metrics --log-timestamps "$@" \
        > /dev/shm/glm-dev/mtpv-$label.log 2>&1 < /dev/null & )
  local t0=$(date +%s)
  until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! pgrep -f "llama-server.*port $PORT" >/dev/null; then
      echo "DIED: $(tr -d '\000' < /dev/shm/glm-dev/mtpv-$label.log | grep -aoE 'GGML_ASSERT[^\n]{0,110}|error[^\n]{0,110}' | tail -1)"
      return 1
    fi
    [ $(( $(date +%s)-t0 )) -gt 300 ] && { echo "TIMEOUT"; return 1; }
    sleep 5
  done
}

gen() { # $1 = out file
  curl -s -m 600 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N,\"temperature\":0,\"seed\":42,\"cache_prompt\":false}" \
    > "$1"
  python3 -c "
import json,sys
d=json.load(open('$1'))
t=d.get('timings',{})
print('  %.2f tok/s | drafted %s accepted %s' % (
    t.get('predicted_per_second',0), t.get('draft_n','-'), t.get('draft_n_accepted','-')))
open('$1.content','w').write(d.get('content',''))"
}

echo "=== glm5next NextN correctness + speed  $(date -u +%FT%TZ) ==="
echo "[baseline, no speculation]"
start none && gen /dev/shm/glm-dev/mtpv-base.json

echo "[draft-mtp, n_max=3]"
start mtp3 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 \
           --spec-draft-threads 32 --spec-draft-threads-batch 32 && gen /dev/shm/glm-dev/mtpv-mtp3.json

echo "[draft-mtp, n_max=2]"
start mtp2 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
           --spec-draft-threads 32 --spec-draft-threads-batch 32 && gen /dev/shm/glm-dev/mtpv-mtp2.json

for p in $(pgrep -f "llama-server.*port $PORT" || true); do kill -9 "$p" 2>/dev/null; done

echo "=== EXACTNESS ==="
for v in mtp3 mtp2; do
  if [ -f /dev/shm/glm-dev/mtpv-$v.json.content ] && \
     cmp -s /dev/shm/glm-dev/mtpv-base.json.content /dev/shm/glm-dev/mtpv-$v.json.content; then
    echo "  $v: IDENTICAL to non-speculative output"
  else
    echo "  $v: *** DIFFERS from non-speculative output ***"
    head -c 120 /dev/shm/glm-dev/mtpv-$v.json.content 2>/dev/null; echo
  fi
done
echo "=== done $(date -u +%FT%TZ) ==="
