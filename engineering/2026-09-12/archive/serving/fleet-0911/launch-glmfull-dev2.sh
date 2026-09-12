#!/bin/bash
# Exclusive-load GLM-5.3 Full on the runtime that actually accepts glm-dsa tensor-split
# (engines/llama.cpp-sr950-glm/build-dev2). MTP2 only — ngram-mod composite is a measured
# dead end. Production Flash on 18131 is stopped first; restore with finish.sh.
set -uo pipefail
D=~/InfoSystemic/AI-Server/serving/fleet-0911
SCRATCH=${SCRATCH:-/tmp/grok-goal-a5c314b1c84d/implementer}
LOG=$D/results/glmfull-dev2.log
BIN=/home/user/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/build-dev2/bin/llama-server
M=/models/GLM-5.3-GGUF/UD-Q4_K_XL/GLM-5.3-UD-Q4_K_XL-00001-of-00011.gguf
DR=/dev/shm/GLM-5.3-MTP-HYBRID-Q4L-Q6H-OUTQ4.gguf
if [[ ! -r "$DR" ]]; then
  DR=/models/GLM-5.3-GGUF/MTP/GLM-5.3-MTP-UD-Q4_K_XL.gguf
fi

# Stop whatever is on the exclusive ports; do not touch DeepSeek 18170.
for prt in 18131 18132 18133; do
  for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    kill "$p" 2>/dev/null || true
  done
done
for i in $(seq 240); do ss -ltn 2>/dev/null | grep -qE ':(18131|18132|18133) ' || break; sleep 1; done
sleep 3

# Drop Qwen/Flash knobs that would otherwise leak (GGML_Q4E_*, X16_Q8_EXPERTS, UNARY).
while read -r v; do unset "$v"; done < <(env | grep -oE '^(GGML|LLAMA|OMP|GOMP|KMP)[A-Z0-9_]*' | sort -u)
# v5b kernel/env flags (MTP-only profile). Do not source ngram composite.
set -a
# shellcheck disable=SC1091
source /home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.glm53-q4-fast-v5b.env
set +a
unset GGML_CPU_PARALLEL_UNARY GGML_GLM_DSA_DENSE GGML_CPU_X16_DUAL || true
export GGML_CPU_NUMA_THREADS="${GGML_CPU_NUMA_THREADS:-15}"

echo "--- per-node free before Full load ---"
numactl -H | grep free
echo "binary=$BIN"
echo "model=$M"
echo "draft=$DR"

# Even 1,1,1,1 is the pinned Full split. Fractional 1.41,1.43,1.43,1.00 aborted
# at ggml-backend-meta.cpp:949 (split size not divisible by Q4_K block granularity).
SPLIT=${GLM_FULL_TENSOR_SPLIT:-1,1,1,1}

nohup taskset -c 0-127 "$BIN" --host 127.0.0.1 --port 18131 --load-mode mmap --fit off --ctx-size 4096 \
  --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999 \
  --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor \
  --tensor-split "$SPLIT" \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --chat-template-file /home/user/InfoSystemic/AI-Server/serving/glm-sr950/chat-template-glm-5.3-llamacpp.jinja \
  --reasoning-format deepseek --reasoning-preserve --no-webui --metrics --verbosity 2 \
  --spec-type draft-mtp --spec-draft-model "$DR" --spec-draft-ngl all \
  --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --spec-draft-n-max 2 --spec-draft-n-default 2 --spec-draft-n-min 0 --spec-draft-p-min 0 \
  --threads 15 --threads-batch 15 --no-cache-prompt \
  --model "$M" --alias glm-sr950,glm-5.3,glm53,GLM-5.3,GLM-5.3-Full \
  > "$LOG" 2>&1 &
SRV=$!
echo "pid=$SRV loading GLM-5.3 Full from SATA (expect 15-40 min) log=$LOG"
T0=$(date +%s)
for i in $(seq 2400); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null || true)
  if [[ "$s" == *'"ok"'* ]]; then
    echo "UP after $(( $(date +%s)-T0 ))s rss=$(awk '/VmRSS/{printf "%.0fGB",$2/1048576}' /proc/$SRV/status 2>/dev/null)"
    exit 0
  fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "DIED/OOM"
    grep -iE 'error|assert|abort|alloc|oom|not implemented' "$LOG" | tail -20
    exit 1
  fi
  if [ $((i % 30)) -eq 0 ]; then
    echo "  ...still loading ${i}x2s rss=$(awk '/VmRSS/{printf "%.0fGB",$2/1048576}' /proc/$SRV/status 2>/dev/null)"
  fi
  sleep 2
done
echo TIMEOUT; tail -20 "$LOG"; exit 1
