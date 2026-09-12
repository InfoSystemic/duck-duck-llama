#!/bin/bash
# GLM-5.3-Flash Q4 with speculation REMOVED (max-bandwidth config), same env otherwise.
set -uo pipefail
D=~/InfoSystemic/AI-Server/serving/fleet-0911
for prt in 18131 18132 18133; do for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill "$p" 2>/dev/null; done; done
for i in $(seq 240); do ss -ltn 2>/dev/null | grep -qE ':(18131|18132|18133) ' || break; sleep 1; done
sleep 3
set -a; while read -r l; do [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
LOG=$D/results/glmraw-$(date +%H%M%S).log
T0=$(date +%s)
nohup /home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server \
  --host \
  127.0.0.1 \
  --port \
  18131 \
  --load-mode \
  mmap \
  --fit \
  off \
  --ctx-size \
  4096 \
  --flash-attn \
  on \
  --batch-size \
  512 \
  --ubatch-size \
  256 \
  --parallel \
  1 \
  --gpu-layers \
  999 \
  --device \
  CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --split-mode \
  tensor \
  --tensor-split \
  1,1,1,1 \
  --jinja \
  --reasoning-format \
  deepseek \
  --reasoning-preserve \
  --no-webui \
  --metrics \
  --verbosity \
  2 \
  --threads \
  15 \
  --threads-batch \
  15 \
  --no-cache-prompt \
  --model \
  /home/user/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --alias \
  glm-flash-goal,glm-flash-q4,GLM-5.3-Flash > "$LOG" 2>&1 &
SRV=$!
echo "pid=$SRV log=$LOG"
for i in $(seq 1200); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null)
  [[ "$s" == *'"ok"'* ]] && { echo "UP in $(( $(date +%s)-T0 ))s"; exit 0; }
  kill -0 "$SRV" 2>/dev/null || { echo DIED; grep -iE "error|assert|abort|fail" "$LOG" | tail -5; exit 1; }
  sleep 2
done
echo TIMEOUT; exit 1
