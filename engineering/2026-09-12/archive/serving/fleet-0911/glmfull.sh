#!/bin/bash
# GLM-5.3 FULL (436 GB) with an UNEVEN tensor-split compensating for the node-3 tmpfs skew.
# The Flash server is stopped first, so nothing else is resident: an OOM can only kill this load.
# Production Flash is restored unconditionally on exit.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
for p in $(ss -ltnpH "sport = :18131" 2>/dev/null|grep -oE 'pid=[0-9]+'|cut -d= -f2|sort -u); do kill $p 2>/dev/null; done
for i in $(seq 300); do ss -ltn 2>/dev/null|grep -q ':18131 ' || break; sleep 1; done; sleep 5
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
echo "--- per-node free before load ---"; numactl -H | grep free
set -a; while read -r l; do [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
export GGML_CPU_PARALLEL_UNARY=4096
export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $D/restore-env.txt|cut -d= -f2-)
BIN=~/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server
M=/models/GLM-5.3-GGUF/UD-Q4_K_XL/GLM-5.3-UD-Q4_K_XL-00001-of-00011.gguf
DR=/models/GLM-5.3-GGUF/MTP/GLM-5.3-MTP-UD-Q4_K_XL.gguf
LOG=$D/results/glmfull.log
nohup "$BIN" --host 127.0.0.1 --port 18131 --load-mode mmap --fit off --ctx-size 4096 \
  --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999 \
  --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor \
  --tensor-split 1.41,1.43,1.43,1.00 \
  --jinja --reasoning-format deepseek --reasoning-preserve --no-webui --metrics --verbosity 2 \
  --spec-type draft-mtp --spec-draft-model "$DR" --spec-draft-ngl all \
  --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --threads 15 --threads-batch 15 --no-cache-prompt --model "$M" --alias GLM-5.3-Full > "$LOG" 2>&1 & SRV=$!
echo "pid=$SRV loading 436 GB from SATA (expect 15-30 min)"
for i in $(seq 2400); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null); [[ "$s" == *ok* ]] && { echo "UP after ${i}x2s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "DIED/OOM"; grep -iE 'error|assert|abort|alloc|oom' "$LOG"|tail -5; exit 1; }
  [ $((i % 150)) -eq 0 ] && echo "  ...still loading, rss=$(awk '/VmRSS/{printf "%.0fGB",$2/1048576}' /proc/$SRV/status 2>/dev/null)"
  sleep 2
done
numactl -H | grep free | sed 's/^/  after: /'
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
$D/bench2.sh 18131 glmfull 192
echo "=== GLMFULL DONE ==="
