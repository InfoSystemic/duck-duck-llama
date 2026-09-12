#!/bin/bash
# Do the two 09-11 fixes generalise? DeepSeek-V4-Flash uses the SAME runtime as GLM-5.3-Flash,
# so the patched libggml-cpu applies. Baseline for comparison: 10.71 tok/s / 193.8 GB/s raw TP4.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
B=~/InfoSystemic/AI-Server/serving/fleet-0903
BIN=$B/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server
MODEL=/models/gguf/DeepSeek-V4-Flash-0731-UD-Q4_K_XL/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf
for prt in 18131 18132; do for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null|grep -oE 'pid=[0-9]+'|cut -d= -f2|sort -u); do kill $p 2>/dev/null; done; done
for i in $(seq 240); do ss -ltn 2>/dev/null|grep -qE ':(18131|18132) ' || break; sleep 1; done; sleep 3
set -a; while read -r l; do case "$l" in LD_LIBRARY_PATH=*) continue;; esac; [ -n "$l" ] && export "$l"; done < $B/results/goal-0910-restore/env.txt; set +a
export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $B/results/goal-0910-restore/env.txt|cut -d= -f2-)
export GGML_CPU_PARALLEL_UNARY=4096
LOG=$D/results/dsfix.log
nohup "$BIN" --host 127.0.0.1 --port 18132 --load-mode mmap --fit off --ctx-size 4096 \
  --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999 \
  --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1 \
  --jinja --no-webui --metrics --verbosity 2 --no-cache-prompt --threads 15 --threads-batch 15 \
  --model "$MODEL" --alias dsv4-flash > "$LOG" 2>&1 & SRV=$!
for i in $(seq 900); do s=$(curl -s -m 2 http://127.0.0.1:18132/health 2>/dev/null); [[ "$s" == *ok* ]] && break
  kill -0 $SRV 2>/dev/null || { echo DIED; grep -iE 'error|assert' "$LOG"|tail -3; exit 1; }; sleep 2; done
echo "UP"; grep -o '/[^ ]*libggml-cpu[^ ]*' /proc/$SRV/maps|sort -u|sed 's/^/  lib: /'
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
$D/bench2.sh 18132 dsfix 192
echo "=== DSFIX DONE (baseline was 10.71 tok/s / 193.8 GB/s / 51.0%) ==="
