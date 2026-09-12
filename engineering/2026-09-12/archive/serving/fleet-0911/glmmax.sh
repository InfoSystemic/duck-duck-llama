#!/bin/bash
# GLM-5.3-Flash max tok/s: ngram-mod composite + the UNARY/SCALE fix + concurrency, one load.
# ngram-mod's cache persists across requests, so every prompt below is distinct.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
for p in $(ss -ltnpH "sport = :18131" 2>/dev/null|grep -oE 'pid=[0-9]+'|cut -d= -f2|sort -u); do kill $p 2>/dev/null; done
for i in $(seq 240); do ss -ltn 2>/dev/null|grep -q ':18131 ' || break; sleep 1; done; sleep 3
set -a; while read -r l; do case "$l" in LD_LIBRARY_PATH=*) continue;; esac; [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $D/restore-env.txt|cut -d= -f2-)
export GGML_CPU_PARALLEL_UNARY=4096
# swap draft-mtp -> ngram-mod,draft-mtp and widen to 8 slots
mapfile -t A < <(grep -v '^$' $D/restore-argv.txt)
ARGV=(); for x in "${A[@]}"; do [ "$x" = "draft-mtp" ] && x="ngram-mod,draft-mtp"; ARGV+=("$x"); done
ARGV+=(--parallel 8 --ctx-size 32768)
LOG=$D/results/glmmax.log; nohup "${ARGV[@]}" > "$LOG" 2>&1 & SRV=$!
for i in $(seq 900); do s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null); [[ "$s" == *ok* ]] && break
  kill -0 $SRV 2>/dev/null || { echo DIED; grep -iE 'error|assert|unknown' "$LOG"|tail -4; exit 1; }; sleep 2; done
echo "UP (ngram-mod,draft-mtp + fix + parallel 8)"
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
echo "### single stream  (reference: draft-mtp+fix = 15.71 mean)"
$D/bench2.sh 18131 gmax1 192
echo "### concurrency  (reference: stock lib C=8 = 29.28 aggregate)"
for C in 4 8; do $D/benchpar.sh 18131 gmax$C $C 192; done
echo "=== GLMMAX DONE ==="
