#!/bin/bash
# Profile with the patched library that separates COMPUTE time from BARRIER wait.
# Production is restored unconditionally on exit.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
for prt in 18131; do for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2|sort -u); do kill "$p" 2>/dev/null; done; done
for i in $(seq 240); do ss -ltn 2>/dev/null | grep -q ':18131 ' || break; sleep 1; done; sleep 3
set -a; while read -r l; do case "$l" in LD_LIBRARY_PATH=*) continue;; esac; [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
# patched lib FIRST, then the production path unchanged
export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $D/restore-env.txt | cut -d= -f2-)
export GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=/dev/shm/bwprobe/opprof.arm GGML_CPU_OP_PROFILE_COUNT=3 GGML_CPU_OP_PROFILE_SKIP=2
mapfile -t ARGV < <(grep -v '^$' $D/restore-argv.txt)
LOG=$D/results/profsplit-$(date +%H%M%S).log
nohup "${ARGV[@]}" > "$LOG" 2>&1 & SRV=$!
echo "pid=$SRV log=$LOG"
for i in $(seq 900); do s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null); [[ "$s" == *ok* ]] && break
  kill -0 $SRV 2>/dev/null || { echo DIED; grep -iE 'error|assert|abort' "$LOG"|tail -4; exit 1; }; sleep 2; done
echo "UP"; grep -o '/[^ ]*libggml-cpu[^ ]*' /proc/$SRV/maps | sort -u | sed 's/^/  lib: /'
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 2
MARK=$(wc -l < "$LOG"); touch /dev/shm/bwprobe/opprof.arm
curl -s -m 200 http://127.0.0.1:18131/completion -H 'Content-Type: application/json' -d '{"prompt":"Count from one to thirty in words.","n_predict":24,"temperature":0,"cache_prompt":false}' >/dev/null
rm -f /dev/shm/bwprobe/opprof.arm
tail -n +$((MARK+1)) "$LOG" | grep CPU_OP_PROFILE > $D/results/profsplit.txt
echo "captured $(wc -l < $D/results/profsplit.txt) lines"
echo "=== PROFSPLIT DONE ==="
