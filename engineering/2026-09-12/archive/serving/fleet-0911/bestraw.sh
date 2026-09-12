#!/bin/bash
# GLM-5.3-Flash raw (max-bandwidth config) + the patched library with UNARY parallelisation.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
for p in $(ss -ltnpH "sport = :18131" 2>/dev/null|grep -oE 'pid=[0-9]+'|cut -d= -f2|sort -u); do kill $p 2>/dev/null; done
for i in $(seq 240); do ss -ltn 2>/dev/null|grep -q ':18131 ' || break; sleep 1; done; sleep 3
set -a; while read -r l; do case "$l" in LD_LIBRARY_PATH=*) continue;; esac; [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $D/restore-env.txt|cut -d= -f2-)
export GGML_CPU_PARALLEL_UNARY=4096
# strip the speculation flags -> raw decode
mapfile -t A < <(grep -v '^$' $D/restore-argv.txt)
ARGV=(); skip=0
for x in "${A[@]}"; do
  case "$x" in --spec-type|--spec-draft-model|--spec-draft-n-max|--spec-draft-p-min|--spec-draft-ngl|--spec-draft-device) skip=1; continue;; esac
  if [ $skip -eq 1 ]; then case "$x" in --*) skip=0;; *) continue;; esac; fi
  ARGV+=("$x")
done
LOG=$D/results/bestraw.log; nohup "${ARGV[@]}" > "$LOG" 2>&1 & SRV=$!
for i in $(seq 900); do s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null); [[ "$s" == *ok* ]] && break
  kill -0 $SRV 2>/dev/null || { echo DIED; grep -iE 'error|assert' "$LOG"|tail -3; exit 1; }; sleep 2; done
echo "UP (raw + UNARY parallel)"
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
$D/bench2.sh 18131 bestraw 192
echo "=== BESTRAW DONE ==="
