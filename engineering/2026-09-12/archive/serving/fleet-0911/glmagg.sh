#!/bin/bash
# Definitive GLM aggregate: production draft-mtp n=2 + the UNARY/SCALE fix, 8 slots.
# Reports BOTH aggregate definitions, since they disagreed at C=8 in s.45.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
for p in $(ss -ltnpH "sport = :18131" 2>/dev/null|grep -oE 'pid=[0-9]+'|cut -d= -f2|sort -u); do kill $p 2>/dev/null; done
for i in $(seq 240); do ss -ltn 2>/dev/null|grep -q ':18131 ' || break; sleep 1; done; sleep 3
set -a; while read -r l; do case "$l" in LD_LIBRARY_PATH=*) continue;; esac; [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $D/restore-env.txt|cut -d= -f2-)
export GGML_CPU_PARALLEL_UNARY=4096
mapfile -t ARGV < <(grep -v '^$' $D/restore-argv.txt)
ARGV+=(--parallel 8 --ctx-size 32768)
LOG=$D/results/glmagg.log; nohup "${ARGV[@]}" > "$LOG" 2>&1 & SRV=$!
for i in $(seq 900); do s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null); [[ "$s" == *ok* ]] && break
  kill -0 $SRV 2>/dev/null || { echo DIED; tail -3 "$LOG"; exit 1; }; sleep 2; done
echo "UP (draft-mtp n=2 + fix + 8 slots)"
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
for C in 4 8; do
  $D/benchpar.sh 18131 gagg$C $C 192
  python3 - "$D" "gagg$C" "$C" <<'PY'
import json,sys,glob
D,TAG,C=sys.argv[1],sys.argv[2],int(sys.argv[3])
rates=[]; 
for f in sorted(glob.glob(f"{D}/results/par-{TAG}-*.json")):
    try: rates.append(json.load(open(f))['timings']['predicted_per_second'])
    except Exception: pass
if rates: print(f"     sum-of-per-stream: {sum(rates):.2f} tok/s  (each {min(rates):.2f}-{max(rates):.2f})")
PY
done
echo "=== GLMAGG DONE ==="
