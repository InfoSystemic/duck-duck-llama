#!/bin/bash
# Restore the box to its production state: GLM-5.3-Flash Q4 on 18131 with the exact
# 09-10 validated env (incl. LD_LIBRARY_PATH -> private-cpu + validated-chunk16-bin).
D=~/InfoSystemic/AI-Server/serving/fleet-0911
# Only kill servers listening on the ports THIS session owns. A bare "any process named
# llama-server" kill takes down other sessions' production servers (it killed the [client]
# client-portal assistant on 18097 on 09-11).
for prt in 18131 18132 18133; do
  for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill "$p" 2>/dev/null; done
done
for i in $(seq 240); do ss -ltn 2>/dev/null | grep -qE ':(18131|18132|18133) ' || break; sleep 1; done
for i in $(seq 120); do ss -ltn 2>/dev/null | grep -qE ':(18131|18132) ' || break; sleep 1; done
sleep 3
# Scrub GENERICALLY: unset every GGML_/LLAMA_/OMP_/GOMP_/KMP_ var in this shell that is not
# in the pinned env file. A hand-maintained list misses the next knob (it missed
# GGML_CPU_PARALLEL_UNARY after catching GGML_CPU_OP_PROFILE*).
while read -r v; do
  grep -q "^${v}=" "$D/restore-env.txt" || unset "$v"
done < <(env | grep -oE '^(GGML|LLAMA|OMP|GOMP|KMP)[A-Z0-9_]*' | sort -u)
nohup taskset -c 0-127 $D/restore-baseline-0911.sh > $D/results/restore.log 2>&1 &
SRV=$!; echo "restoring production GLM-5.3-Flash, pid=$SRV"
for i in $(seq 1200); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null)
  [[ "$s" == *'"ok"'* ]] && { echo "PRODUCTION RESTORED on 18131"; exit 0; }
  kill -0 "$SRV" 2>/dev/null || { echo "RESTORE DIED"; tail -8 $D/results/restore.log; exit 1; }
  sleep 2
done
echo "restore TIMEOUT"; tail -6 $D/results/restore.log; exit 1
