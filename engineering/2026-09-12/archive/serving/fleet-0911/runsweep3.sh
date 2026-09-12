#!/bin/bash
# speculation-depth sweep: needs argv changes, so run configs individually
D=~/InfoSystemic/AI-Server/serving/fleet-0911
while pgrep -f 'bash ./sweep.sh' >/dev/null || pgrep -f 'bash ./runpar.sh' >/dev/null || pgrep -f 'bash ./runprof.sh' >/dev/null || pgrep -f 'bash ./runsweep2.sh' >/dev/null; do sleep 20; done
TSV=$D/results/sweep3-$(date +%H%M).tsv; echo -e "tag\ttok_s\tgb_s\tpct380" > $TSV; echo "TSV=$TSV"
run(){ TAG=$1; shift
  $D/launch.sh "GGML_CPU_DUMMY=0" "$@" > $D/results/launch-$TAG.log 2>&1 || { echo "$TAG LAUNCH_FAIL"; return; }
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench.sh 18131 $TAG 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$TAG\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's/.*tok\/s *\([0-9.]*\) GB.*/\1/p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
run nmax1 --spec-draft-n-max 1
echo "=== sweep3 done ==="; column -t $TSV
