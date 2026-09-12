#!/bin/bash
# Speculation depth vs BANDWIDTH. nmax1=188.5, nmax2=222.2 GB/s -> deeper verify batches make
# every op bigger. Never tested beyond 2 with a correct window analyser.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
TSV=$D/results/abdepth.tsv; echo -e "nmax\ttok_s\tgb_s\tpct380" > $TSV
run(){ N=$1
  echo "### nmax=$N"
  $D/launch.sh "GGML_CPU_DUMMY=0" --spec-draft-n-max $N > $D/results/launch-d$N.log 2>&1 || { echo "d$N LAUNCH_FAIL"; tail -4 $D/results/launch-d$N.log; return; }
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench2.sh 18131 d$N 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$N\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's|.*tok/s *\([0-9.]*\) GB.*|\1|p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
run 4
run 8
echo "=== DEPTH DONE ==="; column -t $TSV
