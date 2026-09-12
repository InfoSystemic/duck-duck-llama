#!/bin/bash
D=~/InfoSystemic/AI-Server/serving/fleet-0911
TSV=$D/results/qwen-pinned-$(date +%H%M).tsv; echo -e "tag\ttok_s\tgb_s\tpct380" > $TSV; echo "TSV=$TSV"
one(){ TAG=$1; MODE=$2
  echo "### $TAG (mode=$MODE)"
  $D/launch-qwen.sh "$MODE" > $D/results/launch-$TAG.log 2>&1 || { echo "$TAG LAUNCH_FAIL"; tail -6 $D/results/launch-$TAG.log; return; }
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench2.sh 18131 $TAG 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$TAG\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's|.*tok/s *\([0-9.]*\) GB.*|\1|p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
one qwen-pinned-mtp4 spec
one qwen-pinned-raw  raw
echo "=== QWEN DONE ==="; column -t $TSV
