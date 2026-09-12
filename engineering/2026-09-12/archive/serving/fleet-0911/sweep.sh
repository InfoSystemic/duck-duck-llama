#!/bin/bash
# sweep.sh <configfile>   configfile: one line per config:  <tag><TAB><ENV overrides>
D=~/InfoSystemic/AI-Server/serving/fleet-0911
TSV=$D/results/sweep-$(date +%m%d-%H%M).tsv
echo -e "tag\ttok_s\tgb_s\tpct380\tstatus" > $TSV
echo "TSV=$TSV"
while IFS=$'\t' read -r TAG OVR; do
  [[ -z "$TAG" || "$TAG" == \#* ]] && continue
  echo "### $TAG : $OVR"
  if ! $D/launch.sh "$OVR" > $D/results/launch-$TAG.log 2>&1; then
    echo -e "$TAG\t-\t-\t-\tLAUNCH_FAIL" >> $TSV; tail -3 $D/results/launch-$TAG.log; continue
  fi
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1
  sleep 3
  OUT=$($D/bench.sh 18131 "$TAG" 192 2>&1)
  echo "$OUT"
  SUMLINE=$(echo "$OUT" | grep '^  == ')
  TPS=$(echo "$SUMLINE" | sed -n 's/.*mean \([0-9.]*\) tok\/s.*/\1/p')
  GB=$(echo "$SUMLINE" | sed -n 's/.*tok\/s *\([0-9.]*\) GB\/s.*/\1/p')
  PC=$(echo "$SUMLINE" | sed -n 's/.*(\([0-9.]*\)% of 380).*/\1/p')
  echo -e "$TAG\t${TPS:--}\t${GB:--}\t${PC:--}\tok" >> $TSV
done < "$1"
echo "=== SWEEP DONE ==="; column -t $TSV
