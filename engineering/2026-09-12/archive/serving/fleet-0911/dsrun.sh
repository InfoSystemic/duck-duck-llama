#!/bin/bash
# DeepSeek-V4-Flash on the tuned engine (port 18132). Deliberately NOT using ngram-mod:
# its n-gram cache persists across requests and inflates repeat prompts (09-10 trap).
D=~/InfoSystemic/AI-Server/serving/fleet-0911
L=~/InfoSystemic/AI-Server/serving/fleet-0903/launch_dsv4flash_tuned_0910.sh
DRAFT=/models/deepseek-v4-tuning/DSV4-Flash-DSpark-draft-DS2A-stage0-IQ3S-protected.gguf
TSV=$D/results/dsv4-$(date +%H%M).tsv; echo -e "tag\ttok_s\tgb_s\tpct380" > $TSV; echo "TSV=$TSV"
one(){ TAG=$1; shift
  echo "### $TAG  args=[$*]"
  "$L" "$@" > $D/results/launch-$TAG.log 2>&1 || { echo "$TAG LAUNCH_FAIL"; tail -8 $D/results/launch-$TAG.log; return; }
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench2.sh 18132 $TAG 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$TAG\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's|.*tok/s *\([0-9.]*\) GB.*|\1|p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
one dsv4-raw  tensor
one dsv4-spec tensor draft-dspark "$DRAFT" 2
echo "=== DSV4 DONE ==="; column -t $TSV
