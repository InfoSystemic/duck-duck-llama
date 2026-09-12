#!/bin/bash
# Sequential driver for the remaining GLM-5.3-Flash work. No pgrep predicates.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
TSV=$D/results/drive-$(date +%H%M).tsv; echo -e "tag\ttok_s\tgb_s\tpct380" > $TSV
echo "TSV=$TSV"
single(){ TAG=$1; OVR=$2; shift 2
  echo "### $TAG  env=[$OVR] args=[$*]"
  $D/launch.sh "$OVR" "$@" > $D/results/launch-$TAG.log 2>&1 || { echo "$TAG LAUNCH_FAIL"; tail -3 $D/results/launch-$TAG.log; return; }
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench2.sh 18131 $TAG 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$TAG\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's|.*tok/s *\([0-9.]*\) GB.*|\1|p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
# 1) CONCURRENCY: one load, four concurrency levels (the decisive duty-cycle test)
echo "### concurrency (--parallel 8 --ctx-size 32768)"
if $D/launch.sh "GGML_CPU_DUMMY=0" --parallel 8 --ctx-size 32768 > $D/results/launch-par8.log 2>&1; then
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  grep -oE 'n_ctx +=[^,]*|n_parallel[^,]*' $D/results/$(ls -t $D/results|grep '^server-'|head -1) 2>/dev/null | head -3
  for C in 1 2 4 8; do $D/benchpar.sh 18131 par$C $C 192; done
else echo "par8 LAUNCH_FAIL"; tail -3 $D/results/launch-par8.log; fi
# 2) remaining single-stream configs, control LAST to detect drift
single st0      "GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=0"
single redpar   "GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS=1024"
single nmax1    "GGML_CPU_DUMMY=0" --spec-draft-n-max 1
single control2 "GGML_CPU_DUMMY=0"
echo "=== DRIVE DONE ==="; column -t $TSV
