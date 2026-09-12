#!/bin/bash
# A/B the UNARY+SCALE fixes on Qwen3.8-Flash-Next via the q4e runtime. Same library both arms.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
TSV=$D/results/abqwen.tsv; echo -e "arm\ttok_s\tgb_s\tpct380" > $TSV
PIN=~/InfoSystemic/AI-Server/serving/fleet-0903/qwen-flash-20tps.json
BASE_LDP=$(python3 -c "import json;print(json.load(open('$PIN'))['runtime_env']['LD_LIBRARY_PATH'])")
run(){ TAG=$1; VAL=$2
  echo "### $TAG (GGML_CPU_PARALLEL_UNARY=$VAL)"
  export QWEN_EXTRA_LDP=/dev/shm/q4ebuild
  if [ "$VAL" = "off" ]; then unset GGML_CPU_PARALLEL_UNARY; else export GGML_CPU_PARALLEL_UNARY=$VAL; fi
  LDP="/dev/shm/q4ebuild:$BASE_LDP" $D/launch-qwen-patched.sh raw > $D/results/launch-$TAG.log 2>&1 \
    || { echo "$TAG LAUNCH_FAIL"; tail -5 $D/results/launch-$TAG.log; return; }
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench2.sh 18131 $TAG 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$TAG\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's|.*tok/s *\([0-9.]*\) GB.*|\1|p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
run qw-off off
run qw-on  4096
echo "=== numerical check ==="
for i in 1 2 3; do
  a=$(python3 -c "import json;print(json.load(open('$D/results/resp-qw-off-$i.json'))['content'])" 2>/dev/null|md5sum|cut -c1-12)
  b=$(python3 -c "import json;print(json.load(open('$D/results/resp-qw-on-$i.json'))['content'])" 2>/dev/null|md5sum|cut -c1-12)
  [ "$a" = "$b" ] && echo "  prompt $i: IDENTICAL ($a)" || echo "  prompt $i: DIFFERS ($a vs $b)"
done
echo "=== ABQWEN DONE ==="; column -t $TSV
