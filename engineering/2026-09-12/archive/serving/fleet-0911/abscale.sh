#!/bin/bash
# A/B the GGML_CPU_PARALLEL_UNARY change, same library both arms, back to back.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
TSV=$D/results/abscale.tsv; echo -e "arm\ttok_s\tgb_s\tpct380" > $TSV
run(){ TAG=$1; VAL=$2
  echo "### $TAG (GGML_CPU_PARALLEL_UNARY=$VAL)"
  for p in $(ss -ltnpH "sport = :18131" 2>/dev/null|grep -oE 'pid=[0-9]+'|cut -d= -f2|sort -u); do kill $p 2>/dev/null; done
  for i in $(seq 240); do ss -ltn 2>/dev/null|grep -q ':18131 ' || break; sleep 1; done; sleep 3
  set -a; while read -r l; do case "$l" in LD_LIBRARY_PATH=*) continue;; esac; [ -n "$l" ] && export "$l"; done < $D/restore-env.txt; set +a
  export LD_LIBRARY_PATH=/dev/shm/profbuild:$(grep '^LD_LIBRARY_PATH=' $D/restore-env.txt|cut -d= -f2-)
  if [ "$VAL" = "off" ]; then unset GGML_CPU_PARALLEL_UNARY; else export GGML_CPU_PARALLEL_UNARY=$VAL; fi
  mapfile -t ARGV < <(grep -v '^$' $D/restore-argv.txt)
  LOG=$D/results/ab-$TAG.log; nohup "${ARGV[@]}" > "$LOG" 2>&1 & SRV=$!
  for i in $(seq 900); do s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null); [[ "$s" == *ok* ]] && break
    kill -0 $SRV 2>/dev/null || { echo "$TAG DIED"; grep -iE 'error|assert' "$LOG"|tail -3; return; }; sleep 2; done
  /dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
  O=$($D/bench2.sh 18131 $TAG 192 2>&1); echo "$O"
  S=$(echo "$O"|grep '^  == ')
  echo -e "$TAG\t$(echo "$S"|sed -n 's/.*mean \([0-9.]*\) tok.*/\1/p')\t$(echo "$S"|sed -n 's|.*tok/s *\([0-9.]*\) GB.*|\1|p')\t$(echo "$S"|sed -n 's/.*(\([0-9.]*\)% of.*/\1/p')" >> $TSV
}
run sc-off off
run sc-on  4096
echo "=== numerical check: greedy output identical? ==="
for i in 1 2 3; do
  a=$(python3 -c "import json;print(json.load(open('$D/results/resp-sc-off-$i.json'))['content'])" 2>/dev/null | md5sum | cut -c1-12)
  b=$(python3 -c "import json;print(json.load(open('$D/results/resp-sc-on-$i.json'))['content'])" 2>/dev/null | md5sum | cut -c1-12)
  [ "$a" = "$b" ] && echo "  prompt $i: IDENTICAL ($a)" || echo "  prompt $i: DIFFERS ($a vs $b)"
done
echo "=== AB DONE ==="; column -t $TSV
