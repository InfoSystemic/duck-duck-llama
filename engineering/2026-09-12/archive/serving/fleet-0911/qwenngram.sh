#!/bin/bash
# ngram-mod composite on Qwen. TRAP (09-10): ngram-mod's cache persists ACROSS requests and
# --no-cache-prompt does not clear it, so each arm gets a FRESH server and each prompt is
# distinct -- never re-measure a prompt this instance has already answered.
D=~/InfoSystemic/AI-Server/serving/fleet-0911
trap '$D/finish.sh' EXIT
PIN=~/InfoSystemic/AI-Server/serving/fleet-0903/qwen-flash-20tps.json
BASE_LDP=$(python3 -c "import json;print(json.load(open('$PIN'))['runtime_env']['LD_LIBRARY_PATH'])")
export GGML_CPU_PARALLEL_UNARY=4096
LDP="/dev/shm/q4ebuild:$BASE_LDP" $D/launch-qwen-patched.sh spec \
    --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.3 \
  > $D/results/launch-qngram.log 2>&1 || { echo LAUNCH_FAIL; tail -8 $D/results/launch-qngram.log; exit 1; }
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1; sleep 3
echo "### ngram-mod,draft-mtp  (reference: MTP4+fix alone = 20.32 mean / 22.1 code / 22.9 analysis)"
$D/bench2.sh 18131 qngram 192
echo "=== QNGRAM DONE ==="
