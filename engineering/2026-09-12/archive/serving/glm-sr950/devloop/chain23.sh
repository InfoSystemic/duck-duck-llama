#!/usr/bin/env bash
set -u
cd /dev/shm/glm-dev
DEV2=~/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/build-dev2/bin/llama-server
ENVF=/home/user/InfoSystemic/AI-Server/serving/glm-sr950/model.glm53-q4-fast-v4.env
pin_and_probe() { ( until grep -q "\[$1\] up" chain23.log 2>/dev/null; do sleep 5; done; sleep 3; ./pin_main.sh; sleep 60; ./schedstat_by_role.sh 120 > results/schedstat-$1-roles.txt 2>&1 ) & }
echo "[chain23] dense oracle (post-fix) $(date +%T)"
export DEV_MODEL=/dev/shm/glm-dev/glm53-q4kxl-4L.gguf DEV_REPS=1 DEV_TOKENS=32
B="GGML_GLM_SHEXP_TP=1 GGML_CPU_NUMA_FUSED_REDUCE=1 GGML_GLM_QA_TP=2 GGML_CPU_X16_Q4_K=1 GGML_CPU_X16_Q5_K=1 GGML_CPU_X16_Q6_K=1 GGML_CPU_ROUTER_F16=1 GGML_CPU_NUMA_MERGE_REDUCE=1 GGML_CPU_X16_ATTN3D=1"
env $B ./devrun.sh o23-base "$DEV2"
env $B GGML_GLM_DSA_DENSE=1 ./devrun.sh o23-dense "$DEV2"
python3 - <<'PY'
import json
def top(f):
    d=json.load(open(f)); return {t['token']: t['logprob'] for t in d['completion_probabilities'][0]['top_logprobs']}
try:
    b=top('/dev/shm/glm-dev/results/probe-o23-base.json'); x=top('/dev/shm/glm-dev/results/probe-o23-dense.json'); c=[k for k in b if k in x]
    print("o23-dense vs base: common", len(c), "maxdiff", round(max(abs(b[k]-x[k]) for k in c),4))
except Exception as e: print("o23-dense vs base: ERROR", e)
PY
unset DEV_MODEL DEV_REPS DEV_TOKENS
R="GGML_CPU_ATTN_REQUANT=q5_K GGML_CPU_SHEXP_REQUANT=q5_K GGML_CPU_OUTPUT_REQUANT=q6_K"
env $B $R ./devrun.sh s23-base "$DEV2"
env $B $R GGML_CPU_X16_DUAL=1 ./devrun.sh s23-dual "$DEV2"
env $B $R GGML_GLM_DSA_DENSE=1 ./devrun.sh s23-dense "$DEV2"
env $B $R GGML_CPU_X16_CHUNK_MIN=64 GGML_CPU_X16_CHUNK_MAX=4096 ./devrun.sh s23-oldchunk "$DEV2"
env $B $R GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US=1000000000 ./devrun.sh s23-noyield "$DEV2"
DENSE=$(grep -q "o23-dense vs base: common 10 maxdiff 0.0" chain23.log && echo 1 || echo 0)
echo "[chain23] launching v6 full (merge attn3d dual=1 dense=$DENSE) $(date +%T)"
pin_and_probe v6-fast
GGML_CPU_NUMA_MERGE_REDUCE=1 GGML_CPU_X16_ATTN3D=1 GGML_CPU_X16_DUAL=1 GGML_GLM_DSA_DENSE=$DENSE FULL_ENVF=$ENVF ./run_full2.sh v6-fast "$DEV2"
echo "[chain23] MTP sweep on v6 $(date +%T)"
./mtp_sweep.sh v6 18091
echo "[chain23] agentic replay on v6 $(date +%T)"
cd ~/InfoSystemic/AI-Server/serving/glm-sr950
for arm in "18 0.75" "12 0.6"; do set -- $arm; GLM_BENCH_WINDOW_GUARD=/bin/true GLM_BENCH_SPEC_N_MAX=$1 GLM_BENCH_SPEC_P_MIN=$2 GLM_REPLAY_OUT=/dev/shm/glm-dev/results/replay-v6-n$1p$2.jsonl ./benchmark-replay.sh v6-n$1p$2 2>&1 | grep -E "max_decode|correct_workloads|draft_acceptance"; done
cd /dev/shm/glm-dev
echo "[chain23] done $(date +%T)"
