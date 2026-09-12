#!/bin/bash
# Qwen3.8-Flash-Next from Kaden's PINNED recipe fleet-0903/qwen-flash-20tps.json.
# Do NOT hand-assemble this env: it needs the x16 kernel knobs, must NOT set
# GGML_CPU_X16_Q8_EXPERTS (fixed int active[256] vs Qwen's 512 experts), and its
# LD_LIBRARY_PATH layers THREE private builds ahead of validated-iq-batch3-bin.
# usage: launch-qwen.sh raw|spec
set -uo pipefail
MODE=${1:-spec}
shift || true   # so extra llama-server args can be appended
for prt in 18131 18132 18133; do for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill "$p" 2>/dev/null; done; done
for i in $(seq 240); do ss -ltn 2>/dev/null | grep -qE ':(18131|18132|18133) ' || break; sleep 1; done
sleep 3
export GGML_CPU_ARGSORT_TOP_K=1
export GGML_CPU_FFN_GATE_UP_FUSION=1
export GGML_CPU_IQ2_XS_REPACK=1
export GGML_CPU_IQ3_XXS_REPACK=1
export GGML_CPU_IQ_R16_BATCH3=1
export GGML_CPU_IQ_R16_NIBBLE2=1
export GGML_CPU_IQ_R16_REPACK=1
export GGML_CPU_MOE_GATE_UP_FUSION=1
export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
export GGML_CPU_NUMA_FUSED_REDUCE=1
export GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS=65536
export GGML_CPU_NUMA_HUGEPAGES=0
export GGML_CPU_NUMA_MERGE_REDUCE=1
export GGML_CPU_NUMA_POLL=100
export GGML_CPU_NUMA_REPACK=1
export GGML_CPU_NUMA_THREADS=15
export GGML_CPU_PARALLEL_COPY=1
export GGML_CPU_PARALLEL_SIGMOID=1
export GGML_CPU_Q5_K_REPACK=1
export GGML_CPU_Q8_0_REPACK=1
export GGML_CPU_Q8_0_REPACK_FORCE=1
export GGML_CPU_REPACK_LOAD_THREADS=16
export GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=4096
export GGML_CPU_X16_ATTN3D=1
export GGML_CPU_X16_Q4_K=1
export GGML_CPU_X16_Q5_BATCH2=1
export GGML_CPU_X16_Q5_BYTES=0
export GGML_CPU_X16_Q5_BYTES_BATCH3=0
export GGML_CPU_X16_Q5_K=1
export GGML_CPU_X16_Q6_K=1
export GGML_CPU_X16_Q8_0=1
export GGML_CPU_X16_Q8_BATCH=1
export GGML_Q4E_EXPERT_EVEN_SPLIT=1
export GGML_Q4E_HC_TP=1
export GGML_Q4E_RS_ROLLBACK=1
export GGML_Q4E_SPLIT=13
export LD_LIBRARY_PATH=${LDP:-}${LDP:+:}/home/kwebb/InfoSystemic/AI-Server/serving/fleet-0903/results/qwen-expert-even-split-policy-0906b/private-split:/home/kwebb/InfoSystemic/AI-Server/serving/fleet-0903/results/qwen-q6-q8-wide-batch-0907/private-cpu:/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-q4e-goal-0904/validated-iq-batch3-bin

MODEL=/home/kwebb/.local/share/ai-models/Qwen3.8-Flash-Next-38bb39ee9782/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf
LOG=~/InfoSystemic/AI-Server/serving/fleet-0911/results/qwen-$(date +%H%M%S).log
ARGS=(/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-q4e-goal-0904/validated-iq-batch3-bin/llama-server --port 18131 --host 127.0.0.1 --load-mode mmap --fit off --ctx-size 4096 --flash-attn on --batch-size 512 --ubatch-size 256 --parallel 1 --gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --split-mode tensor --tensor-split 1,1,1,1 --threads 15 --threads-batch 15 --jinja --reasoning-format deepseek --reasoning-preserve --no-webui --metrics --verbosity 2 --model /home/kwebb/.local/share/ai-models/Qwen3.8-Flash-Next-38bb39ee9782/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf --alias qwen-goal,qwen3.8-flash-next,flash-next,qwen-q6-trial --no-cache-prompt --model "$MODEL" --alias Qwen3.8-Flash-Next)
if [ "$MODE" = spec ]; then ARGS+=(--spec-type draft-mtp --spec-draft-model /models/gguf/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-draft-device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 --spec-draft-ngl all --spec-draft-n-max 4 --spec-draft-p-min 0.3); fi
T0=$(date +%s)
nohup "${ARGS[@]}" "$@" > "$LOG" 2>&1 &
SRV=$!
echo "pid=$SRV mode=$MODE log=$LOG"
for i in $(seq 1200); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null)
  [[ "$s" == *'"ok"'* ]] && { echo "UP in $(( $(date +%s)-T0 ))s"; exit 0; }
  kill -0 "$SRV" 2>/dev/null || { echo DIED; grep -iE "error|assert|abort|unknown|fail" "$LOG" | tail -5; exit 1; }
  sleep 2
done
echo TIMEOUT; exit 1
