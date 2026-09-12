#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 UD-Q4_K_XL on the four-socket SR950, CPU ONLY.
#
# WHY THIS CONFIG:
#  * 99.8% of this quant is repack-eligible by bytes (MXFP4 97.43% + Q8_0 2.54%).
#    GLM-5.2 UD-Q2_K_XL is only 5.6% eligible (94.4% IQ2_XS/IQ3_XXS).
#  * Measured at the real [4096x2048] expert shape, M=1, 16 threads/socket:
#      MXFP4  0.321 ms default -> 0.133 ms repacked (2.42x), 33.5 GB/s
#      (GLM's IQ2_XS: 0.504 ms, 7.2 GB/s -- 4.65x worse)
#  * NO --device CPU-NUMA*/--split-mode tensor. That path allocates into
#    NUMA-local buffers so buft never equals ggml_backend_cpu_repack_buffer_type()
#    and repack becomes unreachable -- it is why GLM has never repacked once.
#    Also: LLAMA_SPLIT_MODE_TENSOR is not implemented for arch 'deepseek4'.
#  * --load-mode mmap is correct WITH repack: llama-model-loader.cpp:1548-1583
#    maps only is_default_buft tensors; a CPU_REPACK tensor already has cur->data
#    allocated, so it takes the else branch and calls ggml_backend_tensor_set()
#    from the mapped source. Source pages stay reclaimable; only the repacked
#    destination is anonymous. Do NOT use --load-mode none.
set -euo pipefail
BIN=/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/build-sr950-glm-pgo/bin/llama-server
MODEL=/models/gguf/DeepSeek-V4-Flash-0731-UD-Q4_K_XL/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf
PORT="${1:-18092}"
# Warm page cache first: sda has read_ahead_kb=128 and the mmap load pattern
# does not trigger readahead, costing 2.7x on load time. See prefetch-model.sh.
/home/kwebb/InfoSystemic/AI-Server/serving/prefetch-model.sh "$MODEL" || true

export GGML_CPU_MOE_GATE_UP_FUSION=1 GGML_CPU_FFN_GATE_UP_FUSION=1
export GGML_CPU_MOE_WEIGHTED_SUM_FUSION=1
# Page placement: --numa distribute alone left 286 GB split 11/116/25/132 GB
# across the four nodes (first-touch follows the loading thread), so ~75% of
# accesses were remote and two memory controllers idled. Worse, llama.cpp's
# --numa distribute OVERRIDES the mempolicy numactl sets: with both, 55 GB of
# 60 GB landed on node 2 alone. The correct pairing is `numactl --interleave=all`
# for PAGE placement plus `--numa numactl` ("use the CPU map provided by
# numactl") for the CPU map. This is exactly what the long-running Qwen3.8-27B
# on 18081 does, and it yields textbook placement -- 10371/10366/10372/10368 MB
# across the four nodes -- and 112 GB/s achieved of 138.9 GB/s interleaved, i.e.
# 81% efficiency. Do NOT use --numa distribute here.
exec numactl --interleave=all "$BIN" --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
  --alias dsv4-sr950,dsv4-flash,deepseek-v4-flash \
  --load-mode mmap --fit off --ctx-size "${DSV4_CTX:-65536}" \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 --parallel 1 \
  --cache-prompt --cache-reuse 256 --jinja \
  --predict 32768 --temp 1.0 --top-p 0.95 --top-k 20 \
  --timeout 3600 --threads-http 4 --no-webui --metrics --log-timestamps \
  --numa numactl --threads 64 --threads-batch 64 \
  ${DSV4_EXTRA:-}
