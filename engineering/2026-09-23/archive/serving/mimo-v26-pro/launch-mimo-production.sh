#!/usr/bin/env bash
# The configuration MiMo-V2.6-Pro should actually be served with on this box, and why each part of it.
#
#   BUILD=build-prod-0922j                      pinned binary AND libraries (0922j = 0922i + vectorised row max in the
#                                               flash-attention softmax: attention 1.25-1.4x, bit-identical;
#                                               0922i = 0922g + prompt-batch VNNI kernels
#                                               that run 16-row groups in pairs, one register broadcast per two dot
#                                               products: dense 1.26x, MoE gate/up 1.21x single-core, bit-identical;
#                                               + the bit-identical MoE weighted-sum fusion, enabled below;
#                                               0922g = 0922f + even-split kernel calls,
#                                               SCALE threading, larger MoE-down tiles for prompt batches;
#                                               0922f = 0922e + flash-attention K tiles
#                                               transposed 16x16 in registers + L1-blocked 12x32 tile GEMMs: attention
#                                               1.2-1.5x faster, output bit-identical to 0922e) (pinned-serving-config rule): the DFlash
#                                               fixes of 09-22 + the grouped-query flash-attention kernel + the x16
#                                               multi-column GEMM with activation block sums computed once per matmul
#                                               and vectorised MXFP4 scales + the mimovl image-limit fix + the XML
#                                               tool-call parser fix (common/chat.cpp: MiMo emits the compact
#                                               <tool_call><function=f><parameter=a>v</parameter>... form its own
#                                               template renders; the Qwen3-Coder handler demanded newlines, dropped
#                                               compact calls and forced the model off-format -- 3/20 malformed,
#                                               some running away to max_tokens). Compute output == 0922b bit for bit.
#                                               Source state: patches/SOURCE-STATE-build-prod-0922e.diff.
#   GGML_CPU_X16_CHUNK_MAX_BATCH=64             prompt batches use 64-row weight chunks (decode keeps the 16 set in
#                                               launch-mimo-tp.sh): each chunk re-reads every activation row, so 16-row
#                                               chunks streamed a 512-token batch's activations 424x per matmul. Proxy
#                                               prefill +8..14% (128 was worse: load imbalance). Bit-identical.
#   GGML_CPU_X16_MOE_TILE_BATCH=512             prompt batches: the MoE down projection (6144 rows x 512 inputs per
#                                               node) in 512-row tiles instead of 64 -- 37K tiny work items per layer
#                                               became 4.6K; x16-gemm-bench 1.21x. Decode keeps 64. Bit-identical.
#   GGML_CPU_MOE_WEIGHTED_SUM_FUSION=2          (0922i) the MoE output -- MUL by the router weights over [6144, 8, tokens]
#                                               then 7 ADD nodes -- as ONE pass that reads the expert outputs once
#                                               (~460 -> ~113 MB per layer per node for a 512-token batch, 8 barriers ->
#                                               1). Mode 2 rounds every product before adding, in slot order, exactly
#                                               like the unfused graph; mode 1 (vec_mad FMA) would NOT be bit-identical.
#                                               Proxy profile: MUL + ADD 7.5% -> 2.5% of prefill. Golden identical.
#   UBATCH=1024                                 (0922i) prompt micro-batches of 1024 tokens instead of 512: each MoE
#                                               expert's weights (1.9 GB per layer per node in all) are streamed once per
#                                               micro-batch, so twice the tokens share each read. Bit-identical at 512,
#                                               1024 and 2048 (proxy golden + 8K-token probe); compute buffers ~2x (small).
#   GGML_CPU_PARALLEL_UNARY=4096                unary ops AND (0922g) SCALE of >= 4096 elements run on all threads; SCALE
#                                               was forced single-threaded (attn_out_scaled ~2 ms per layer per prompt
#                                               ubatch). Same knob GLM-Flash runs (+2.1%). Bit-identical.
#   (0922g) kernel calls split tokens EVENLY    fewest calls of <= 10 (Q8_0) / 11 (MXFP4) columns instead of 8 + a thin
#                                               tail: an expert's ~11 prompt tokens are one call, not 8 + 3 (bench:
#                                               MXFP4 +16%, dense +4%). GGML_CPU_X16_GEMM_WMAX overrides. Bit-identical.
#   GGML_CPU_X16_GEMM=1                         prompt batches AND speculative-verify rows go through a multi-column
#                                               VNNI kernel (repack.cpp / arch/x86/repack.cpp) that reuses each loaded
#                                               weight vector across up to 8 tokens, where the GEMV reloaded it per
#                                               token. BIT-IDENTICAL output (same float ops per row and column).
#                                               Proxy: prefill +21..39%, decode +15%.
#   GGML_CPU_FA_GQA=2                           ops.cpp's grouped-query split-KV flash attention for decode, verify AND
#                                               prompt batches. The old kernel streamed the whole KV cache once per
#                                               (row, Q head) -- 128x per layer for an 8-row verify under 4-way TP -- and
#                                               summed V in FP16. Per layer-node: 64K x 8 rows 160 -> 14 ms, 16K 31 -> 4,
#                                               prompt batches 1.6-2.0x; F32 throughout (rel.err 1e-7 vs 1e-2..4e-2).
#                                               Not byte-identical to the old build: the old kernel's rounding is gone.
#   exact attention weights (no GGML_CPU_ATTN_REQUANT)
#                                               Q4_K attention is ~14% faster but FAILED the golden gate (divergence at
#                                               token 11 of 48).
#   SPEC=dflash NMAX=7 PMIN=0.5                 MiMo's own drafter. p_min truncates the 8-token block at the first
#                                               low-confidence position, which lifts the worst case (open prose) 41%
#                                               while leaving code flat. A fixed draft length is the wrong control.
#   CTX=262144                                  63 of 73 layers are sliding-window-128, so only 10 carry full KV:
#                                               256K costs ~12.5 GiB. The old 8192 was throwing away the model's range.
#   MMPROJ=1 (F32 projector, launch-mimo-tp.sh) the F16 projector overflowed into NaN inside the ViT and answered
#                                               '????' for most real images; F32 matches Xiaomi's reference.
#   IMAGE_MAX_TOKENS=1280                       an image larger than ~1.3 MP is downscaled to fit. Uncapped, the
#                                               projector allows 12.8 MP = 12.5K tokens per image, i.e. minutes of ViT
#                                               encode + prefill on this CPU for one screenshot.
#   LLAMA_ARG_CACHE_RAM=32768                   host-RAM prompt cache (default 8 GiB). Measured: a session displaced by
#                                               another resumed with 4837 of 4863 tokens restored, 1.9 s instead of 80 s.
#                                               ~1.2 GiB per 20K-token session; 32 GiB keeps ~25 agents' contexts warm.
#   GGML_CPU_OP_PROFILE (dormant)               `touch /tmp/mimo-prof-arm` records the next 16 graphs' per-op timings
#                                               into the server log (journalctl --user -u mimo-v26-pro); rm to re-arm.
set -euo pipefail
cd "$(dirname "$0")"
echo "${NMAX:-7}"   > /tmp/dflash-draft-n
echo "${PMIN:-0.5}" > /tmp/dflash-draft-pmin
rm -f /tmp/mimo-prof-arm
# (09-23) a reload that can no longer fit on some NUMA node is killed early instead of being OOM-killed at its end,
# 30 minutes of SSD reads later; see load-watchdog.sh. $$ becomes the server: every step below is an exec.
./load-watchdog.sh $$ &
exec env BUILD="${BUILD:-build-prod-0922j}" SPEC=dflash NMAX="${NMAX:-7}" PMIN="${PMIN:-0.5}" \
     CTX="${CTX:-262144}" MMPROJ="${MMPROJ:-1}" IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1280}" \
     GGML_CPU_FA_GQA="${GGML_CPU_FA_GQA:-2}" GGML_CPU_X16_GEMM="${GGML_CPU_X16_GEMM:-1}" \
     GGML_CPU_X16_CHUNK_MAX_BATCH="${GGML_CPU_X16_CHUNK_MAX_BATCH:-64}" \
     GGML_CPU_X16_MOE_TILE_BATCH="${GGML_CPU_X16_MOE_TILE_BATCH:-512}" \
     GGML_CPU_PARALLEL_UNARY="${GGML_CPU_PARALLEL_UNARY:-4096}" \
     GGML_CPU_MOE_WEIGHTED_SUM_FUSION="${GGML_CPU_MOE_WEIGHTED_SUM_FUSION:-2}" UBATCH="${UBATCH:-1024}" \
     LLAMA_ARG_CACHE_RAM="${LLAMA_ARG_CACHE_RAM:-32768}" \
     GGML_CPU_OP_PROFILE='*' GGML_CPU_OP_PROFILE_ARM_FILE=/tmp/mimo-prof-arm GGML_CPU_OP_PROFILE_COUNT=16 \
     LLAMA_SPEC_DRAFT_N_FILE=/tmp/dflash-draft-n \
     LLAMA_SPEC_DRAFT_PMIN_FILE=/tmp/dflash-draft-pmin \
     ./launch-mimo-tp.sh "$@"
