# GLM Q8 batched integer-chain and sum experiment

Status: built and exact-output validated; not deployed or a full-model speed claim. Production remains unchanged on the validated MTP + KV rollback + wider-pool runtime.

The deployed packed x16 Q8 batch kernel shares weights across tokens but uses one serial eight-step integer dot chain per token and separately prepares activation sums/scales. This experiment compares exact SAD activation sums, inline preparation for 16-output calls, and two/four independent integer chains. The integer maximum magnitude is bounded by 32*255*128, so the partial sums and their merge cannot overflow signed 32-bit. Every final corrected integer, FP32 scale product and sequential per-block FMA stays unchanged. Quantization and memory layouts are unchanged.

The first gate compares complete direct-kernel outputs with both the actual deployed library and a scalar canonical calculation over full signed-byte ranges, finite FP16 scale extremes, token counts, output strides and reduction lengths. Timing is separate and includes hot and rotating weights; it is not an end-to-end inference result. A representative multi-socket graph gate is required before a service benchmark.

Completed validation and timing: [RESULT.md](RESULT.md). The final four-socket synthetic test with 32 distinct weight sets measures only a 1.0082x median cycle speedup; this does not establish a model tok/s gain.
