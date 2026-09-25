# Qwen-Image-2.1 on CPU: 20-step images on a machine busy serving LLMs

**Set up 2026-09-22.** This is text-to-image generation and image editing with Qwen-Image-2.1 on the same four-socket server, which has no GPU. The model has two parts:

- a 7B single-stream diffusion transformer (DiT);
- a Qwen3-VL-8B text encoder.

The engine is stable-diffusion.cpp at `c92d73c`; Qwen-Image-2.1 support landed there on September 20.

The weights, pinned by revision and SHA-256-checked, are listed in the [README](../engineering/2026-09-23/archive/serving/qwen-image-2.1/README.md).

## Speed

Measured on sockets 0–2 (45 physical cores), counting only time while the job was not paused:

| Stage | 512×512 | 1024×1024 |
| --- | ---: | ---: |
| Text encoder | 2 s | 2 s |
| Per step (conditional + unconditional) | 20 s | 97–118 s |
| VAE decode | 28 s | 118 s |
| 20-step image | ~7.5 min | ~35 min |

The output was checked against stable-diffusion.cpp's own example (same prompt, seed and sampler): the composition matches and the sign in the image is legible. At 1024×1024 with default settings, the model renders requested text correctly.

## What made it faster

**Upcast the DiT from Q8_0 to F16 at load.**

- Cost: 6 GB of RAM. The conversion is exact.
- Why it helps: llamafile's Q8_0 GEMM on Cascade Lake is bound by per-32-block scale arithmetic, while F16 goes through its AVX-512 FMA kernel.
- Result: 28 → 20 s per step at 512×512.

**Turn llamafile on.** stable-diffusion.cpp's bundled ggml builds with `GGML_LLAMAFILE=OFF`, which leaves every matmul on the row-by-row `vec_dot` path.

**Three sockets, not one or four.** ggml's matmuls scale poorly across sockets on this machine. [`mmbench.cpp`](../engineering/2026-09-23/archive/serving/qwen-image-2.1/bench/mmbench.cpp) links the static ggml and times one matmul by shape, type and NUMA layout ([results](../engineering/2026-09-23/archive/serving/qwen-image-2.1/bench/results/mmbench-1100.txt)):

| 4096 × 4096 × 1100 | 1 socket, 15 threads | 3 sockets, 45 threads, memory on one node | 3 sockets, 45 threads, interleaved |
| --- | ---: | ---: | ---: |
| Q8_0 | 0.82 TFLOPS | 0.43 | 0.49 |
| F16 | 0.74 | 0.50 | 0.49 |
| F32 | 0.65 | 0.88 | 0.69 |

At the attention shape, 45 threads are slower than 15. Larger shapes do scale: F16 at 24576 × 4096 × 1100 reaches 1.82 TFLOPS on three interleaved sockets, against 1.19 on one.

For the whole DiT:

| Sockets | Seconds per step |
| --- | ---: |
| 1 | 39 |
| 3 | 20 |
| 4 (60 threads, measured before the F16 upcast) | no faster than 3 |

One cell of the results file, F16 at 24576 on memory bound to one node, reads 9.9 s. That is an outlier, not a trend.

**Keep the VAE in F16.** stable-diffusion.cpp hard-codes its convolution weights to F16. It decodes faster on one socket (16 s) than on three (28 s).

**No flash attention.** ggml's CPU flash attention is not tiled for this case and accumulates V in FP16 ([why that matters](cpu-flash-attn-f16-accumulation.md)).

## Sharing a machine that serves models

The LLM servers own the worker cores, and one busy worker core stalls a tensor-parallel decode by about 30%. So the image job never simply runs on them. [`yield-gate.py`](../engineering/2026-09-23/archive/serving/qwen-image-2.1/yield-gate.py) sends the job SIGSTOP within about 0.2 s when any of these holds:

- a model server is busy;
- another process keeps half a core busy on the job's cores or their hyperthread siblings;
- the siblings carry the load of three or more cores.

It resumes the job 5–10 s (randomised) after all of these clear, so two image jobs queue instead of colliding. The job runs with `oom_score_adj 1000`, so a RAM squeeze kills it and never a model server.

The cost is wall time. During an active tuning session on another model, the first 1024×1024 image took 26 hours of wall time for 35 minutes of compute, pausing 69 times. That is the design working: a decode or a benchmark always wins.

The eight spare CPUs were tried too. They are about 90% busy with CI and container work, and could not finish one 512×512 step in 10 minutes.
