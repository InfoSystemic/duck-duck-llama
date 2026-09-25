# Qwen-Image-2.1 on smeagol (CPU, stable-diffusion.cpp)

Text-to-image and image editing with [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)
on this box's CPUs (smeagol has no GPU). Qwen-Image-2.1 is a 7B single-stream DiT with a
Qwen3-VL-8B text encoder. Set up 2026-09-22.

## Use

```bash
cd ~/InfoSystemic/AI-Server/serving/qwen-image-2.1
./qwen-image.sh -p "a red fox reading a newspaper"             # 1024x1024, 20 steps -> out/qwen-image-<time>.png
./qwen-image.sh -p "a red fox reading a newspaper" -W 512 -H 512   # ~5x faster draft
./qwen-image.sh -p "make it night" -r in.png -o out/edit.png   # edit; -r adds the Qwen3-VL vision tower
./qwen-image.sh -p "This is an RGBA image with transparency. A glass teapot. The image has alpha channel and the background is transparent." -o out/teapot.png
```

Any `sd-cli` flag overrides a default (`--cfg-scale 6.0 --sampling-method euler --steps 20 -W 1024 -H 1024 -s -1`,
i.e. a random seed; pass `-s N` to reproduce an image). Sizes must be multiples of 32; the model supports up to 2048x2048 and
2752x1536. `-v` prints per-step timing. The yield gate's pause/resume lines go to stderr.

## How it shares the box

The llama-server fleet owns the 120 worker CPUs, and one busy worker core stalls a
tensor-parallel decode ~30%. So `qwen-image.sh` never just runs on them:

- **`MODE=gated` (default)** runs on the 15 physical worker cores of each socket in `NODES`
  (default `0,1,2`: 45 threads, memory interleaved over those nodes; a single node is
  `--membind`-local). `yield-gate.py` SIGSTOPs the job within ~0.2 s whenever a llama-server or
  any other process with `/models/` in its command line is busy, whenever another process keeps
  half a core on the job's cores or their hyperthread siblings, or when the siblings carry 3+
  cores. It resumes 5-10 s (randomized) after all of that clears, so two image jobs queue rather than collide. A decode or another session's benchmark always
  wins; the image just takes longer.
- **`MODE=spare`** uses only the 8 spare CPUs (15,31,47,63,79,95,111,127). They are shared with the
  CD pipelines, docker and netdata at ~90% busy, so the first 512x512 step did not finish in 10 min.
  Usable only for tiny jobs.
- The job is `oom_score_adj 1000` either way: a RAM squeeze kills it, never a model server. Peak RSS
  is 22-28 GB (weights 18.6 GB after the F16 upcast, plus a 2.3 GB compute buffer at 1024x1024). While
  paused it keeps that RAM.

If a sibling session holds a socket (e.g. a benchmark pinned to 48-62), `NODES` that avoid it
keep this job's threads and memory off it entirely.

## Speed (measured 2026-09-22, sockets 0-2, time while not paused)

| | 512x512 | 1024x1024 |
|---|---:|---:|
| text encoder (any size) | 2 s | 2 s |
| per step (cond + uncond) | 20 s | 97-118 s |
| VAE decode | 28 s | 118 s |
| 20-step image | ~7.5 min | ~35 min |

Wall time is longer by however long the gate yields. During an active MiMo tuning session it
yielded ~50% of a 15-minute 512x512 run; the first 1024x1024 sample (`out/smeagol-coffee-1024.png`)
took 26 h of wall time for 35 min of compute, pausing 69 times through a MiMo redeploy and a day
of tuning runs.

## Why these settings

- **DiT upcast Q8_0 -> F16 at load** (`TYPE_RULES`, exact, costs 6 GB RAM): llamafile's Q8_0 GEMM on
  this Cascade Lake is bound by per-32-block scale math, F16 goes through its AVX-512 FMA kernel.
  28 -> 20 s/step at 512x512.
- **3 sockets, not 1 or 4:** ggml's matmuls scale poorly across sockets here — on the 4096x4096
  attention shape, 45 threads are *slower* than 15 on one socket (likely the per-row activation
  conversion, which every thread touches in ~91-element slices). 1 socket: 39 s/step; 3 sockets:
  20 s/step; the 60-thread Q8_0 run was no faster than 45. GEMM throughput by shape, type and NUMA layout is in
  `bench/results/mmbench-1100.txt` (`bench/bench.sh` re-runs it through the gate).
- **`GGML_LLAMAFILE=ON` at build time** — sd.cpp's ggml defaults it OFF, which leaves every matmul
  on the row-by-row `vec_dot` path.
- **The VAE stays F16** whatever the rules say: sd.cpp hard-codes the Wan-style VAE's conv weights
  to F16 (`src/model/vae/wan_vae.hpp`). It decodes faster on one socket (16 s) than three (28 s).
- **No flash attention**: ggml's CPU flash-attn is not tiled and accumulates V in FP16; the default
  F32 matmul + softmax path is used.

## Validation

`out/validate-cat-512-f16.png` reproduces sd.cpp's own example
(`out/reference-sdcpp-cat-512.png`: "a lovely cat holding a sign says 'qwen2.1.cpp'", 512x512,
20 steps, cfg 6, euler, seed 42) — same composition, legible sign. The reference used an int8 DiT
and a Q4_K_M text encoder, so pixels differ slightly. `out/smeagol-coffee-1024.png` (default
settings, seed 42) renders "SMEAGOL COFFEE" correctly at 1024x1024.

## What is installed

Weights, `/models/gguf/Qwen-Image-2.1` (14 GB; `download.sh` re-fetches, `verify-sha256.py`
re-checks against Hugging Face — run it under `taskset -c 15,31,47,63,79,95,111,127`):

| file | repo @ revision |
|---|---|
| `qwen-image-2.1-Q8_0.gguf` (7.6 GB) | `unsloth/Qwen-Image-2.1-GGUF` @ `2c31ccd392b367a6637841a143813320a02dff55` |
| `vae/qwen_image_2.1_vae_bf16.safetensors` (0.7 GB) | `unsloth/Qwen-Image-2.1-FP8` @ `9e5206424833a00b340bf5396b37486d1306e61f` |
| `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` (5.1 GB) | `unsloth/Qwen3-VL-8B-Instruct-GGUF` @ `b93a7ee713758252c555be4210c00540df954dc2` |
| `mmproj-F16.gguf` (1.2 GB, editing only) | same |

SHA-256 (all four matched Hugging Face's LFS records on 2026-09-22, `sha256-verify.log`):

```
c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5  qwen-image-2.1-Q8_0.gguf
71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c  vae/qwen_image_2.1_vae_bf16.safetensors
e3d1a6e87c5cb31e054f2c3bc0dd82ffde052f613de5eef3665b7bd33c9b703e  Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf
d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4  mmproj-F16.gguf
```

Engine, `~/InfoSystemic/AI-Server/engines/stable-diffusion.cpp` @ `c92d73c` (master; Qwen-Image-2.1
support landed 2026-09-20 in PR #1994), ggml submodule `4bf5f60`. Rebuild on the spare cores:

```bash
cd ~/InfoSystemic/AI-Server/engines/stable-diffusion.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LLAMAFILE=ON -DGGML_OPENMP=ON
taskset -c 15,31,47,63,79,95,111,127 nice -n 10 cmake --build build --target sd-cli -j 8
```

## Files

| path | what |
|---|---|
| `qwen-image.sh` | the launcher (modes, NUMA, OOM guard, defaults) |
| `yield-gate.py` | generic "borrow idle cores" wrapper; `GATE_DEBUG=1` prints per-second stats |
| `download.sh`, `verify-sha256.py` | pinned re-fetch and hash check |
| `bench/` | `mmbench.cpp` GEMM micro-benchmark, `bench.sh`, `results/` |
| `logs/`, `out/` | run logs (with `/usr/bin/time -v`), images |
