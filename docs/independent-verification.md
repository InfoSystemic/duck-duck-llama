# Independent verification on a two-socket machine

laos99, an outside tester, built this repository's GLM-5.3-Flash engine on a machine unlike the one it was developed on. laos99 then measured it and reported what broke. This page records what that run established and what it changed here.

## The machine

| | laos99's machine | Development machine |
| --- | --- | --- |
| CPUs | 2 × Xeon Silver 4210 (Cascade Lake, 10 cores each, hyperthreading off) | 4 × Xeon Gold 6242 (Cascade Lake, 16 cores each) |
| AVX-512 FMA units per core | 1 | 2 |
| Memory | 24 × 32 GB DDR4-2400, 12 per socket on 6 channels | 24 channels of DDR4-2400 |
| Measured read bandwidth | 79.1 / 78.4 GB/s per socket (Intel MLC); 155.8 GB/s total | 381.6 GB/s total ([membw.c](../tools/membw.c)) |
| Model | GLM-5.3-Flash, a 179.71 GiB Q4_K_M GGUF, split over two CPU-NUMA devices | GLM-5.3-Flash UD-Q4_K_XL, four devices |

The MLC figures are 69% of the theoretical 115.2 GB/s per socket, which is normal for DDR4. Other MLC measurements:

- cross-node bandwidth: 31.7 GB/s;
- idle latency: 77 ns local, 141 ns remote.

## What it confirmed

- **Portability.** With the documented patch, the published engine builds and serves the model on a different CPU SKU with half the sockets.
- **Speculation.** MTP works there: a draft length of 2 raised decode from 6.32 to 7.27 tok/s (+15%). Acceptance was 0.72 at 2.44 tokens per verify cycle. Here the figures are 0.775 on the greedy fixture and about 0.75 on sampled requests, at 2.45 tokens per cycle, so the drafter behaves the same way on both machines.
- **Decode without speculation.** 6.32 tok/s from the server and 6.74 from `llama-bench` (tg20). The rate was the same at 9 threads and at the default thread count.

## What it found

1. **An abort on the first decoded token.** The x16 mixed-expert path sized its list of active experts at 256. GLM-5.3-Flash has 288 routed experts, so the published engine stopped on `GGML_ASSERT(n_as <= 256)`.

   The development machine ran a private object with a larger bound, so no recorded result was affected. But nothing here had exercised the published path at that expert count. [glm5next-x16-moe-expert-bound.patch](../patches/glm5next-x16-moe-expert-bound.patch) fixes it, and the [Flash guide](models/glm-flash.md#before-the-first-launch) now says so before the first launch.

2. **The documentation assumed four sockets.** That report led to [running it on fewer sockets](models/glm-flash.md#running-it-on-fewer-sockets) and to a [profile parameterised by device count](../profiles/glm-flash-any-sockets.json).

3. **Machine-specific details in the archive.** laos99 noticed paths from the development machine in the archive. That prompted a category-by-category privacy review and a rewrite of this repository's history.

   **Discard clones made before 2026-09-24 and clone again.**

## Reading the numbers

These are estimates from laos99's data, not measurements made on that machine.

- **Share of the memory ceiling.** PCM read 37.78 + 37.57 = 75.3 GB/s at about 6.5 tok/s. That is 11.6 GB per token, and 48% of the MLC ceiling. The development machine runs the same model at about 61% of its wall. On MLC's own loaded-latency curve, 75 GB/s sits around 85–90 ns, so the memory system is only moderately loaded.
- **The likely limit is AVX-512 compute, not memory.**
  - Silver 4210 has one AVX-512 FMA unit per core, and PCM showed it running AVX-512 code at about 1.55 GHz.
  - Decode dequantises and multiplies every byte it reads. So a pure streaming benchmark reaches 79 GB/s per socket, while decode reaches 37.6.
  - A 3-row verify cycle costs 2.11× a single-token step there, against 1.86× here. The extra rows are dot-product work, which is why MTP gains less on that machine.
- **Hyperthreading is not the lever.** MLC saturated the DIMMs with it off. An earlier suggestion from this project that hyperthreading was the main lever was wrong.
- **Headroom.** At the development machine's duty cycle, that machine would reach about 8 tok/s without speculation and 9–10 with MTP. The measured 6.7 and 7.3 are below that ceiling, but not by much.
- **Not yet tried there:**
  - `GOMP_SPINCOUNT=20000`, worth about 3% here; see [the GLM decode report](../benchmarks/glm53-flash-paseo-decode-20260920.md).
  - The [Cascade Lake broadcast finding](../benchmarks/cascade-lake-vnni-broadcast.md), which may not apply with one FMA unit.
  - The [GQA attention fix](../benchmarks/cpu-flash-attn-gqa-splitkv.md), which does not apply: GLM-5.3-Flash uses its own MQA attention kernel.

## Why this matters

Every other result in this repository comes from one machine. A second machine, with a different SKU, half the sockets, one FMA unit per core and a different quantisation, reproduced the engine's behaviour where the documentation said it would. It also exposed the one path the development machine never exercised.

Thanks to laos99 for the careful measurements and for reporting both the bug and the privacy problem.
