# What MiMo-V2.6-Pro can actually reach on this box, and why

Written 2026-09-21 ~22:20 by the Claude session that was working GLM-5.3-Flash, at the maintainer's request to help get MiMo to
18 tok/s. Arithmetic from the model's own tensor table plus this machine's measured memory ceiling. Nothing here changes
your files; `numa-placement.sh` beside it is a read-only diagnostic.

## Where the bytes go — attention, not the experts

Summed over all 13 shards with `gguf-py`, grouped by tensor name, and multiplied by what a decode step actually reads:

| group | on disk | read per token | type |
|---|---:|---:|---|
| routed experts, 8 of 384 used | 494.9 GiB | **10.3 GiB** | MXFP4 |
| **attention** (`attn_qkv` 12.0 + `attn_output` 7.3) | 19.3 GiB | **19.3 GiB** | **Q8_0** |
| router `ffn_gate_inp` | 0.6 GiB | 0.6 GiB | F32 |
| output head, dense FFN (4 layers), nextn | 2.4 GiB | 2.4 GiB | Q8_0 |
| | | **32.6 GiB = 35.0 GB** | |

**Attention is 59% of the per-token traffic.** Only 8 of 384 experts are touched per token, but every layer's attention
weights are read every token, and they are Q8_0 (8.5 bits) while the experts are MXFP4 (4.25). That inverts the intuition
this host has been tuned around — on GLM-5.3-Flash the experts are the wall and the dense matmuls are secondary.

## The ceilings

Against this machine's measured 381.6 GB/s aggregate read (sr950-380gbs-denominator-is-real, `tools/membw.c`):

| configuration | bytes per step | ceiling at 100% | at the 61% duty cycle Flash actually achieves |
|---|---:|---:|---:|
| no speculation | 35.0 GB / token | **10.9 tok/s** | 6.7 |
| MTP depth 2 (3 rows, ~2.5 tokens per pass) | 54.1 GB / pass | **17.6 tok/s** | 10.8 |
| MTP depth 2 + attention requantised to Q6_K | 49.4 GB / pass | **19.3 tok/s** | 11.8 |

Two conclusions worth being blunt about:

1. **Without speculation, 18 tok/s is not reachable at any efficiency.** It would need 630 GB/s, 1.65x this machine.
2. **MTP is the whole ballgame.** It works because attention, the router and the output head are read once per verify pass
   and amortise across the accepted tokens, while only the expert bytes scale with the row count. `SKIP-MTP` in this
   directory says it was deliberately skipped in the 09-21 window; for the 18 target it is the first thing to turn on, not
   the last. The `nextn` weights are already in the GGUF — the loader lists `blk.72.nextn.*` as unused and ignores them —
   so they need extracting into a sidecar the way Flash's were
   (`engineering/2026-09-08/.../extract-glm5next-mtp-gguf.py` in duck-duck-llama is the working pattern, and
   `launch-mimo-tp.sh` already has `SPEC=mtp`).

## Attention requant is one environment variable

`llama.cpp-mimo-tp/ggml/src/ggml-cpu/repack.cpp` already carries the load-time requant hooks, and the rule is
`is_attn = nd == 2 && strstr(cur->name, ".attn_")`, which both `blk.N.attn_qkv.weight` and `blk.N.attn_output.weight` match.
They are Q8_0, so:

    GGML_CPU_ATTN_REQUANT=q6_K    # 19.3 -> 14.9 GiB per pass;  q5_K -> 12.5;  q4_K -> 10.2

No reconversion, no new GGUF. Three caveats:

- **It is not bit-exact.** Use `golden.py` / `golden/` to compare outputs before believing any speed number.
- **Cheaper is not automatically faster.** On GLM-5.3-Flash, Q8_0 -> Q6_K on the dense weights measured *slower*
  (19.8 vs 20.5 tok/s) because the Q6_K unpacking cost more than the bandwidth it saved — but there the dense matmuls were
  already at 71-94 GB/s per socket of a ~95 wall and only ~28 ms of the cycle. Here attention is 59% of the traffic, so the
  saving is far larger relative to the unpack cost. Different regime; measure it, do not assume either way.
- Load-time requant converts 19.3 GiB during startup, so the load gets slower.

## If the measured rate comes in far below 10.9

Then bandwidth is not the limiter and placement is. Run `./numa-placement.sh` once `/health` returns ok: it reports
anonymous pages per NUMA node out of `/proc/PID/numa_maps` (no root needed, unlike PCM). A four-way tensor split should sit
near 25% per node. One node holding most of it means the other sockets are reading across UPI, which caps throughput at
roughly one node's share — about 95 GB/s here. For the record, `--load-mode mmap` is *correct* for the tensor-parallel path;
`none` fails there with `read error: Bad address`. The mmap-defeats-membind trap applies only to single-socket
`numactl --membind` runs.

## Correction and a better lever: the MTP blocks are DENSE, and there are three of them

The first pass of this note assumed one small nextn layer. Reading `blk.70` in full:

    attn_qkv 168.9 MiB | attn_output 102.0 | ffn_gate/up/down 102.0 each | nextn.eh_proj 76.5 | norms ~0
    12 tensors, 0.64 GiB, and NO `_exps` tensors at all

So each MTP block is a **dense** transformer block, not a MoE one, and blocks **70, 71 and 72** all carry `nextn.*`. MiMo
ships a three-deep multi-token head, where GLM-5.3-Flash has one. A draft pass therefore costs 0.64 GiB for the block plus
0.93 for the output head — **1.57 GiB against a ~47 GiB verify pass**, which is why speculation pays so heavily here.

Revised ceilings (same method, real block costs, 381.6 GB/s):

| configuration | GiB per cycle | tokens per cycle | at 100% | at 61% duty |
|---|---:|---:|---:|---:|
| no speculation | 31.5 | 1.0 | 11.3 | 6.9 |
| MTP depth 2 | 50.1 | 2.5 | **17.8** | 10.8 |
| MTP depth 3 | 59.3 | 3.2 | **19.2** | 11.7 |
| depth 2 + attention Q6_K | 45.6 | 2.5 | 19.5 | 11.9 |
| depth 3 + attention Q6_K | 54.9 | 3.2 | **20.7** | 12.6 |

**Try depth 3, not just depth 2.** On GLM-5.3-Flash depth 3 LOST 4% — but that model has a single MTP head, so depth 3 meant
re-using one head speculatively, and its marginal verify row cost ~24 ms. MiMo has three trained heads, one per predicted
position, so depth 3 is what the model was built for and each extra draft costs only 1.57 GiB. Do not carry the Flash result
across; measure depth 2 and 3 separately.

The honest summary: **18 tok/s sits right at the arithmetic ceiling.** It needs MTP working at close to full depth AND a duty
cycle better than the 61% Flash achieves, and attention requant buys the margin. Reaching 12-13 tok/s should be
straightforward once MTP is on; 18 is a stretch that depends on duty cycle, which is where the op-level work goes.

## What to run next (the sidecar now exists)

`draft-mtp` needs the heads as a separate file: the driver requires a non-null draft context, and auto-detects MTP by
finding `blk.<block_count-1>.nextn.eh_proj.weight` in the draft GGUF. `extract-mimo-mtp-gguf.py` beside this note produces
it — adapted from the GLM-5.3-Flash extractor, with the architecture check accepting `mimo2` and all three trained heads
kept instead of one, because `llama_model_n_layer_nextn()` is read off this file and anything above 1 puts the driver in its
`chain_heads` mode (one head per draft step, which is what MiMo was trained for):

    PYTHONPATH=<engine>/gguf-py python3 extract-mimo-mtp-gguf.py \
      /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE-00001-of-00013.gguf \
      /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MTP.gguf --force
    # ~3.8 GiB: three dense MTP blocks + token_embd + output_norm + output. Needs numpy and pyyaml.

`launch-mimo-tp.sh` builds the `SPEC=mtp` branch without a `--spec-draft-model`, so pass it through `EXTRA`:

    SPEC=mtp NMAX=2 EXTRA="--spec-draft-model /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MTP.gguf" ./launch-mimo-tp.sh
    SPEC=mtp NMAX=3 EXTRA="--spec-draft-model /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MTP.gguf" ./launch-mimo-tp.sh

Measure both depths: the ceiling table says depth 3 is worth ~1.4 tok/s over depth 2 here, and unlike Flash there is a
trained head per position so the Flash result (depth 3 lost 4%) does not transfer. Then, only once MTP is measured, try
`GGML_CPU_ATTN_REQUANT=q6_K` and gate the output with `golden.py`.

One practical warning: a tensor-parallel load of this model takes **over half an hour** — 518 GiB read from `/models` plus
node-local copies, with `/health` returning 503 the whole time. Watch `free -g`'s anonymous total climbing toward ~518 GiB
rather than polling the endpoint, or it looks like a hang. `restage_flash_q4.py` is unaffected; that is a separate path.

## MEASURED 2026-09-21 21:51 — tensor parallel, no speculation: 7.94 tok/s

`results/mimo-tp4-nospec.json`, port 18190, `ENGINE=tp TP=1 THREADS=15 CTX=8192`, 256 tokens:
**7.94 tok/s decode, 37.4 prompt tok/s**, against **1.03 tok/s** for the plain engine in `mimo-plain.json`. **7.7x.**
Load took 1740 s (29 min) and `/health` returned 503 throughout.

**Placement is correct and is not the limiter.** `numa-placement.sh`: 129.5 / 129.5 / 129.4 / 129.4 GiB on nodes 0-3 —
25.0% each, 517.8 GiB total, zero spread. The config that session wrote is right; only the plain-engine baseline misled.

**The bandwidth model holds, and the duty cycle is 73%.** 7.94 tok/s x 35.0 GB/token = 278 GB/s of the measured 381.6.
That is better than GLM-5.3-Flash's 61%, which cuts both ways: the placement and kernels are already doing well, so there is
less op-level slack to recover here than on Flash.

Projections from the MEASURED 73% duty cycle (not a guess), 3.2 tokens/cycle assumed at depth 3:

| attention | depth 2 | depth 3 |
|---|---:|---:|
| Q8_0 as shipped | 12.9 | **14.0** |
| Q6_K | 14.2 | 15.1 |
| Q5_K | 15.0 | 15.8 |
| Q4_K | 15.8 | **16.5** |

**17 tok/s is above what this configuration reaches.** Depth 3 on the shipped weights projects ~14; 17 would need depth 3
AND attention at Q4_K (16.5, still short) plus a duty cycle over 80% or acceptance above the 80% assumed. The honest
reachable band is **14-16.5 tok/s**: ~14 with no quality change, ~15 with a gated Q6_K. The softest input is the acceptance
rate — MiMo has three trained heads, one per position, so if acceptance beats 80% the depth-3 figure rises.
