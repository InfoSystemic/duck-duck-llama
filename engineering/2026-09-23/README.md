# Engineering snapshot, 2026-09-23

This snapshot covers September 20–23:

- MiMo-V2.6-Pro-RL, a 1.02-trillion-parameter model, brought up and tuned on the four-socket CPU server.
- The last GLM-5.3-Flash production revisions (g–l) and the depth campaign behind them.
- Cascade Lake micro-benchmarks.
- Qwen-Image-2.1 image generation on CPU.

It is a curated subset: 216 files plus the two engine layers that rebuild the MiMo runtime. Start with the curated pages:

- the [MiMo model guide](../../docs/models/mimo-v26-pro.md)
- the [MiMo benchmark](../../benchmarks/mimo-v26-pro-cpu-20260923.md)
- the [block-drafter case study](../../docs/case-studies/block-drafter.md)
- the updated [GLM-5.3-Flash guide](../../docs/models/glm-flash.md)

## MiMo-V2.6-Pro-RL

| Area | Entry point |
| --- | --- |
| What runs, what is on disk, how the 518 GiB GGUF was converted without the 534 GiB checkpoint ever existing on disk | [README.md](archive/serving/mimo-v26-pro/README.md) |
| Pass/fail goal lines for the tuning session, each with its evidence | [GOAL-20260922.md](archive/serving/mimo-v26-pro/GOAL-20260922.md) |
| Bandwidth arithmetic: attention is 59% of the bytes per token and the experts only 31%, so a useful rate requires speculation | [BANDWIDTH-CEILING-20260921.md](archive/serving/mimo-v26-pro/BANDWIDTH-CEILING-20260921.md) |
| Speculation: the MTP dead end, the six defects between the fork and a working DFlash drafter, `p_min`, and the cycle model | [STATE-SPECULATION-20260922.md](archive/serving/mimo-v26-pro/STATE-SPECULATION-20260922.md) |
| CPU kernels: grouped-query split-KV flash attention, multi-column x16 GEMM, paired-group VNNI, exact MoE weighted sum, and builds 0922b–0922j | [STATE-FA-GQA-20260922.md](archive/serving/mimo-v26-pro/STATE-FA-GQA-20260922.md) |
| Vision (F16 overflow in the ViT, fixed by an F32 projector) and XML tool-call parsing | [STATE-VISION-20260922.md](archive/serving/mimo-v26-pro/STATE-VISION-20260922.md) |
| Feature patches, one per change, including the refuted MTP post-norm hypothesis | [patches/](archive/serving/mimo-v26-pro/patches/) |
| Conversion: MXFP4 expert passthrough, skeleton with sparse holes, resumable per-shard fill, pinned manifest | [converter patch](archive/serving/mimo-v26-pro/mimo-v26-mxfp4-converter.patch), [run-skeleton.sh](archive/serving/mimo-v26-pro/run-skeleton.sh), [fill_experts.py](archive/serving/mimo-v26-pro/fill_experts.py), [hf-manifest-54b10491.json](archive/serving/mimo-v26-pro/hf-manifest-54b10491.json) |
| The mask-token embedding the drafter needs, written in place (6,528 bytes, restorable) | [patch-mask-embedding.py](archive/serving/mimo-v26-pro/patch-mask-embedding.py) |
| Serving: production launcher, per-NUMA-node RAM guard, load watchdog, placement check | [launch-mimo-production.sh](archive/serving/mimo-v26-pro/launch-mimo-production.sh), [wait-for-ram.sh](archive/serving/mimo-v26-pro/wait-for-ram.sh), [load-watchdog.sh](archive/serving/mimo-v26-pro/load-watchdog.sh), [numa-placement.sh](archive/serving/mimo-v26-pro/numa-placement.sh) |
| Probes: acceptance by workload, per-draft-length detail, depth curve, cycle-model fit, golden parity | [draft-probe.py](archive/serving/mimo-v26-pro/draft-probe.py), [spec-detail.py](archive/serving/mimo-v26-pro/spec-detail.py), [depth-probe.py](archive/serving/mimo-v26-pro/depth-probe.py), [fit-cycle.py](archive/serving/mimo-v26-pro/fit-cycle.py), [golden.py](archive/serving/mimo-v26-pro/golden.py) |
| Functional checks: tool calls, audio, real-image vision, the ViT against Xiaomi's reference module | [toolcall-probe.py](archive/serving/mimo-v26-pro/tools/toolcall-probe.py), [audio-probe.py](archive/serving/mimo-v26-pro/tools/audio-probe.py), [vision-real-probe.py](archive/serving/mimo-v26-pro/tools/vision-real-probe.py), [vision-embd.cpp](archive/serving/mimo-v26-pro/tools/vision-embd.cpp), [vision-ref.py](archive/serving/mimo-v26-pro/tools/vision-ref.py) |
| Kernel benchmarks: flash attention at production shapes against float64, x16 GEMM against the GEMV, Cascade Lake issue-rate kit | [fa-bench.cpp](archive/serving/mimo-v26-pro/tools/fa-bench.cpp), [x16-gemm-bench.cpp](archive/serving/mimo-v26-pro/tools/x16-gemm-bench.cpp), [ubench/](archive/serving/mimo-v26-pro/tools/ubench/README.md) |
| Every production deploy, 0922b–0922j: golden comparison, smoke, vision, audio, tool calls, draft probe, depth | [results/](archive/serving/mimo-v26-pro/results/) |
| Draft-length and `p_min` sweeps | [dflash-sweep.md](archive/serving/mimo-v26-pro/results/dflash-sweep.md), [dflash-pmin-sweep.md](archive/serving/mimo-v26-pro/results/dflash-pmin-sweep.md) |

## GLM-5.3-Flash, September 20 evening

| Area | Entry point |
| --- | --- |
| The depth campaign: where the 8K→128K slope goes, the plan and its progress log, and why 24+ tok/s at 4K is a model limit | [DEPTH-CAMPAIGN-PLAN-20260920.md](archive/serving/fleet-0920-flash18/DEPTH-CAMPAIGN-PLAN-20260920.md) |
| Revision g: pooled-indexer fusion that survives an output placed over its inputs | [pool-kernel.inc](archive/serving/fleet-0920-flash18/cpu/pool-kernel.inc), [pool-scalar.inc](archive/serving/fleet-0920-flash18/cpu/pool-scalar.inc), [cfuse/pool-fusion.inc](archive/serving/fleet-0920-flash18/cpu/cfuse/pool-fusion.inc), [build-all-cfuse.sh](archive/serving/fleet-0920-flash18/cpu/build-all-cfuse.sh), [patch](../../patches/glm5next-pool-fusion-overlap-proof.patch) |
| Revision h: blocked lightning-indexer score kernel and radix top-k, both bit-identical | [li-fast.inc](archive/serving/fleet-0920-flash18/cpu/li-fast.inc), [topk-fast.inc](archive/serving/fleet-0920-flash18/cpu/topk-fast.inc), [tests](archive/serving/fleet-0920-flash18/cpu/test/li_fast_check.cpp), [patch](../../patches/glm5next-indexer-score-blocked.patch) |
| Production drop-ins g–l with library hashes and self-rolling-back deploy scripts; the headers of i–l record why each was rejected | [g](archive/serving/fleet-0920-flash18/deploy-0920g/95-f18-0920.conf.proposed), [h](archive/serving/fleet-0920-flash18/deploy-0920h/95-f18-0920.conf.proposed), [i](archive/serving/fleet-0920-flash18/deploy-0920i/95-f18-0920.conf.proposed), [j](archive/serving/fleet-0920-flash18/deploy-0920j/95-f18-0920.conf.proposed), [k](archive/serving/fleet-0920-flash18/deploy-0920k/95-f18-0920.conf.proposed), [l](archive/serving/fleet-0920-flash18/deploy-0920l/95-f18-0920.conf.proposed) |
| Revision g's depth curves, to 41K and to 128K | [to 41K](archive/serving/fleet-0920-flash18/results/depth-curve-prod-revg-40k.json), [to 128K](archive/serving/fleet-0920-flash18/results/depth-curve-prod-revg.json) |
| Parity, rate and perplexity helpers | [cfuse_parity.sh](archive/serving/fleet-0920-flash18/run/cfuse_parity.sh), [quick_rate.py](archive/serving/fleet-0920-flash18/run/quick_rate.py), [ppl_glm.py](archive/serving/fleet-0920-flash18/run/ppl_glm.py), [qsplit_ab.sh](archive/serving/fleet-0920-flash18/run/qsplit_ab.sh) |

## Qwen-Image-2.1 on CPU

- [README](archive/serving/qwen-image-2.1/README.md): settings, speed, and validation.
- [yield-gate.py](archive/serving/qwen-image-2.1/yield-gate.py): lends idle cores to a batch job and gives them back within 0.2 s.
- [GEMM benchmark](archive/serving/qwen-image-2.1/bench/mmbench.cpp) and its [results](archive/serving/qwen-image-2.1/bench/results/mmbench-1100.txt).

These are summarised in the [Qwen-Image benchmark](../../benchmarks/qwen-image-21-cpu.md).

## Rebuilding the MiMo engine

[source-bundles.json](source-bundles.json) lists two engine layers. Build the runtime as follows:

1. Start from the public base `2e0e57f1008053bae4902a772da85e3eb99d4aff`.
2. Apply the September 12 [glm5n-goal-0904 patch](../2026-09-12/patches/llama.cpp-glm5n-goal-0904.patch).
3. Apply [layer 2](patches/llama.cpp-mimo-tp-layer2.patch), which contains:
   - the September 12 NUMA-tuned DeepSeek fork as it stood on September 21: JigSawPT's DeepSeek-V4.1 port, merged by hand with this project's CPU-NUMA work;
   - one local commit adding MiMo-V2 tensor parallelism: a three-segment fused QKV, and head granularity sized to V for `attn_output` and the V cache (K 192 / V 128).
4. Apply [layer 3](patches/llama.cpp-mimo-tp-layer3-build-prod-0922j.patch), the uncommitted production source state `build-prod-0922j`.

The resulting source was compared file by file with the engine tree that built the production binary: all 2,741 engine files are identical.

```bash
git clone https://github.com/unslothai/llama.cpp.git llama.cpp-mimo && cd llama.cpp-mimo
git checkout 2e0e57f1008053bae4902a772da85e3eb99d4aff
R=../duck-duck-llama/engineering
patch -p1 < $R/2026-09-12/patches/llama.cpp-glm5n-goal-0904.patch
patch -p1 < $R/2026-09-23/patches/llama.cpp-mimo-tp-layer2.patch
patch -p1 < $R/2026-09-23/patches/llama.cpp-mimo-tp-layer3-build-prod-0922j.patch
```

Build flags, environment and server arguments are set in [launch-mimo-production.sh](archive/serving/mimo-v26-pro/launch-mimo-production.sh). Each value is explained in its header. A runtime configuration is the binary plus that environment plus those flags. Changing any one of them changes the result, even when nothing reports an error.

Some of the model support in these layers comes from public sources and is not claimed as locally authored: JigSawPT's DeepSeek-V4.1 port, the `mimo2` architecture, and the DFlash drafter runtime and converter. The DFlash drafter model is Xiaomi's.

The six defects described in the speculation notes lay between the drafter and the target:

- in the conversion;
- in the embedding table the drafter borrows from the target;
- in the target's graph.

None was in the drafter's weights. The local changes are the feature patches in [patches/](archive/serving/mimo-v26-pro/patches/). See the [patch attribution](../../patches/ATTRIBUTION.md).

## Contract

The [archive contract](../README.md#archive-contract) applies. This snapshot builds redaction into assembly, instead of relying on a later history rewrite:

- **Text:** every text file passes through an ordered list of transforms that remove operator names and home paths, another project's name, and links to private notes. A file whose bytes change is marked `filtered`, and its manifest entry records both hashes.
- **Evidence JSON:** passes through the repository's own `clean()` filter, then a guard that drops prompts, completions and response text.
- **Refused outright:** compiled artifacts and images.
- **Excluded and listed:** raw op traces, server logs, per-request captures, and intermediate source-state diffs. The manifest lists 47 exclusions, each with its reason.
- **Engine layers:** passed through the same transforms and required to come out unchanged. Neither needed redaction.

Two sections of the archived speculation notes carry a dated note added at publication:

- The depth-loss section points to the attention-kernel fix, which was found the same evening.
- The vision section, which blamed the windowed-attention mask, points to the real cause: F16 overflow, fixed by the F32 projector.

The text of both sections is otherwise as written at the time.

```bash
python3 tools/verify_engineering_snapshot.py --snapshot engineering/2026-09-23
```

AI coding sessions did this work under the maintainer's direction. The notes are their working records, written to be picked up by the next session. That is why they address "you" and carry operating instructions.
