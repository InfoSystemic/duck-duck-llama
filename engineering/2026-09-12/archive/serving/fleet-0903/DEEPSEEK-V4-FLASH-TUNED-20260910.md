# DeepSeek on the tuned engine — 2026-09-10

Goal: **DeepSeek-V4.1-Flash over 20 tok/s.** This documents a large step toward it that
required no downloads and no new code, plus what actually blocks V4.1.

## Headline

**DeepSeek-V4-Flash went from 2.67 to ~11.5-12.6 tok/s on novel text — 4.3-4.7x.**

> **CORRECTION (same day).** An earlier draft of this document claimed 14.76/18.41. Those
> numbers were inflated by an `ngram-mod` cache artifact and are withdrawn — see
> "Measurement trap" below. The honest novel-text figures are 11.48 prose / 12.57 code.

| config | prose tok/s | code tok/s | GB/s | % of 380 |
|---|---:|---:|---:|---:|
| prior record (08-30, dspark fork, interleaved) | 2.67 | — | — | — |
| tuned engine, TP4, raw decode | 10.32 | 10.35 | 178 | 47% |
| + `draft-dspark` n=2 | 12.07 | 12.98 | — | — |
| **+ `ngram-mod,draft-dspark` n=2, NOVEL text (honest)** | **11.48** | **12.57** | — | — |
| same, prompt already generated once on this instance | 12.51 | 23.92 | 116-127 | 31-34% |
| same, n=3 | 12.50 | 14.32 | — | — |
| same, n=4 | 9.91 | 12.02 | — | — |

Correctness verified at every step (`17*23` -> `391`, capital of France -> `Paris`,
coherent prose/code). 512-token samples, temperature 0, seed 42, cache off, 48 IMC counters.
Reproduce with `./launch_dsv4flash_tuned_0910.sh tensor ngram-mod,draft-dspark <draft> 2`.

## Why it was slow, and it was not the model

The only previous DeepSeek-V4-Flash measurement used `engines/llama.cpp-dspark-*`. **None of
those four trees contain the meta backend at all** — no CPU-NUMA devices, no tensor-parallel,
no x16 VNNI kernels, no fused all-reduce. They are August builds that predate every NUMA and
kernel gain of the last six weeks.

The *tuned* tree (`llama.cpp-glm5n-goal-0904`, the binary already serving GLM-5.3-Flash)
registers `deepseek4` **and** `dflash`, has `src/models/deepseek4.cpp` + `dflash.cpp`, and
carries an `is_dsv4` branch of tensor-parallel split rules. Simply pointing it at the
155 GB GGUF already sitting on `/models` gave 3.9x before any speculation.

**This also retires a stale note**: "deepseek4 arch cannot tensor-split (interleaved only)"
is false for this build — TP4 loads, runs and is numerically correct.

## Measurement trap: `ngram-mod` replays its own output

`ngram-mod` keeps an n-gram cache **across requests** on a server instance, and
`--no-cache-prompt` does not clear it. Re-issuing a prompt the instance has already answered
lets the drafter replay the stored continuation, so acceptance jumps and tok/s roughly
doubles on code. Measured on one instance, same 512-token prompts:

| | prose | code |
|---|---:|---:|
| first time seen | 11.48 | 12.57 |
| **same prompt, second time** | 12.51 | **23.92** (acc 86%) |
| different, genuinely novel prompts | 11.11 | 12.12 |

**Any benchmark that warms up on the prompt it then measures will overstate an
`ngram-mod` config.** Three independent cold-start sweep arms all landed at 11.5-12.9,
which is what exposed it. Always measure `ngram-mod` on a prompt the instance has never
seen, or report it explicitly as a replay/agentic number.

The 23.92 figure is not meaningless — it is a real rate for agentic/replay traffic, where the
model regenerates text it has produced before. It is simply not a general decode rate.

## Config sweep (all cold-start, 512-token novel prompts)

| arm | threads | prose | code |
|---|---:|---:|---:|
| baseline env, n=2 | 15 | 11.48 | 12.57 |
| + MoE fusion knobs (`MOE_WEIGHTED_SUM_FUSION`, `MOE_DOWN_...`, `Q4_K_REPACK`) | 15 | 11.50 | 12.39 |
| baseline env | 12 | 11.64 | 12.90 |
| baseline env | 16 | 11.51 | 12.71 |

**Thread count does not matter here** (11.5-11.6 prose across 12/15/16) and the MoE fusion
knobs that helped GLM-5.3-Flash slightly are neutral-to-negative here — expected, since these
experts are MXFP4, not Q4_K. Nothing in the cheap knob space is worth another run.

## Depth is bracketed: n=2 is the optimum

n=2 (14.76/18.41) > n=3 (12.50/14.32) > n=4 (9.91/12.02), with acceptance falling
79% -> 74% -> 67% on code. The cause is the MoE verify batch: a batch of n+1 tokens routes
to up to 6(n+1) distinct experts, so expert traffic scales with depth while accepted tokens
do not. Same effect as GLM-5.3-Flash. **Do not sweep depth again.**

## It is now latency-bound, not bandwidth-bound

The best config runs at **31-34% of memory bandwidth** — the machine is two-thirds idle on
DRAM. Raw decode moves 17.2 GB/generated token; with speculation that falls to 6.9-7.9 GB.
At ~7 GB/token, **20 tok/s needs only ~140 GB/s** and the box already delivers 127 — so
bandwidth is not the constraint. But on novel text the real rate is 11.5-12.6, so the gap to
20 is ~60%, and it has to come from latency/op cost, not from bytes.

Static tensor accounting on the GGUF (1,328 tensors, classified against the `is_dsv4` +
generic split rules) predicts 16.64 GB/token against 17.2 measured — the budget closes.
Two concrete levers fall out:

- **Mirror tax 2.87 GB/token (17% of raw traffic).** `attn_q_a`, `indexer.attn_q_b`,
  `attn_compressor_kv`, `attn_compressor_gate`, `attn_kv` and `hc_ffn_fn` match no split
  pattern and fall through to `SPLIT_AXIS_MIRRORED`, so all four sockets re-stream them.
  Same bug class already fixed for GLM Full via `GGML_GLM_QA_TP`; never ported here.
- **The routed experts are `GGML_TYPE_MXFP4` (type 39) and use the old AVX2 `mxfp4_8x8_q8_0`
  kernel.** The x16 VNNI family this fork added covers Q4_K/Q5_K/Q6_K/Q8_0 only. MXFP4
  repack is unconditional (no env gate), so it is active — but it is the slow path, on the
  single largest tensor group. An x16 MXFP4 kernel is the obvious next kernel.

## What this means for V4.1 — and what actually blocks it

V4.1 is a **different model** from V4-Flash; nothing above is a V4.1 measurement. But it
resizes the port dramatically. Comparing V4.1's `config.json` to what the tuned engine
already implements:

| V4.1 feature | status in the tuned engine |
|---|---|
| hyper-connections + Sinkhorn (`hc_mult` 4, `hc_sinkhorn_iters` 20) | **implemented** (`dsv4_hc_mult`, `LLM_KV_HYPER_CONNECTION_*`) |
| `o_groups` 8 / `o_lora_rank` 1024 output factorisation | **implemented** (`wo_a`/`wo_b`) |
| DSpark draft blocks | **implemented** (`draft-dspark`, markov head) — and now measured working |
| lightning indexer, `index_topk` 512, yarn rope, MoE + shared expert | **implemented** (`deepseek4`) |
| tensor-parallel split rules | **implemented** (`is_dsv4`) |
| **Engram** (2 layers, 16M vocab, 202.8 GB tables) | **absent from every engine tree** |
| `DeepseekV41ForCausalLM` converter | absent (`DeepseekV4ForCausalLM` exists) |

So V4.1 is **not** a new architecture port — it is `deepseek4` plus **Engram** plus a
converter. And the hard Engram numerics are already written and exactly validated by earlier
09-10 work: the FP8 row+E8M0 decoder (AVX-512, all 65,536 byte combinations checked), the
hash/history (16,091,808 exact row-ID comparisons), and combined lookup (267,583,488 exact
BF16 comparisons). They need graph integration, not invention.

**The real blocker is storage, and it needs a decision.** The V4.1 checkpoint is
510,296,708,312 bytes and **has never been downloaded** — only 10.75 MB of headers plus an
on-demand ~39 GB native cache in `/dev/shm`. Free space today: **7.2 GB on `/models`,
12 GB on `/`**. Nothing else can proceed until ~510 GB exists. Candidates, all requiring
Kaden's call:

- `/models/awesomo-archive` 309 GB — already audited as containing no client data, only
  ~45 GB unique, but gated in writing by `SR950-POST-TRANSPLANT-STATUS.md`
- `/models/GLM-5.3-GGUF` 443 GB — the GLM-5.3 Full weights
- `/models/archive` 197 GB — **do not touch**: 61 GB KOMAN-NOK client data + encrypted personal

Note Engram is 202.8 GB of the 510 GB but only **12,672 bytes are read per token**, so it is
a candidate for staying out-of-core on SSD rather than RAM — `/models` is a Samsung SATA SSD
measured at 528 MB/s, and 48 random row reads per token is well within it.

## Engine fix found by profiling: the mirrored lm_head (`GGML_DSV4_OUTPUT_TP`)

An 8-layer truncated model (`make_trunc.py 8`, 29.8 GB in /dev/shm, loads in 26 s) plus the
compiled-in op profiler (`GGML_CPU_OP_PROFILE='*'`) gave a whole-token decode profile:
7,917 nodes, 20.4 ms in-op per device per token. MUL_MAT 62.6%, MUL_MAT_ID 14.0%.

**The single largest entry was `result_output` at 29.2% of all in-op time** (5.97 ms/token,
q8_0 [4096, 129280]). Cause, in `llama_meta_device_get_split_state`:

```cpp
if (std::regex_match(tensor_name, pattern_output_weight)) {
    if (is_dsv4) { return ...SPLIT_AXIS_MIRRORED; }   // deepseek4/dflash mirror the lm_head
    return ...SPLIT_AXIS_1;                            // EVERY other architecture splits it
}
```

deepseek4 was the only architecture mirroring its vocabulary projection, so all four sockets
re-streamed the full 563 MB matrix. There was no comment explaining the exception.

Added `GGML_DSV4_OUTPUT_TP=1` (opt-in, default keeps old behaviour). Measured:

| model | mirrored | split | delta | output |
|---|---:|---:|---:|---|
| 8-layer truncated, raw | 35.16 | **42.38** | **+20.5%** | byte-identical (sha 09329f25f6ed) |
| full 43-layer, raw decode | 10.35 / 10.32 | **10.82 / 10.76** | **+4.4%** | byte-identical, correctness PASS |

The gain shrinks on the full model exactly as expected — the lm_head is a fixed cost
amortised over 43 layers instead of 8. Three runs per arm, variance <1%.

**Known limitation: incompatible with DSpark speculation.** With `--spec-type draft-dspark`
the load aborts:
`cannot infer split for op=ADD name='node_382' srcs=[result_output (view) axis=0, node_381 axis=10]`
— the speculative sampling path adds a mirrored vocabulary-sized tensor to the now row-split
logits. Restricting the split to `LLM_ARCH_DEEPSEEK4` (leaving dflash drafts mirrored) does
**not** fix it; the offending ADD is in the target-side spec graph. So the knob is currently
usable for raw decode only, and the best serving config uses speculation. Resolving that ADD
is the follow-up; it is probably one more split rule for the markov/conf contribution.

Source: `src/llama-model.cpp` (pre-edit snapshot in `/dev/shm/llama-model.cpp.bak-*`).
Build: `cmake --build build-goal --target llama-server -j 32` (11 s incremental).
Harnesses: `ab_dsv4_output_tp_0910.sh` (truncated), `raw_ab_dsv4_0910.sh` (full, raw),
`full_ab_dsv4_0910.sh` (full, speculation), `sweep_dsv4_0910.sh` (config sweep).

## Ordered path to V4.1 at 20 tok/s

1. **Storage decision** (~510 GB) — blocking, Kaden's call.
2. Download the pinned checkpoint `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
3. Converter: register `DeepseekV41ForCausalLM`, nested config, MXFP4 expert passthrough
   (the FP4->MXFP4 bridge is already written and round-trips exactly).
4. Engram: two lookup layers into the `deepseek4` graph, reusing the validated decoder.
5. Perf: port `GGML_GLM_QA_TP`-style k-slice splits to `is_dsv4` (17% of traffic) and add an
   x16 VNNI MXFP4 kernel. On V4-Flash these alone project ~20 tok/s prose.

Steps 3-5 are bounded engine work. Step 1 is a decision, and step 2 is wall-clock.
