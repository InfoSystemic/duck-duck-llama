# DeepSeek-V4.1-Flash: revised port plan — 2026-09-10

This supersedes the "remaining work" list in `DEEPSEEK-V41-BRINGUP-20260910.md`. That list was
built after checking **upstream llama.cpp at 4ea6d1bb and the Qwen engine**, both of which
lack V4.1. It did not check `engines/llama.cpp-glm5n-goal-0904` — the tuned tree that already
serves GLM-5.3-Flash — which implements far more of V4.1 than anyone realised.

## 1. The port is much smaller than believed

V4.1's HF `config.json` is nested (`text_config` + `vision_config`), and `text_config` uses
**standard HF key names that `DeepseekV4Model` already reads**: `num_hidden_layers`,
`hidden_size`, `num_attention_heads`, `moe_intermediate_size`, `n_routed_experts`,
`n_shared_experts`, `num_experts_per_tok`, `q_lora_rank`, `qk_rope_head_dim`, `head_dim`,
`rope_theta`, `rope_scaling`, `routed_scaling_factor`, `scoring_func`, `topk_method`,
`norm_topk_prob`, `vocab_size`, `rms_norm_eps`, `num_nextn_predict_layers`.

Feature-by-feature against what the tuned engine already does:

| V4.1 feature | value | status |
|---|---|---|
| hyper-connections + Sinkhorn | `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6 | **exercised today** — V4-Flash-0731's GGUF carries `deepseek4.hyper_connection.{count=4, sinkhorn_iterations=20, epsilon=1e-6}` and ran at 10.8 tok/s |
| lightning indexer | `index_topk` 512, `index_head_dim` 128 | **exercised** (`deepseek4.attention.indexer.*`) |
| compressed rope / ratios | `compress_rope_theta` 160000, `compress_ratios` | **exercised** (`deepseek4.attention.compress_*`) |
| `o_groups` / `o_lora_rank` | 8 / 1024 | **exercised** — profile shows `attn_wo_a` q8_0 [4096,1024,2] (V4-Flash uses o_groups=2; V4.1 uses 8) |
| swiglu clamp | `swiglu_limit` 10.0 | **exercised** (`deepseek4.swiglu_clamp_{exp,shexp}`) |
| MoE + 1 shared expert, MXFP4 experts | 384/6/1 | **exercised** |
| DSpark draft | `dspark_*`, markov rank 256 | **exercised** — `--spec-type draft-dspark` measured working |
| tensor-parallel split rules | — | **exercised** (`is_dsv4` branch, TP4 correct) |
| **Engram** | 2 layers, 16M vocab, 202.8 GB | **THE ONLY GAP** |

So the engine work is: **a converter subclass + Engram.** Not an architecture port.

## 2. The storage blocker is smaller than believed, and does not require reclaiming /models

All 48 shard sizes fetched by HEAD (total verified 510.3 GB, matching the audit). The Engram
tables are **cleanly isolated in exactly two shards**:

| shard | size | contents |
|---|---:|---|
| `model-00001-of-00048` | 1.0 GB | vision only |
| `model-00047-of-00048` | 101.5 GB | engram only |
| `model-00048-of-00048` | 101.5 GB | engram only |

| | GB | shards | download @107 MB/s |
|---|---:|---:|---:|
| full checkpoint | 510.3 | 48 | 79 min |
| engram + vision only (skippable for a speed build) | 204.0 | 3 | — |
| **needed for a speed measurement** | **306.3** | **45** | **48 min** |

**306 GB fits in RAM today.** `/dev/shm` has 306 GB free and the GLM-5.3-Flash Q4 tmpfs holds
another 186 GB that is volatile-by-design and re-stageable from a pinned revision. So a first
V4.1 decode measurement needs **zero bytes reclaimed from `/models`**. The full 510 GB build
(with real Engram) is what needs the storage decision — and even then, Engram reads only
12,672 bytes/token, so it belongs mmap'd on the SSD, not in RAM.

## 2b. ATTEMPTED AND MEASURED: converter works, graph does not (2026-09-10)

I built the converter and drove it end to end against real V4.1 weights. **The converter side
is done and works**; the graph side has four concrete gaps, found one at a time by running.

**What works — the first V4.1 GGUF ever produced.** 7 shards (31 GB) of the pinned revision
-> `convert_hf_to_gguf.py` -> a valid 32.2 GB GGUF with all 384 experts repacked to MXFP4 and
V4.1's real hyperparameters (`block_count` 4, `embedding_length` 5120, `expert_count` 384,
`expert_used_count` 6, `hyper_connection.count` 4, `sinkhorn_iterations` 20). Code:
`DeepseekV41Model` in `conversion/deepseek.py` (+ one lazy-map line in `conversion/__init__.py`),
`make_v41_trunc_config_0910.py`. Pre-edit snapshots in `results/goal-0910-restore/*.bak`.

Converter fixes that were needed, in the order they surfaced:
1. register `DeepseekV41ForCausalLM` in the **lazy arch->module map** (`conversion/__init__.py`)
   — registering the class alone is not enough; the real converter never imports the module.
2. emit `text_config` keys at top level: `ModelBase` flattens `text_config` only in
   `TextModel.__init__`, which runs *after* `ModelBase.__init__ -> index_tensors()`, and
   `index_tensors` already needs `num_hidden_layers`.
3. `num_hash_layers: 0` — V4 had token-id->expert `tid2eid` routing tables; V4.1 has none.
4. map `attn.indexer.k_norm` / `attn.indexer.wk` (ids exist, V4's layer_map predates them).
5. **`generate_extra_tensors()` MUST chain to `super()`** — that is where the parent merges
   and writes the MXFP4 experts. Overriding it without chaining silently produced a 3.4 GB /
   93-tensor model instead of 32.3 GB / 105.

**Where it stops — four real graph differences from V4:**

| # | difference | evidence |
|---|---|---|
| 1 | **Engram** (2 layers, 16M vocab) | no implementation in any engine tree |
| 2 | **Indexer key path**: V4.1 has `indexer.wk` + `indexer.k_norm`; V4 derives keys through an `indexer.compressor.*` block | `Unsupported DeepSeek-V4 tensor 'layers.2.attn.indexer.k_norm.weight'` |
| 3 | **No head-level hyper-connection mix**: V4.1's only globals are `embed`/`head`/`norm`; deepseek4 requires `output_hc_{fn,base,scale}` | `tensor 'output_hc_fn.weight' not found` |
| 4 | **Compression ratios 1 and 2**: V4.1 uses `[0,0,2,2,...,1,1,...]`; the loader accepts only 0/4/128. V4.1 also has no `compressor.ape` | `DeepSeek-V4 loader only supports compression ratios 0, 4, and 128` |

Stubbing 2-4 (zero `ape`, zero head-hc, ratios forced to 0) got the model to load-time and
then **crashed with SIGFPE inside graph setup** — a degenerate divide once every KV/index
source list is emptied. That is the honest stopping point: each stub exposes the next thing
the deepseek4 graph assumes about V4's attention, and further stubbing would measure a model
that is no longer V4.1.

**Therefore: no V4.1 tok/s number exists, and one cannot be obtained by stubbing.** The
attention path (items 2 and 4) has to be implemented properly in `src/models/deepseek4.cpp`
before any V4.1 measurement is meaningful.

## 2c. What gap 4 actually requires (settled from V4.1's own reference)

`results/deepseek-v41-intake-0910/official/inference/model.py` documents the mechanism:

> "Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate
> ... only yields every `compress_ratio` steps, holding the partial group in
> `kv_state`/`score_state`", with `state_shape = (max_batch_size, compress_ratio, head_dim)`
> and `wkv` in float32 when `compress_ratio > 1`.

So in V4.1 **`compress_ratio` is a single general pooling factor**. The llama.cpp V4 port
instead hard-codes two instances of it as separate mechanisms:

```
src/models/deepseek4.cpp:263    static constexpr int64_t DSV4_CSA_RATIO  = 4;
src/models/deepseek4.cpp:264    static constexpr int64_t DSV4_HCA_RATIO  = 128;
src/llama-kv-cache-dsv4.cpp:18  static constexpr uint32_t DSV4_CSA_RATIO = 4;
src/llama-kv-cache-dsv4.cpp:19  static constexpr uint32_t DSV4_HCA_RATIO = 128;
```

with the forward pass branching `if (ratio == DSV4_HCA_RATIO ...)` / `if (ratio ==
DSV4_CSA_RATIO ...)` and the dsv4 KV cache allocating distinct state buffers keyed by those
exact values.

**So gap 4 is a generalisation, not a new mechanism**: drive the pooled-KV state shape and
the "emit every N steps" cadence from `ratio` itself instead of from the two constants, and
drop the `coff = ratio == 4 ? 2 : 1` special case. That is real work in
`src/models/deepseek4.cpp` + `src/llama-kv-cache-dsv4.cpp`, and it cannot be numerically
validated on this box without a GPU reference — but it is a bounded, well-defined change
rather than an open-ended port.

Gap 2 (indexer `wk`/`k_norm` instead of `indexer.compressor.*`) is the same shape of problem
in the indexer key path.

## 2d. COMPLETE SPEC for gaps 2 and 4 (read from `official/inference/model.py`)

No guessing is required; the reference defines both precisely.

**`Compressor` is ONE mechanism with a degenerate case** (not V4's two):

```
ratio == 1:  return self.norm(self.wkv(x))          # bf16, NO wgate, NO state buffers
ratio  > 1:  kv, score = wkv(x.float()), wgate(x.float())      # fp32
             pooled = (kv.unflatten(1,(-1,ratio)) * score.unflatten(1,(-1,ratio)).softmax(2)).sum(2)
             return self.norm(pooled)
             # decode: fill slot = start_pos % ratio in (batch, ratio, head_dim) state;
             #         emit ONLY when (start_pos + 1) % ratio == 0, else return None
```

Consequences for the llama.cpp port, all verified against the source:
- **There is no `ape` at any ratio.** V4.1 dropped the compressor's absolute position
  embedding entirely, so the zero-stub used above was semantically faithful, not a hack.
  `attn_comp_ape` should simply not be required for this architecture.
- **`coff` is always 1** — `wkv`/`wgate` are both `Linear(dim, head_dim)`. V4's
  `coff = ratio == 4 ? 2 : 1` is a V4-only quirk.
- **ratio 1 needs no gate, no state, and no fp32 promotion** — it is a plain projection.
  Requiring `attn_comp_wgate` for every non-zero ratio (as the V4 loader does) is wrong here.
- The "emit every `ratio` steps / hold a partial group" cadence is what the dsv4 KV cache
  must express generically instead of via `DSV4_CSA_RATIO` / `DSV4_HCA_RATIO`.

**Gap 2 follows from the same file:** `Indexer.owns_k = layer_id in args.kv_source_layers`,
with the comment *"the index keys are derived from the compressor's latent, so only a layer
that compresses its own KV can produce them; every other indexer reads them from that layer's
cache"*. That is why V4.1 has `indexer.wk` + `indexer.k_norm` on KV-source layers instead of
V4's per-layer `indexer.compressor.*` block.

With this spec the remaining work is ordinary implementation in `src/models/deepseek4.cpp`
and `src/llama-kv-cache-dsv4.cpp`. What it still lacks is a numerical oracle: the only V4.1
reference on this box is the PyTorch server (port 18170), which runs the full 40-layer model,
so it cannot validate a truncated GGUF directly.

## 2e. PROVEN: no subset of V4.1 layers can run on the current engine

I tried to dodge the graph gaps by truncating to layers that need neither a compressor nor an
indexer. V4.1's `compress_ratios` is
`[0,0,2,2,...,2,1,1,...,1,0,0,0]`, `kv_source_layer_ids=[2,8,14,20]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]` — so **layers 0 and 1 have ratio 0 and are
neither KV- nor index-source**. A 2-layer model of them needs no compressor and no indexer:
architecturally clean, no stubs for the attention path at all.

Built it: 4 shards / 17.4 GB downloaded, converted cleanly (**54 tensors, 17.4 GB GGUF**,
every `deepseek4.*` key present and matching V4-Flash's key set exactly — only
`embedding_length_out = 20480` is extra, and that is legitimately `hc_mult * n_embd`).

**It still SIGFPEs**, exactly like the 4-layer stubbed build. Root cause found:

```
src/llama-kv-cache-dsv4.cpp:46    const uint64_t n_rows = ((uint64_t) pos_max + 1)/ratio;
                          :497    plan.state_pos.push_back((int32_t) (pos%ratio));
                          :499    const int64_t n_visible = (int64_t) (pos + 1)/ratio;
                          :519    if ((pos + 1) % ratio != 0) {
                          :526    plan.state_write_idxs.push_back(cache_off + pos/ratio);
```

`ratio` is `hparams.dsv4_compress_ratios[il]`. **With every ratio 0 these divide by zero.**
The dsv4 KV cache assumes a dsv4 model always has compressed layers; only
`dsv4_comp_size()` (line 29) guards with `max(1, ...)`.

So the two escape routes are mutually exclusive:

| layer selection | outcome |
|---|---|
| include any layer with ratio 1 or 2 | loader rejects: *"only supports compression ratios 0, 4, and 128"* |
| include only ratio-0 layers | SIGFPE: divide by zero in the dsv4 KV cache |

**Therefore no V4.1 tok/s measurement is obtainable on this engine at all — not for the full
model, not for any truncation — until compression ratios 1 and 2 are implemented.** That is
now a proven blocker rather than an estimate, and it makes the gap-4 generalisation in §2c/§2d
the single mandatory prerequisite for any V4.1 number.

## 2f. Correction: gap 4 is a NEW KV-cache variant, not a constant refactor

§2c called the fix "a generalisation, drive the ratio from the model". Reading every call
site refutes that. The 28 uses of `DSV4_CSA_RATIO`/`DSV4_HCA_RATIO` are **not symmetric**:

```
src/llama-kv-cache-dsv4.cpp:1316   ..., DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO, ...   <- window = 2x ratio
src/llama-kv-cache-dsv4.cpp:1322   ..., DSV4_HCA_RATIO,   DSV4_HCA_RATIO, ...   <- window = 1x ratio
src/llama-kv-cache-dsv4.cpp:2051   plans_csa(..., DSV4_CSA_RATIO, true,  ...)   <- flag true
src/llama-kv-cache-dsv4.cpp:2054   plans_hca(..., DSV4_HCA_RATIO, false, ...)   <- flag false
src/llama-kv-cache-dsv4.cpp:547    if (ratio == DSV4_CSA_RATIO && !plan.state_pos.empty())
src/llama-kv-cache-dsv4.cpp:602    if (ratio == DSV4_HCA_RATIO && ... && plan.state_write_idxs.empty())
```

plus a **third** cache, `kv_lid` (lightning indexer), also keyed off `DSV4_CSA_RATIO`.

So in the V4 port CSA and HCA are two *different mechanisms* that happen to be selected by
ratio value — different window sizes, different plan-building flags, tier-specific
conditionals. Substituting V4.1's 2 and 1 for 4 and 128 would silently inherit V4's
asymmetry, and V4.1's reference has no such asymmetry: its `Compressor` is **one** uniform
pooling mechanism (§2d).

**Therefore the correct fix is a new KV-cache variant for V4.1's uniform pooled tiers, not a
parameterisation of the existing one.** That is a core-component design change, it touches
~28 call sites plus the `kv_lid` path, and it has **no numerical oracle on this box** — so I
have not written it. Doing so blind would produce a model that loads and emits a tok/s number
with no way to tell whether the attention is correct, which is the failure mode this project
already recorded once ("a throughput-only sweep read 14.1 on garbage").

## 2g. RESOLVED — DeepSeek-V4.1-Flash now RUNS in llama.cpp (first time)

§2e claimed no V4.1 truncation could run. That was right about the symptom and **wrong about
the cause**, which I found only after installing gdb instead of inferring from the source:

```
Thread 1 "llama-server" received signal SIGFPE, Arithmetic exception.
#0  llama_kv_cache::build_input_k_rot(ggml_context*) const
#1  llm_graph_context::build_inp_dsv4() const
```

`src/llama-kv-cache.cpp:1420`:

```cpp
int nrot = 64;
do { nrot *= 2; } while (n_embd_head_k_all % nrot == 0);
nrot /= 2;
```

When `n_embd_head_k_all == 0` — which is exactly what a DSV4 sub-cache (CSA/HCA/LID) reports
when its layer filter matched **no** layers — `0 % nrot == 0` holds for every nrot, so the
loop doubles until `nrot` overflows to 0 and the next `% nrot` raises SIGFPE. This is a
**generic engine bug, not a V4.1 semantic issue**; the same file already guards
`n_embd_head_k_all > 0` at lines 194 and 321, so the omission here is an oversight.

Guarding it (`if (attn_rot_k && n_embd_head_k_all > 0)`) makes V4.1 load and serve.

### First DeepSeek-V4.1-Flash measurements in llama.cpp (TP4, 15 thr/socket)

Layers 0-1 only (the two layers with `compress_ratio == 0`, no engram) — a real V4.1
configuration needing no attention stubs:

| model | `GGML_DSV4_OUTPUT_TP=0` | `=1` | gain | output |
|---|---:|---:|---:|---|
| 1 layer | 51.70 | **124.15** | **+140%** | byte-identical (`f1182d0329`) |
| 2 layers | 48.23 | **100.38** | **+108%** | byte-identical (`c96d4c8738`) |

V4.1's lm_head is q8_0 [5120, 129280]; mirrored on 4 sockets it dwarfed everything else at
this depth. The split is numerically exact (identical hashes both depths, 4 runs each,
variance <1%).

### Ratios 1/2 DO load — the guard was only a whitelist

§2f claimed ratio 1/2 support needed "a new KV-cache variant". Wrong again, and again from
reading rather than running. Deleting the `throw` (and making `attn_comp_wgate`
`TENSOR_NOT_REQUIRED` at ratio 1, since the reference builds no gate there) is enough for a
**ratio-2 compressed layer to load and serve**: 3-layer model, 92.55 tok/s.

Caveat: the forward pass still only *applies* pooling for `ratio == 4 / 128`, so a ratio-2
layer currently loads its compressor weights without using them. Correct for speed
measurement, not for output.

### Fifth structural difference: V4.1 SHARES compressed KV across layers

Building 8 layers failed on `blk.3.attn_compressor_kv.weight not found`. The index shows why:

```
layers owning a compressor: [2, 8, 14, 20]   == kv_source_layer_ids
layers owning an indexer  : [2, 8, 14, 20, 24, 28, 32, 36] == index_source_layer_ids
```

**Only the 4 KV-source layers own a compressor; the other 36 read that source layer's
compressed cache.** deepseek4 assumes every `ratio != 0` layer owns one — a V4 assumption.
This cross-layer KV sharing is a large part of why V4.1 is efficient, and implementing it is
the main remaining graph work.

### Projection to the full 40-layer model

Measured at **four depths** with `GGML_DSV4_OUTPUT_TP=1` (depths 3 and 8 use
`RATIO_SOURCES_ONLY=1`, zeroing the ratio on layers that own no compressor):

| depth | tok/s | ms/token |
|---:|---:|---:|
| 1 | 124.15 | 8.055 |
| 2 | 100.38 | 9.962 |
| 3 | 92.55 | 10.805 |
| 8 | 49.46 | 20.218 |

Least squares: **per-layer 1.741 ms, fixed 6.17 ms, R^2 = 0.9946**.

| configuration | projected 40L |
|---|---:|
| **raw decode** | **13.19 tok/s** |
| + DSpark speculation, conservative 1.2x | 15.8 tok/s |
| + DSpark speculation, optimistic 1.5x | 19.8 tok/s |
| PyTorch reference, measured today | 1.24-1.30 tok/s (~10x slower) |

So the llama.cpp path is worth roughly **10x** the PyTorch path. **It is still short of
20 tok/s**, and the projection is *optimistic*: layers 2-39 carry compression (ratio 2 or 1)
and 8 of them carry the indexer, none of which is implemented or costed here, and Engram is
excluded entirely. A two-point fit over depths 1 and 2 also has a long lever arm to 40.

**Honest status: V4.1 measured at 100-124 tok/s on 1-2 layer truncations; full-model
projection ~12 tok/s raw; the >=20 tok/s goal is NOT met.** Speculation (DSpark) would add
perhaps 1.2-1.5x on top, which brackets 15-18 — so reaching 20 additionally needs the
extraction work (47% -> 58%) described in section 3b.

## 3. Ordered plan

1. **Converter** — subclass `DeepseekV4Model` in `conversion/deepseek.py`:
   register `DeepseekV41ForCausalLM`, flatten `text_config` into hparams, emit the
   `engram_*` KVs, pass MXFP4 experts through (the FP4->MXFP4 bridge is already written and
   round-trips exactly), skip `vision_*`/`aligner`. Small.
2. **Engram, stubbed** — accept and ignore the 12 engram tensors so a 45-shard model loads.
   Gives a valid SPEED number for 38 of 40 layers; label it as speed-only, not quality.
3. **Download 45 shards (306 GB)** into tmpfs — 48 min, no `/models` reclaim.
4. **Convert + measure** raw decode and `ngram-mod,draft-dspark` n=2 on the tuned engine.
   Expect the V4-Flash ballpark (~10-12 tok/s) since active bytes/token are comparable.
5. **Engram for real** — integrate the already-validated decoder (FP8+E8M0 AVX-512, all
   65,536 byte combinations checked; 16,091,808 exact row IDs; 267,583,488 exact BF16).
   Needs the 2 extra shards => the storage decision.
6. **Perf to 20** — carry over `GGML_DSV4_OUTPUT_TP` (needs its DSpark ADD fixed), the
   2.87 GB/token mirror tax, and an x16 VNNI MXFP4 expert kernel.

Steps 1-4 need no permission and no reclaim. Step 5 is where storage becomes blocking.

## 3b. Is 20 tok/s reachable for V4.1? Computed from its real config — YES, in principle

Per-token active bytes from `text_config` (40 layers, dim 5120, ffn 2304, 6+1 of 384 experts,
FP4 experts at 4.25 bpw, FP8 attention, vocab 129,280):

| component | GB/token |
|---|---:|
| routed + shared experts | 5.26 |
| attention (fp8) | 5.06 |
| lm_head | 0.66 |
| **ideal total, perfectly split** | **10.99** |
| with V4-Flash's measured 1.21x mirror/overhead factor | 13.30 |

| basis | 20 tok/s requires | % of 380 GB/s |
|---|---:|---:|
| ideal 10.99 GB/tok | 220 GB/s | **58%** |
| realistic 13.30 GB/tok | 266 GB/s | 70% |

**This box has already sustained 241 GB/s (63.5%) on GLM-5.3-Flash Q8 raw decode**, so the
220 GB/s figure is inside demonstrated range, and DSpark speculation reduces bytes per
*generated* token further. The DeepSeek stack currently achieves 178 GB/s (47%), so the gap
is extraction efficiency, not capacity.

**V4.1 is therefore fundamentally unlike GLM-5.3 Full**, which needs 451 GB/s against a
380 GB/s ceiling and is arithmetically impossible. V4.1 at 20 tok/s is a hard but legitimate
engineering target, contingent on (a) the four graph gaps and (b) raising extraction from
47% toward ~58%.

Note this **supersedes the "expect 10-12 tok/s" estimate below**: that extrapolated from
V4-Flash's measured 17.2 GB/token, but V4.1's FP4 experts make it *lighter* per token
(10.99 GB ideal), not heavier.

## 4. Honest expectation

V4-Flash on this box does **11.5-12.6 tok/s on novel text** after all of today's work.
V4.1 has comparable active bytes/token (~5.0 GB vs ~4.3 GB), so a first V4.1 number in the
same 10-12 range is the reasonable expectation — **not 20**. Reaching 20 needs the step-6
kernel work on top, and that is a separate, real project.
