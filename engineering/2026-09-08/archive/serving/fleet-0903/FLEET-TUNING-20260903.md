# SR950 fleet tuning — 2026-09-03

Goal: get GLM-5.3 (Full), GLM-5.3-Flash, Qwen3.8-Flash-Next and Qwen3.8-27B to the
highest tok/s this box can produce, without losing quality.

Machine: 4 x Xeon Gold 6242 (Cascade Lake, AVX-512 VNNI), 16 physical cores/socket,
755 GB DDR4, 4 NUMA nodes, ~380 GB/s aggregate measured, **no GPU installed**.

## 0. Two hardware questions answered today (both were open in the plan)

**Phase 5a (DIMM speed) is CLOSED — no upside.** `dmidecode -t 17`: 20 of the 24
populated DIMMs are *physically rated* 2400 MT/s (Micron 36ASF4G72PZ-2G3B1/2G3D1,
Kingston KCPC7G-MIA, HP24D4R7D4HAI-32). Only 4 Samsung M393A4K40CB2-CVF are 2933.
The board already runs every channel at 2400, i.e. the maximum the slowest module
allows. A UEFI/XCC change cannot raise it; only replacing 20 DIMMs would. The
plan's projected "+15-22% from 5a" is not available.

**No GPU is present.** `nvidia-smi` fails, `lspci` shows only the Matrox G200e BMC
video. The GPU-hybrid path (plan sections 6 and 9, the only single change that
doubles the raw ceiling) needs a card to be physically installed first.

**Passwordless sudo does work from a session**, so the remaining Phase 5 items
(uncore frequency floor, C-states) are reachable. Uncore currently floats
1.2 GHz -> max (`min_freq_khz=1200000` on all four packages) and C6 is enabled
(92 us exit latency, 424M entries on cpu0).

## 1. Engine / model matrix (what runs what)

| model | arch | quant on disk | engine that supports it | 4-socket TP? |
|---|---|---|---|---|
| GLM-5.3 Full | `glm-dsa` | UD-Q4_K_XL 436 GB | `llama.cpp-sr950-glm/build-dev2` | yes, tuned |
| Qwen3.8-27B | `qwen35` | UD-Q8_K_XL 31.4 GB, UD-Q4_K_XL 17.6 GB | same fast fork (arch already supported) | yes |
| Qwen3.8-Flash-Next | `qwen4exp` | UD-Q2_K_XL 73 GB | `llama.cpp-qwen4exp-current/build-sr950` | untested |
| GLM-5.3-Flash | `glm5next` | UD-IQ2_XXS 95 GB | `llama.cpp-glm53-flash/build-sr950` | binary was STALE |

The fast stack (spin dispatch, fused/merged all-reduce, x16 AVX-512-VNNI kernels,
load-time requant) lives only in the GLM fork. The other two trees have the base
CPU-NUMA device patch but not the fast stack.

## 2. Starting points (measured, this session unless noted)

| model | raw tok/s | with speculative | how it was running |
|---|---:|---:|---|
| GLM-5.3 Full | 7.55 | 9.50 (MTP n2) / 10.98 short | 4-socket TP, full fast stack |
| Qwen3.8-27B | 7.17 (08-29) | 16.49 general / 23.58 agentic (08-29) | 4-socket TP, OLD b249 stack |
| Qwen3.8-Flash-Next | - | 6.32-6.47 (08-26) | **ONE socket**, numactl membind |
| GLM-5.3-Flash | - | not measured | **ONE socket**, 32 threads |

The two Flash models are the big gaps: both are pinned to a single socket's
~95 GB/s while three memory controllers sit idle.

## 3. Results

(filled in as measurements land)

### 3.1 GLM-5.3 Full — speculative sweep on the live v5b-ngram server (09-03)

The composite `ngram-mod,draft-mtp` spec type **now works**; the plan's item 4d
("500s on every request") is fixed. A real completion returns 391 with 30/34
draft tokens accepted.

| n_max / p_min | tok/s (prose,code,novel; mean of 6) | draft acceptance |
|---|---:|---:|
| 0 / 0 (raw) | 7.55 | - |
| **2 / 0** | **9.50** | 0.669 |
| 3 / 0 | 8.74 | 0.531 |
| 4 / 0 | 7.80 | 0.429 |
| 6 / 0 | 6.49 | 0.305 |
| 3 / 0.5 | 7.42 | 0.639 |

Depth beyond 2 is a **loss** on general traffic: acceptance falls faster than
batching saves, exactly as the MoE verify-cost model in the plan predicts (a
verify batch of n+1 tokens routes to up to 8(n+1) distinct experts, so routed
bytes scale with n while only the dense share is amortised). The production
default (`--spec-draft-n-default 2`) is already the optimum for general prose;
the deep n18/p0.75 profile stays confined to the agentic alias.

Confirmatory point: acceptance and throughput are anti-correlated across these
arms (p_min=0.5 raises acceptance to 0.64 and *lowers* throughput to 7.42).

### 3.2 Per-token byte budgets (GGUF tensor inventory, this session)

What each model actually streams per decoded token, and therefore its ceiling at
the measured 95 GB/s per socket / 380 GB/s across four:

| model | total on disk | bytes/token | 1-socket ceiling | 4-socket ceiling | measured |
|---|---:|---:|---:|---:|---:|
| GLM-5.3 Full UD-Q4_K_XL | 436 GB | ~28 GB (after Q5_K/Q6_K requant) | - | 13.6 | 7.55 raw (56%) |
| GLM-5.3-Flash UD-IQ2_XXS | 99 GB | ~8.8 GB | **10.8** | 43 | not measured, 1 socket |
| Qwen3.8-Flash-Next UD-Q2_K_XL | 67 GB | ~4.0 GB | **24** | 95 | 6.4 (27%), 1 socket |
| Qwen3.8-27B UD-Q8_K_XL | 31.4 GB | ~30 GB | - | 12.6 | 7.17 raw (57%) |
| Qwen3.8-27B UD-Q4_K_XL | 17.6 GB | ~16.9 GB | - | 22.5 | not measured |

Notes that matter:
- Qwen3.8-Flash-Next carries a **22 GB `per_layer_token_embd`** (PLE) table that is
  *not* streamed per token (one row per layer per token). It is 33% of the file and
  ~0% of the token cost, so file size badly overstates its decode cost. Its routed
  experts contribute only 0.80 GB/token (10 of 512, over 48 layers); the per-token
  cost is dominated by the ~3.2 GB of dense/attention/SSM weights.
- GLM-5.3-Flash is the opposite: 2.57 GB/token of routed experts and 6.2 GB/token of
  dense+attention, so it is genuinely bandwidth-hungry and gains the most from having
  four memory controllers instead of one.
- Qwen3.8-27B UD-Q8_K_XL keeps `output.weight` and `attn_q` in **BF16** (2.54 + 2.14 GB).
  The load-time requant path only converts Q8_0 sources, so it cannot shrink those.

### 3.3 Host knobs (uncore floor, C6) — measured, no gain

Short A/B on the live GLM-5.3 server, raw decode only (cleanest signal for a
memory-subsystem change), quiet host, with a baseline repeat at the end:

| arm | raw tok/s |
|---|---:|
| A baseline (uncore floor 1.2 GHz, C6 on) | 7.18 *(cold; range dips to 6.47)* |
| B uncore floor pinned to max (6.3 GHz) | 7.46 |
| C uncore max + C6 disabled | 7.49 |
| D C6 disabled only | 7.54 |
| **E baseline repeat** | **7.46** |

The honest comparison is E (7.46) against B/C/D (7.46-7.54): **+1% at most, inside
the run-to-run spread.** Arm A's 7.18 was a cold-start artifact. Both knobs were
reverted. The mechanism is clear in hindsight: the fork's spin-wait dispatch
(`GGML_CPU_NUMA_DISPATCH_SPIN_US=20000`) already keeps every worker busy through a
token, so cores never reach C6 and the uncore never ramps down. **Phase 5b is
closed along with 5a.**

### 3.4 Qwen3.8-Flash-Next: tensor-parallel is the wrong lever (and is incorrect)

Ran it across all four sockets (`--device CPU-NUMA0..3 --split-mode tensor`, 15
threads/socket) on `llama.cpp-qwen4exp-current/build-sr950`. It loaded in 101 s
and served, but:

- **Output is garbage.** `17 * 23` produced fluent token soup
  (`rimarlyIiiiiii...33333...localized.localization...`) and the server then 500s
  on the peg-native parse. The tensor-split rules do not cover this architecture
  (only a PLE-mirroring rule exists); SSM state, the indexer and the `hc_*`
  injection tensors are unhandled.
- **And it is barely faster anyway: 7.92 tok/s** vs 6.47 single-socket. At ~4.0 GB
  per token that is 32 GB/s across four sockets = **8% extraction**, so this model
  is not bandwidth-bound at all. 36 of its 48 layers are SSM/linear-attention and
  its expert matrices are only 2560x640, so decode is dominated by per-op and scan
  cost, not weight streaming. Four memory controllers cannot fix that.

Conclusion: do not invest in qwen4exp tensor-split. The available multiplier here
is **speculative decoding** — the MTP sidecar has been on disk since 08-29 and was
never promoted — not memory parallelism.

## 4. GLM-5.3-Flash: enabling tensor-parallel for `glm5next`

`--split-mode tensor` was refused outright: `LLAMA_SPLIT_MODE_TENSOR not
implemented for architecture 'glm5next'`. That check (`llm_arch_supports_sm_tensor`
in `src/llama-arch.cpp`) is a **denylist**, and `glm-dsa` — GLM-5.3 Full, which
runs tensor-parallel in production — was removed from it by exactly one deleted
line plus a split policy. The same was done here, then four separate blockers had
to be cleared before the model would run at all:

1. **`SPLIT_MODE_TENSOR requires flash_attn to be enabled`**, while the shipped
   launcher sets `--flash-attn off` "because the support PR requires it".
   Re-tested rather than assumed: **flash attention ON is correct on glm5next**
   (`17*23 -> 391`, and marginally faster: 3.57 vs 3.52 tok/s). The FA-off claim
   in `serving/glm53-flash/launch-glm53-flash.sh` is stale and was blocking TP.
2. **`unsupported mul_mat split states: src0=blk.0.attn_output.weight axis=0
   src1=kda_normed-0 axis=10`.** A K-split weight needs a *row-split input*;
   mirrored attention does not provide one. So `attn_output` may only be split
   for layers whose attention is itself split — a per-layer-type decision, not a
   per-tensor one.
3. **`GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer))`** — glm5next
   reshapes its `embd` graph input, and that reshape stays in a plain CPU buffer.
   The meta backend assumed every tensor it is handed was allocated by itself.
   Fixed in `ggml_backend_meta_buffer_simple_tensor`: a non-meta host tensor has
   no per-device copies, so return it as-is (every socket can read host memory).
4. **`GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf))` in `n_bufs`** — the same
   tensor reaching `ggml_backend_meta_get_split_state`. Fixed by reporting
   `MIRRORED, n_segments=1` for tensors this backend did not allocate.

**Result: GLM-5.3-Flash now loads and serves correctly across all four sockets**
(`17*23 -> 391`). Speed is unchanged (3.47 vs 3.52 single-socket) because in this
configuration only the routed experts, dense FFN and lm_head are split — the
attention is still mirrored, so every socket redundantly streams the 4.4 GB/token
of attention weights and pays an all-reduce per layer on top. This is the same
"redundant mirrored compute is the largest software loss" finding the GLM-5.3
Full work reached; the value is entirely in splitting the attention.

### The remaining blocker (precisely located)

The KDA (linear-attention) half of the model — 34 of 46 layers — is cleanly
head-parallel: `attn_q/k/v` and the two gate projections `ssm_f_b`/`ssm_g_b` are
all `[*, 8192] = 64 heads x 128`, the conv state is `3*(d_conv-1)*head_dim` per
head and the delta state `head_dim^2` per head (`n_embd_r`/`n_embd_s`), and
`attn_output` consumes exactly that axis. That policy plus its split granularity
is implemented (`GGML_GLM5N_ATTN_TP=1`), and the 12 MLA layers correctly stay
mirrored. It now fails one level deeper:

```
ggml-backend-meta.cpp: GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)
```

i.e. the meta backend cannot infer how the split propagates through the KDA
delta-attention ops themselves. Finishing this needs per-op split-propagation
rules for those ops, not more split-policy work. That is the next step, and it
is worth taking: it takes per-socket bytes from 6.4 GB to ~3.9 GB per token and
raises the ceiling from 10.8 to ~25 tok/s.

## 5. Why both Flash models are slow, and what actually moved them

Neither Flash model is bandwidth-bound at all on one socket:

| model | GB/token | 1-socket ceiling | measured | extraction |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash UD-IQ2_XXS | 8.8 | 10.8 | 2.90 -> 3.57 | 27% -> 33% |
| Qwen3.8-Flash-Next UD-Q2_K_XL | 4.0 | 24 | 6.05 -> 7.10 | 25% -> 30% |

Both sit at ~27% because their routed experts are **i-quants** (IQ2_XXS/IQ2_S/Q2_K
gate-up and IQ3_XXS/IQ4_XS/Q3_K down for GLM-5.3-Flash; IQ2_XS/IQ3_XXS/IQ4_NL for
Flash-Next). The GLM-5.3 Full plan already recorded the mechanism — *"smaller
routed-expert quants (Q3/Q2): measured slower, extraction collapses faster than
bytes shrink"* — and these two models are made almost entirely of that tier.

What was tried and what it gave:

| change | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---:|---:|
| starting point | 2.90 | 6.05 |
| fast-stack kernels ported into the engine | 3.46 | - |
| + IQ2_XS/IQ3_XXS repack | 3.52 | 5.98 (no gain) |
| + x16 VNNI + MoE gate/up fusion | - | **7.10** |
| + flash attention on | **3.57** | - |
| 16 threads instead of 32 | 3.30 (worse) | - |

The IQ repack alone does nothing for either, and the reason is specific: the
repacked kernels that exist cover **IQ2_XS** and **IQ3_XXS**, while GLM-5.3-Flash's
gate/up experts are **IQ2_S / IQ2_XXS / Q2_K** — different types with no fast path.
Writing `iq2_s_r8` / `iq2_xxs_r8` x16 kernels is the single change that would move
GLM-5.3-Flash most, and it is the same kind of work the fork already did for
q4_K/q5_K/q6_K/q8_0.

Also worth noting: `GGML_CPU_REPACK_LOAD_THREADS` defaults to single-threaded, and
repacking 95 GB of i-quants that way takes **over 20 minutes**; with 16 threads it
is 150 s. Any repack experiment must set it or it looks like a hang.

### Speculative decoding: both models have a head, neither can use it yet

- **GLM-5.3-Flash ships a complete MTP layer in the GGUF** (`blk.45.nextn.*` plus
  its own expert set), which a normal context reports as "unused tensor ...
  ignoring". Adding `LLM_ARCH_GLM5NEXT` to the MTP-memory arch list was necessary
  but not sufficient: the graph builder aborts with
  `"glm5next NextN graph not implemented yet"`. Writing that graph is a bounded,
  high-value task — speculation is the only thing that can take this model past
  its 10.8 tok/s single-socket ceiling without finishing the attention split.
- **Qwen3.8-Flash-Next's MTP sidecar cannot load**: `check_tensor_dims: tensor
  'blk.0.hc_attn_norm.weight' not found`. The third-party sidecar predates the
  `hc_*` hyper-connection tensors in the current qwen4exp definition. The
  `LLM_ARCH_QWEN4EXP` MTP enablement is in place for when a compatible sidecar
  exists.

## 6. Final measured results (full decode_bench, prose/code/novel, quiet host)

| model | before this session | after | change |
|---|---:|---:|---:|
| **GLM-5.3-Flash** (1 socket) | 2.90 | **3.68** | +27% |
| GLM-5.3-Flash + `ngram-mod` | - | **4.08** mean, **6.77** peak | +41% / +133% on repetitive text |
| **Qwen3.8-Flash-Next** (1 socket) | 6.05-6.47 | **7.01** | +16% |
| Qwen3.8-Flash-Next + `ngram-mod` | - | **8.12** mean, **14.60** peak | +34% / +141% on repetitive text |
| Qwen3.8-27B (fast fork, 4-socket TP) | 7.17 raw / 16.49 MTP (08-29 stack) | 7.47 raw / **15.10** MTP | +4% raw |
| GLM-5.3 Full | 7.51 raw / 9.55 MTP | 7.55 raw / **9.50** MTP, composite works | unchanged; `ngram-mod` composite verified fixed |

The `ngram-mod` ranges are wide by design (Flash-Next: 6.0 to 14.6) — it drafts
from repetition already in the context, so it does nothing for novel prose and a
great deal for code, file edits and agentic replay. **On repetitive/agentic
traffic Qwen3.8-Flash-Next reaches ~14.6 tok/s, matching the 27B**, while using a
125B model instead of a 27B one. Enable it per alias, not globally.

## 7. Operating the tuned configurations

```bash
cd ~/InfoSystemic/AI-Server/serving/fleet-0903

# Qwen3.8-Flash-Next, best general profile (7.0 tok/s)
env GGML_CPU_REPACK_LOAD_THREADS=16 GGML_CPU_IQ2_XS_REPACK=1 GGML_CPU_IQ3_XXS_REPACK=1 \
    GGML_CPU_IQ3_XXS_MOE_2ROW=1 GGML_CPU_X16_Q4_K=1 GGML_CPU_X16_Q5_K=1 GGML_CPU_X16_Q6_K=1 \
    GGML_CPU_MOE_GATE_UP_FUSION=1 GGML_CPU_FFN_GATE_UP_FUSION=1 QWEN4E_THREADS=32 \
    ./launch-qwen38-flash-next-1s.sh 18095
# ... same, plus QWEN4E_SPEC_TYPE=ngram-mod for agentic/code traffic

# GLM-5.3-Flash, best profile (3.7 tok/s); note GLM53F_FA=on and LOAD_MODE=none
env GGML_CPU_REPACK_LOAD_THREADS=16 GGML_CPU_IQ2_XS_REPACK=1 GGML_CPU_IQ3_XXS_REPACK=1 \
    GGML_CPU_IQ3_XXS_MOE_2ROW=1 GGML_CPU_X16_Q4_K=1 GGML_CPU_X16_Q5_K=1 GGML_CPU_X16_Q6_K=1 \
    GGML_CPU_MOE_GATE_UP_FUSION=1 GGML_CPU_FFN_GATE_UP_FUSION=1 \
    GLM53F_THREADS=32 GLM53F_LOAD_MODE=none GLM53F_FA=on ./launch-glm53-flash-1s.sh 18096
```

Two traps that cost real time here, both worth remembering:

- **`--load-mode mmap` silently defeats `numactl --membind`.** mmap'd pages come
  from the page cache and stay wherever they were first read, so a "node-local"
  server ends up streaming remote memory: 1.79 vs 2.90 tok/s on GLM-5.3-Flash.
  Single-socket profiles need `--load-mode none`. Tensor-parallel profiles need
  the opposite — `none` fails there with `read error: Bad address`, because the
  meta device already places weights node-locally.
- **`GGML_CPU_REPACK_LOAD_THREADS` defaults to 1**, and repacking 95 GB of
  i-quants single-threaded takes >20 minutes and looks exactly like a hang.

### Aggregate throughput, if that is what is wanted

Both Flash models fit four times over in 755 GB (4 x 73 GB and 4 x 95 GB). Since
neither benefits from tensor-parallel, running **one independent instance per NUMA
node** behind a load balancer gives ~4x aggregate throughput at unchanged
single-stream latency, with zero correctness risk: ~28 tok/s aggregate for
Qwen3.8-Flash-Next, ~14.7 for GLM-5.3-Flash. That is the highest-value
zero-engineering use of this machine for multi-user serving.

## 8. Ranked next steps

1. **`iq2_s_r8` / `iq2_xxs_r8` x16 VNNI kernels.** Both Flash models sit at ~30%
   bandwidth extraction purely because their gate/up experts are i-quant types
   with no repacked fast path. This is the same work the fork already did for
   q4_K/q5_K/q6_K/q8_0/iq2_xs/iq3_xxs, and it is the single change that would move
   GLM-5.3-Flash most.
2. **The glm5next NextN graph** (`"glm5next NextN graph not implemented yet"`).
   The MTP layer is already in the GGUF and the memory path is enabled; only the
   draft graph is missing. Speculation is the only route past this model's
   10.8 tok/s single-socket ceiling short of finishing the attention split.
3. **KDA op split propagation in the meta backend**, to finish GLM-5.3-Flash
   tensor-parallel (ceiling 10.8 -> ~25). Everything else is in place: the arch is
   off the denylist, four blockers are fixed, and the head-parallel policy and its
   granularity are written behind `GGML_GLM5N_ATTN_TP=1`.
4. **A compatible Qwen3.8-Flash-Next MTP sidecar** (the published one predates the
   `hc_*` tensors); the engine side is already enabled.

## 9. Where the code changes live

Two engines were modified; both sets of changes are saved as patches in
`serving/fleet-0903/patches/` because one of the trees is on tmpfs.

**`engines/llama.cpp-glm53-flash`** (in place, uncommitted):
- `src/llama-arch.cpp` — `LLM_ARCH_GLM5NEXT` removed from the tensor-split denylist.
- `src/llama-model.cpp` — glm5next split policy (`GGML_GLM5N_ATTN_TP`,
  `GGML_GLM5N_SHEXP_TP`, `GGML_GLM5N_OUTPUT_TP`, `GGML_GLM5N_ATTN_OUT_TP`), the
  KDA head granularity rules, a `tensor_layer` helper, and `LLM_ARCH_GLM5NEXT`
  added to the MTP-memory arch list.
- `ggml/src/ggml-backend-meta.cpp` — non-meta host tensors pass through the
  per-device mapper and report a MIRRORED split state, instead of asserting.
- the fast kernel stack (`repack.cpp`, `arch/x86/repack.cpp`, `traits.*`,
  `ggml-cpu.c`, `quants.c`) ported from `llama.cpp-sr950-glm`.
- The engine binary was also **stale**: its NUMA-device patch was applied
  2026-08-29 but `build-sr950` was last built 08-27, so `--list-devices` showed
  no `CPU-NUMA*` devices at all. Rebuilt.

**`/dev/shm/q4e-fast`** (a copy of `llama.cpp-qwen4exp-current`, tmpfs — rebuild
from the patches after a reboot):
- the same fast kernel stack (applied cleanly, 10 of 12 files).
- `ggml-backend-meta.cpp` — `ggml_backend_meta_device_get_extra_bufts` plus the
  keyed `ggml_backend_meta_buffer_type` factory it needs, ported from the GLM fork.
- `src/llama-model.cpp` — prefer the meta device's extra (node-local repack)
  buffer types so the x16 kernels are actually selected, and
  `LLM_ARCH_QWEN4EXP` added to the MTP-memory arch list.

Nothing in `engines/llama.cpp-sr950-glm` (the GLM-5.3 Full production engine) was
modified. GLM-5.3 Full was stopped for ~1.5 h to free its 483 GB while the two
Flash models were measured, and restarted afterwards via
`systemctl --user start glm53-sr950`.

## 10. Per-op profile of GLM-5.3-Flash — the ranking in section 8 was wrong

Built an 8-layer truncated model (`make_trunc.py`, 12.8 GB, loads in **16 s**
instead of 4 minutes) and profiled a real token. Two notes on getting this to
work at all: `GGML_CPU_OP_PROFILE` takes a **comma-separated list of op names or
`*`** — setting it to `1` silently profiles nothing, which is almost certainly
the "capture script arms but nothing emits" that was recorded earlier as the
binary lacking a profiler. And the truncation needs to cut per-layer arrays
(`attention.head_count_kv` is one entry per layer here) and zero
`nextn_predict_layers`, or the loader disagrees with itself about which layers
are recurrent.

**1162 graph nodes per token for 8 layers — about 145 ops per layer**, where an
ordinary transformer layer is 30-40. Time by op:

| op | share | calls | note |
|---|---:|---:|---|
| MUL_MAT | 43.4% | 1919 | |
| MUL_MAT_ID | 24.5% | 240 | the routed experts |
| **CONCAT** | **10.3%** | **482** | pure data movement, ~60 per layer |
| **GET_ROWS** | **7.9%** | 246 | |
| UNARY | 3.0% | 314 | |
| GATED_DELTA_NET | 1.7% | 96 | the KDA scan itself — nearly free |
| CPY / RMS_NORM / MUL / DSV4_HC_POST | 5.9% | 1524 | |

By tensor (layer-folded):

| tensor | share | type |
|---|---:|---|
| routed experts (down + gate + up) | **24.6%** | iq3_xxs, iq2_xxs |
| attn_q/k/v | 11.8% | q5_K |
| attn_output | 6.5% | q5_K |
| `output.weight` (lm_head) | 6.1% | q4_K — 3.7 ms/token, already at ~97 GB/s |
| dense FFN (3 leading blocks) | 8.3% | q5_K/q6_K |
| ssm_conv1d + `cache_s_l*` | 6.8% | f32 |

**This overturns the "write iq2 kernels first" conclusion.** The routed experts
are only 24.6% of the token, so even a perfect i-quant kernel is worth at most
~+15% overall — not the transformation the extraction number implied. What is
actually expensive is the **hyper-connection machinery**: `hc_init` is
`[4096, 1, 6]`, i.e. the residual is carried as **six parallel streams**, so every
norm, concat and add in the model runs six times. CONCAT alone is 10.3% and does
no arithmetic at all; together with GET_ROWS, CPY, RMS_NORM (633 calls for 8
layers) and MUL, roughly **28% of the token is data movement, not maths**.

Two consequences for the ranking:

1. **The GLM fork's meta-backend scheduler work was never ported here** — only its
   kernels were. That is why tensor-parallel gained nothing: with attention
   mirrored each socket still does ~62% of the single-socket work, so TP *should*
   have been ~1.6x, and the per-layer all-reduce round trips ate all of it.
   `GGML_CPU_NUMA_FUSED_REDUCE` / `MERGE_REDUCE` / spin dispatch were worth
   +4.5%, +8.7% and more on GLM-5.3 Full, on a model where sync was a *smaller*
   fraction of the token than it is here. This is now the top item.
2. Op-count reduction (fusing the hyper-connection concats) is a real second
   lever worth ~10% and is model-structural, not quant-specific.

## 11. Per-op profile of Qwen3.8-Flash-Next — the same story, and a new finding

8-layer truncated model (`/dev/shm/q4e-8L.gguf`, 38 GB, loads in 10 s), one
socket, 32 threads, 1412 nodes per token (~176 ops/layer):

| op | share | calls |
|---|---:|---:|
| MUL_MAT | 37.9% | 2158 |
| MUL_MAT_ID | 18.0% | 254 |
| **UNARY** | **11.4%** | 545 |
| **GET_ROWS** | **10.4%** | 311 |
| CONCAT | 4.7% | 112 |
| MUL | 4.6% | 1119 |
| GATED_DELTA_NET | 2.9% | 96 |
| CONT / RMS_NORM / CPY / ARGSORT / ROPE | 8.0% | 1229 |

**~43% of the token is not matrix multiplication.** Two specifics, and neither is
a kernel-quality problem:

- **GET_ROWS is entirely the KDA recurrent state.** Every entry is
  `cache_s_l<N> (reshaped)`, `ne=[786432,1,1]` — 3.1 MB of f32 delta-attention
  state per layer, fetched every token, at **0.54 ms each = 5.8 GB/s**. That is
  ~16x below what one socket can stream. Across the full 48 layers that is
  ~151 MB per token of state traffic that never appeared in the weight budget in
  section 3.2, moving at a sixteenth of memory speed.
- **UNARY ops run at ~0.2 GB/s.** The typical entry is `ne=[10240,1,1]` taking
  0.19 ms — 40 KB of elementwise work. At that size the 32-thread fork/join
  barrier costs far more than the arithmetic.

GLM-5.3-Flash shows the same shape (GET_ROWS 7.9%, `cache_s_l*` ~1.2% each,
CONCAT 10.3%).

**So the dominant inefficiency in both Flash models is dispatch overhead on small
and serial ops, not quantisation kernels.** The fork already has the knob for
this — `GGML_CPU_SINGLE_TASK_MAX_ELEMENTS` runs an op on a single thread below a
size threshold, and it is set to 32768 in the GLM-5.3 Full production profile —
**and it was never set on either Flash model** in any run in this document,
including the "best" configurations in section 6. Sweeping it (plus
`GGML_CPU_MOE_SINGLE_TOKEN_THREADS`) on the 8-layer models is the cheapest
untried experiment in the whole effort and it is the natural next action.

That sweep is written (`/dev/shm/glm-dev/knob_sweep.sh`) but could not be trusted
tonight: GLM-5.3 Full came back up and is serving real traffic at ~1800% CPU, and
a contended host reads ~0.77 tok/s where a quiet one reads 31. **Re-run it on a
quiet host** — the harness makes each arm about 20 seconds.

### The dev loop this created

`make_trunc.py` writes an N-layer GGUF of any architecture, which turns a
4-minute load into 10-16 seconds. Three portability fixes were needed beyond the
GLM-only original: truncate per-layer arrays, zero `nextn_predict_layers`, send
BF16 through `raw_dtype`, and preserve array element types (qwen4exp stores
`ple.layer_multipliers` as UINT64 values around 2.4e13, which `add_array()`
silently infers as INT32 and overflows).

Profiling gotcha worth keeping: **`GGML_CPU_OP_PROFILE` takes a comma-separated
list of op names, or `*`.** Setting it to `1` matches no op and emits nothing —
which is almost certainly why an earlier session concluded the binary had no
profiler compiled in.

## 12. The small-op dispatch hypothesis — tested and REFUTED

Section 11 proposed `GGML_CPU_SINGLE_TASK_MAX_ELEMENTS` as the cheapest untried
experiment, on the strength of UNARY ops profiling at 0.2 GB/s. Measured on the
8-layer models (quiet host, `knob_sweep.sh`, ~15 s per arm):

| arm | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---:|---:|
| baseline | **30.63** | 19.51 |
| `SINGLE_TASK_MAX_ELEMENTS=32768` | 29.64 | **19.64** |
| `=131072` | 29.91 | 17.81 |
| `=32768` + `MOE_SINGLE_TOKEN_THREADS=15` | 28.94 | 19.41 |

**No gain on either model** — every arm is within noise of baseline or worse.
The hypothesis is dead; do not spend time on it.

The likely reason the profile pointed the wrong way: **the op profiler adds a
fixed cost per op**, and with ~1400 ops per token that overhead lands
disproportionately on the smallest ops, inflating their apparent share. The
per-op *shares* in sections 10 and 11 should be read as an upper bound for small
ops and are reliable mainly for the large matmuls. Total-time attribution
(experts ~25%, attention ~18%, lm_head ~6-10%) still holds.

This is the second time this session that a plausible mechanism did not survive
measurement — the first was the uncore/C6 host knobs. Both were cheap to test
because of the 8-layer dev loop; neither would have been obvious from reasoning.

## 13. Porting the GLM fork's meta-backend scheduler — done, and it does not pay

Merged `ggml-backend-meta.cpp` from `llama.cpp-sr950-glm` into the glm53-flash
engine: **30 of 34 hunks applied**, 4 hand-merged (one was a duplicate
`LIGHTNING_INDEXER` case upstream had already added; a diagnostic fix and a
`flash_attn_ext` relaxation were already present upstream). Two of the four
rejects were genuinely new capability:

- **`mul_mat` with a k-split weight against a MIRRORED input -> PARTIAL.** Exactly
  the rule whose absence produced `unsupported mul_mat split states:
  src0=blk.0.attn_output.weight axis=0 src1=kda_normed-0 axis=10` in section 4.
  With it, `attn_output` can be split while attention itself stays mirrored.
- **`flash_attn_ext` with mirrored K/V** (query split, attention replicated).

Measured on the 8-layer model, four sockets (single socket = **19.51** tok/s):

| arm | tok/s |
|---|---:|
| TP, attention mirrored | 17.65 |
| TP + `attn_output` split (unblocked by the new rule) | 17.24 |
| TP + fused reduce + spin dispatch | 16.86 |
| TP + merged reduce | 17.71 |
| TP + shared-expert split | 17.72 |

**Tensor-parallel is still slower than one socket for this model, even with the
whole scheduler stack.** Fused/merged reduce moved it 17.24 -> 17.71 (+2.7%), not
the +13% it was worth on GLM-5.3 Full. The reason is structural and now measured
rather than assumed: with attention mirrored, all four sockets redundantly stream
the same 4.4 GB/token of attention weights, and no scheduler tuning removes
redundant streaming. The lever was never the sync; it is the mirroring.

### The head-split, two blockers deeper

`GGML_GLM5N_ATTN_TP=1` advanced through two more real blockers:

1. `cannot infer split for op=CONCAT ... scalar_only=1 srcs=[node_26 axis=0,
   node_27 axis=0]`. GLM-5.3-Flash packs its KDA attention by concatenating the
   projections **along the split axis**: `node_28 = concat(Wq.x, Wk.x, dim 0)`,
   `node_30 = concat(node_28, Wv.x, dim 0)`. That is the fused-QKV case
   `ggml_backend_meta_split_state` documents its **segments** mechanism for, and
   it was unimplemented — concatenating along the split axis now appends the two
   operands' segments (implemented, 16-segment cap respected).
2. It now reaches `GGML_ASSERT(split_state.n_segments == 1)` in the
   source/consumer reconciliation loop, which does not handle a multi-segment
   split state on the consumer side.

**Stopped here deliberately.** Each fix advances one level, but the downstream
segment paths are unexercised for this shape, and pushing a subtle scheduler
without test coverage is exactly how the v6 profile on GLM-5.3 Full produced
fluent nonsense that passed a first-token oracle. The remaining work is bounded
and principled — teach the reconciliation loop multi-segment consumers — but it
needs a real correctness harness, not a throughput sweep.

### Safety note on the shipping configuration

The single-socket profile in section 7 (**3.68 tok/s**) was measured before this
merge but remains valid for the current binary: with `--gpu-layers 0` and no
`--device CPU-NUMA*` the meta backend is never instantiated, so everything in
this section is dead code on that path. Confirmed empirically — the 8-layer
single-socket rate is **19.54** after the merge versus 19.51 before.

## 14. Implemented the glm5next NextN graph — speculative decoding now works

`"glm5next NextN graph not implemented yet"` was a hard `GGML_ASSERT`, so the MTP
draft layer that ships inside every GLM-5.3-Flash checkpoint (`blk.45.nextn.*`)
could never be used. It is now implemented.

**Why this was the right thing to attempt rather than the tensor-split work:**
speculative decoding is exactness-preserving at temperature 0 — a drafted token is
only emitted if it matches what the trunk would have produced. A wrong NextN graph
therefore costs *acceptance*, never correctness. That is the opposite risk profile
from editing the tensor-split scheduler, where a mistake produces fluent nonsense.

**What the block is.** `blk.45` is a full DSA/MLA attention block plus the MoE FFN
with shared expert — but it has **no `hc_*` tensors**, so unlike every trunk layer
it is an ordinary pre-norm residual block, not a hyper-connected one. The graph
reuses the existing `build_dsa_layer` and `build_layer_ffn`, passing
`inp_kp = nullptr` so attention takes its dense path: not an approximation, since
the indexer only selects a subset of positions and dense attends to all of them.

**The bug that mattered.** The first working build drafted at **1.1% acceptance**.
The cause was not in the new graph at all: the glm5next *trunk* never published
`res->t_h_nextn`, the post-norm hidden state a NextN head consumes (glm-dsa sets it
right after `output_norm`). The draft head was predicting from nothing. Publishing
it took acceptance to **46.8%**, and then a second fix was needed — glm5next filters
`inp_out_ids` *before* its final collapse, so `t_h_nextn` came out `n_outputs` wide
while the MTP context reads `n_tokens` (`tensor read out of bounds`). Since both the
hc-collapse and the final norm are per-token, doing them before the filter is
equivalent; the filter now happens after.

### Correctness: proven, not asserted

`mtp_verify.sh` runs the same greedy prompt with and without speculation and
compares the output **byte for byte**. The truncated 8-layer model is ideal for
this: its output is gibberish, but *deterministic* gibberish, so any divergence is
a real bug.

| arm | 8-layer model | exactness |
|---|---:|---|
| no speculation | 19.56 tok/s | - |
| `draft-mtp` n_max=3 | 7.65 | **IDENTICAL** |
| `draft-mtp` n_max=2 | 9.02 | **IDENTICAL** |

(The truncated model cannot judge *acceptance* — a draft head trained on a
45-layer trunk's hidden state gets an 8-layer one, so ~1% there is expected and
not diagnostic. Acceptance is only meaningful on the full model.)

### Full model, four sockets, measured

| config | tok/s | acceptance |
|---|---:|---:|
| TP, no MTP (section 13) | 3.47 | - |
| **TP + MTP** | **3.74** | **50.0%** |
| single socket, no MTP (the previous best, section 6) | 3.68 | - |

**MTP under tensor-parallel already beats the previous best configuration**, while
running the topology that is otherwise *slower*. The lift is +7.8% over the TP
baseline rather than the ~2x a 50% acceptance rate suggests, and the reason is the
one the GLM-5.3 Full plan documented: on an MoE a verify batch of n+1 tokens routes
to up to 8(n+1) distinct experts, so verification costs far more than a plain decode
and the optimal draft depth is shallow.

**Not yet measured: single socket + MTP**, which is the configuration that should be
fastest of all (single socket is +6% over TP before speculation). It needs 95 GB on
one NUMA node, which does not fit while GLM-5.3 Full is resident. Run it in the next
quiet window with `GLM53F_SPEC=1` on `launch-glm53-flash-1s.sh` — the launcher
already supports it.

Also still open: the per-request `speculative.n_max` override is ignored on
`/completion` for this build (identical draft counts across n_max 0-4), so draft
depth has to be swept by restarting rather than per request.

## 15. Which GLM-5.3-Flash configuration is actually deployable

The two GLM models cannot both be single-socket resident, and that decides the
answer rather than raw tok/s does:

- GLM-5.3 Full tensor-parallel puts ~121 GB on each of the four 193 GB nodes,
  leaving ~72 GB free per node.
- GLM-5.3-Flash **single socket** needs its whole 95 GB on one node. 95 > 72, so
  it cannot run while GLM-5.3 Full is up. (And `--load-mode mmap` is not a way
  around it: mmap ignores `--membind`, which is the 1.79-vs-2.90 tok/s trap in
  section 7.)
- GLM-5.3-Flash **tensor-parallel** spreads the same 95 GB as ~24 GB per node,
  which fits alongside GLM-5.3 Full comfortably.

So there are two regimes, and the fastest measured configuration differs:

| regime | best measured config | tok/s |
|---|---|---:|
| **GLM-5.3 Full also resident** | TP (4 sockets) + MTP | **3.74** |
| Flash alone on the box | single socket + MTP | not yet measured (single socket is +6% over TP before speculation, so ~4.0 expected) |

This is why implementing the NextN graph mattered more than it first appears:
tensor-parallel was a *dead end for throughput* (section 13, 3.47 vs 3.68 single
socket), but it is the only topology that coexists with GLM-5.3 Full — and MTP is
what makes that topology finally beat the single-socket alternative it replaces.
Without speculation the coexisting configuration was strictly worse than running
Flash alone; with it, 3.74 > 3.68.

## 16. Single socket + MTP measured — and the draft depth is what decides it

Took a short, restorable outage on GLM-5.3 Full to measure the one configuration
that cannot coexist with it. The result overturned the estimate in section 15:

| single-socket config | tok/s | acceptance |
|---|---:|---:|
| no MTP (the section 6 best) | 3.68 | - |
| MTP, n_max=3 | 3.05-3.11 | 27.5% |
| **MTP, n_max=1** | **4.37** | **65.8%** |

**MTP at n_max=3 is a LOSS on single socket** (3.08 vs 3.68) and only becomes a
win when the draft is made shallow. This is the same MoE verify-cost effect the
GLM-5.3 Full sweep found in section 3.1 — a verify batch of n+1 tokens routes to
up to 8(n+1) distinct experts, so depth is expensive — except that here the
optimum is **n=1**, shallower still than GLM-5.3 Full's n=2, because this model's
routed experts are a larger share of a much cheaper token.

Note also that acceptance is not a constant of the model: it read 50.0% under
tensor-parallel with stock kernels, 27.5% single-socket at n=3 with the repack/x16
kernels, and 65.8% single-socket at n=1. Depth changes it (a deeper draft is
graded on harder later tokens) and so, slightly, do the repacked kernels, which
round differently and flip the draft head's top-1 on near-ties.

### Final state of GLM-5.3-Flash

| | tok/s | vs session start |
|---|---:|---:|
| session start (shipped launcher, 1 socket, mmap) | 2.90 | - |
| + fast kernel stack, flash-attn on | 3.68 | +27% |
| **+ NextN MTP at n_max=1** | **4.37** | **+51%** |
| `ngram-mod` on repetitive text (separate axis, stacks poorly with MTP) | 6.77 peak | |

Deployable-configuration guidance from section 15 still holds: with GLM-5.3 Full
resident, only the tensor-parallel profile fits, and there the best is TP + MTP at
3.74. Run `launch-glm53-flash-1s.sh` with `GLM53F_SPEC=1 GLM53F_SPEC_N_MAX=1` when
Flash has the box.

### Operational finding: /dev/shm scratch files stall the GLM-5.3 Full load

GLM-5.3 Full sat at 640 GB RSS for 34 minutes without becoming healthy, with 2 GB
free. It went healthy **within seconds** of deleting 48 GB of my truncated dev
models from `/dev/shm`. This is the documented "a loaded /dev/shm makes the load an
unrecoverable thrash" failure, and it is easy to cause by accident: the load peaks
~137 GB above its 483 GB steady state, so scratch files that look harmless at
20% tmpfs usage are enough to stall it. Keep `/dev/shm` clear of anything but the
production draft models while GLM-5.3 Full is loading.

### Caveat on the section 16 numbers, and a systemd trap

`systemctl --user stop glm53-sr950` did **not** keep GLM-5.3 Full down: the unit is
`Restart=on-failure` with a 30 s delay and it came back up ~3 minutes later on its
own. So the single-socket Flash measurements above were taken while GLM-5.3 Full
was concurrently loading, and the two then competed until **the OOM killer killed
GLM-5.3 Full** (`journalctl --user -u glm53-sr950`: "A process of this unit has
been killed by the OOM killer", 22:21 local). The unit was left `activating /
start-post` with no main process and needed `systemctl --user reset-failed` before
it would start again.

The direction of the error is favourable: 4.37 tok/s was measured **under
contention**, so it is a floor, not a ceiling. The ranking it establishes —
MTP n_max=1 (4.37) > no MTP (3.68) > MTP n_max=3 (3.08) — is unaffected, since the
n=3 arm was contended the same way. Re-measure on a genuinely idle box for a
publishable figure.

**To take the box for a measurement, mask the unit rather than stopping it:**
`systemctl --user stop glm53-sr950 && systemctl --user mask glm53-sr950`, then
`unmask` + `start` afterwards. And clear `/dev/shm` first (previous section).

## 17. Where the fleet stands

| model | at session start | best measured now | change |
|---|---:|---:|---:|
| **GLM-5.3 Full** UD-Q4_K_XL | 7.51 raw / 9.55 MTP / 13.4 agentic | 7.55 raw / 9.50 MTP / **15.2 agentic** | composite spec verified working; already near its 13.6 ceiling |
| **Qwen3.8-27B** UD-Q8_K_XL | 7.17 raw / 16.49 MTP (08-29 stack) | 7.47 raw / **15.10** MTP | +4% raw on the fast fork |
| **Qwen3.8-Flash-Next** UD-Q2_K_XL | 6.05-6.47 | **7.01**, 14.60 on repetitive | +16%, +141% on repetitive |
| **GLM-5.3-Flash** UD-IQ2_XXS | 2.90 | **4.37** | **+51%** |

Ceilings, for honesty about what is left: GLM-5.3 Full is bandwidth-capped at 13.6
raw and is essentially there once speculation is counted. Qwen3.8-27B at Q8 is
capped at 12.6 raw (a Q4_K_XL artifact would raise that to 22.5 and is untested).
The two Flash models are nowhere near their bandwidth ceilings (24 and 10.8 on one
socket) but are not bandwidth-bound — see sections 10-12.

### What was built, not just tuned

- **`glm5next` tensor-parallel**: taken off the arch denylist and made to work
  end-to-end (4 engine blockers fixed, plus 2 new meta-backend split rules and a
  segments implementation for concat-along-the-split-axis). It turned out not to
  be a throughput win, but it is the only topology that fits beside GLM-5.3 Full.
- **The `glm5next` NextN graph**: implemented from scratch; speculative decoding on
  GLM-5.3-Flash did not exist before and now runs at 65.8% acceptance.
- **`res->t_h_nextn` in the glm5next trunk**: the missing line that made the draft
  head predict from nothing (1.1% -> 46.8% acceptance).
- **The fast kernel stack** ported into two more engines.
- **A dev loop**: `make_trunc.py` (N-layer GGUF of any arch, 10-16 s loads),
  `mtp_verify.sh` (byte-exact speculation check), `profile_8L.sh`, `knob_sweep.sh`,
  `tp_sweep_8L.sh`. Four separate hypotheses were killed cheaply because of it.

### Ranked next steps

1. **Re-measure GLM-5.3-Flash single-socket + MTP n=1 on a genuinely idle box.**
   4.37 was taken under contention and is a floor. Mask the systemd unit first.
2. **Sweep `n_max` for Qwen3.8-27B and Flash-Next the same way.** The n=1 result on
   GLM-5.3-Flash suggests the house default of 3 is too deep for MoE models
   generally; Qwen3.8-27B is on n=3 today.
3. **`iq2_s_r8` / `iq2_xxs_r8` x16 kernels** — worth ~+15% on GLM-5.3-Flash
   (routed experts are 24.6% of its token), not the transformation once assumed.
4. **The KDA head-split**, blocked on multi-segment consumers in the meta
   backend's reconciliation loop. Needs a correctness harness, not a speed sweep.
5. **A Qwen3.8-27B UD-Q4_K_XL run** — 16.9 GB/token vs 30, ceiling 22.5 vs 12.6.

### One verification still owed

Implementing the NextN graph required a change to the glm5next **trunk** path, not
just the MTP path: `inp_out_ids` filtering moved from before the hc-collapse to
after the final norm (section 14). The two orderings are mathematically equivalent
because both the collapse and the norm are per-token, and the change is indirectly
evidenced — a draft head cannot agree with its trunk 65.8% of the time if the trunk
is wrong — but **a direct `17 * 23 -> 391` probe has not been run against the
current binary**, only against the pre-change one. Run
`probe.py 18096 glm53-flash` on the next single-socket start before treating this
engine as shipping-verified.

### A second, sharper way to stall the GLM-5.3 Full load: per-node page cache

After the `/dev/shm` fix, GLM-5.3 Full still got OOM-killed 3.5 minutes into a
load, on a machine reporting **705 GB available**. The kernel message names the
real constraint:

```
oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=2, ... global_oom
Out of memory: Killed process (llama-server) ... file-rss:456085504kB
```

`CONSTRAINT_MEMORY_POLICY / nodemask=2` is a **NUMA-node-constrained** OOM: the
tensor-parallel loader was allocating with a strict `mbind()` to node 2, and node 2
could not satisfy it. System-wide free memory is irrelevant to that allocation.
The 456 GB of `file-rss` is the cause — repeated single-socket model loads earlier
in the session had filled specific nodes' page cache, and the kernel would not
reclaim it fast enough against a strict policy.

**`free -g` cannot show you this.** Check `numactl -H | grep free` per node, and
before starting GLM-5.3 Full after heavy model-loading work:

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

That took every node from 2-27 GB free to 173-188 GB free and the load proceeded.
The cost is a cold re-read of the 436 GB model from the SATA SSD; the alternative
is an OOM kill 3 minutes in and a wedged systemd unit.

Together with the `/dev/shm` finding, the rule for this box is: **before loading
GLM-5.3 Full, clear /dev/shm of non-production files and drop the page cache.**

## 18. The TP-vs-single-socket comparison in this document was invalid

Every tensor-parallel measurement above (sections 4, 13, 15, 16) was taken with the
**stock kernels**, while every single-socket measurement was taken with the **fast
x16 AVX-512-VNNI kernels**. They were never comparable.

The cause: `make_gpu_buft_list` in `src/llama-model.cpp` offers a device's "extra"
buffer types — for a CPU-NUMA device that is the node-local **repack** buffer,
which is what selects the x16 kernels — but it asks the *backend registry* for
them. A Meta (tensor-parallel) device has no registry of its own; its extras have
to be composed from the per-device ones via
`ggml_backend_meta_device_get_extra_bufts`. That function came across with the
meta-backend merge in section 13, but **nothing ever called it**, so under TP the
model landed on the plain node buffer and ran stock kernels every time.

`GGML_CPU_NUMA_REPACK=1` was also never set in `launch-glm53-flash-tp.sh`.

Both are now fixed: the meta device offers its composed repack bufts before the
plain node buffer, and the TP launcher exports the full fast-kernel set
(`NUMA_REPACK`, `REPACK_LOAD_THREADS`, `FUSED_REDUCE`, `MERGE_REDUCE`, spin
dispatch, `ROUTER_F16`, the x16 and IQ-repack knobs, MoE gate/up fusion).

This retrospectively explains the anomaly in section 13 that I recorded but did not
chase: with attention mirrored, per-socket bytes are 6.6 GB against 8.5 GB for one
socket, so TP should have been ~1.29x *faster*, and it measured 0.91x. A ~30% gap
that "no scheduler tuning removes redundant streaming" does not account for — it
was the kernels, not the scheduling.

**Every TP number in this document should be treated as a lower bound pending
re-measurement**, including the conclusion in section 13 that tensor-parallel is a
dead end for GLM-5.3-Flash.

## 19. 20+ tok/s found — and it was the dense model, not the Flash ones

Chasing 20+ on GLM-5.3-Flash was the wrong target. Its token is 46 layers x ~145
ops with a six-stream hyper-connection residual, so it is op-bound: measured under
TP *with* the repack fix it gets **21% extraction, worse than the 33% it gets on a
single socket**, because tensor-parallel makes every one of those small ops 4x
smaller while adding sync. More sockets cannot help a model shaped like that.

**Qwen3.8-27B is the model this machine can actually drive**, because it is dense:
contiguous streaming, large matrices, near-perfect TP scaling, and 59% extraction
rather than 33%. The untested `UD-Q4_K_XL` artifact (17.6 GB vs 31.4 GB) was
sitting on disk the whole time.

| Qwen3.8-27B config | tok/s |
|---|---:|
| UD-Q8_K_XL raw (4-socket TP) | 7.47 |
| **UD-Q4_K_XL raw** | **10.39** (+39%) |
| UD-Q8_K_XL + MTP n=3 | 15.10 |
| UD-Q4_K_XL + MTP n=3 | 15.36 |
| **UD-Q4_K_XL + `ngram-mod,draft-mtp` composite, agentic replay** | **27.2 - 28.2** |

**28 tok/s at 89-93% draft acceptance**, verified reproducing a Python module
correctly. That is ~2.1 GB/token of effective work against 380 GB/s of DDR4.

Two things this taught that generalise:

- **Halving the trunk's bytes barely moved the MTP number** (15.10 -> 15.36) even
  though it moved raw decode 39%. Once speculation is on, the *draft* pass is the
  bottleneck, not the trunk: the in-model NextN block is a full transformer layer
  plus a 1.04 GB lm_head, so every drafted token costs a real forward pass.
- **That is also why depth loses**, on the dense model exactly as on the MoE ones:
  n3 14.96 > n4 14.26 > n6 13.12 > n8 11.79 > n12 10.65. `ngram-mod` wins because
  its drafts cost *nothing* per token — it drafts from repetition already in the
  context, which is precisely what agentic and code-editing traffic is made of.
  (`n_max=24` is not usable at all: `GGML_ASSERT(obj_new)` in ggml.c, the graph
  context overflows.)

Keep the composite on an agentic alias, not the general one: on prose/code/novel it
measures 14.6 vs 15.4 for plain MTP, matching the 08-29 finding that a short ngram
match regresses cold prose.

## 20. Qwen3.8-Flash-Next tensor-parallel: first correct run, and the exact remaining bug

Section 3.4 concluded "do not invest in qwen4exp tensor-split." That was wrong, and
the arithmetic says so plainly: Flash-Next reads **4.3 GB/token** against the 27B's
16.9. **Four sockets at today's unimproved 29% extraction is 25.6 tok/s.** Extraction
does not need to improve at all; TP just has to be correct.

### What now works

Porting the four meta-backend correctness fixes from the glm5next effort
(non-meta host tensors pass through the per-device mapper and report MIRRORED;
concat along the split axis produces segments; mul_mat with a k-split weight against
a mirrored input yields PARTIAL; multi-segment handler results are authoritative)
plus a staged split policy (`GGML_Q4E_SPLIT`) gives, for the first time:

| `GGML_Q4E_SPLIT` | what is split | output | tok/s |
|---|---|---|---:|
| **0** | routed experts + dense FFN, all else mirrored | **CORRECT** — "391." and coherent | 5.52 |
| 2 | generic, state mirrored | aborts in `handle_gated_delta_net` | - |
| 3 | generic, whole SSM half mirrored | aborts, `SPLIT_AXIS_UNKNOWN` | - |
| 1 | all generic rules | fluent garbage | 8.32 |

Two things this establishes:
- **qwen4exp can run tensor-parallel and produce correct output.** Previously every
  configuration produced garbage. Stage 0 is a correct baseline to bisect against.
- **The speed is there.** Stage 1 runs at 8.32 tok/s — already +20% over the best
  single-socket result (6.95) — with only part of the model split, no repack, no
  fast kernels and no thread tuning. What it lacks is correctness.

Also fixed on the way: the meta device must only be offered node-local repack
buffers when repacking is requested (`GGML_CPU_NUMA_REPACK`), because the repack
buffer's `set_tensor` asserts `size == ggml_nbytes(tensor)` and a TP load writes
per-device slices. Preferring it unconditionally turns every TP load into an abort.

### The remaining bug, and what it is NOT

Stages 2 and 3 aborting proves the recurrent state and the SSM weights must be split
*together* — `handle_gated_delta_net` requires src[5] (state) to be split whenever
srcs 0-4 are. The generic rules do split them together, so that is not it.

Checked and **ruled out**: the segment arithmetic. For qwen4exp
`ssm_d_state=128, ssm_n_group=16, ssm_dt_rank=48` gives `key_dim=2048`,
`value_dim=6144`, `head_ratio=3`; `attn_qkv` is `[2560, 10240]` and the rule's
`GGML_ASSERT(ne[axis] == 2*key_dim + value_dim)` = 10240 holds, `cache_s` segments
to `16*128*128 x 3 = 786432` and `cache_r` to `2048*3 x 5 = 30720`, both matching
the tensors exactly.

Also **ruled out**: the broadcast grouping. qwen4exp is lumped with Qwen 3.5 rather
than Qwen 3 Next, and those interleave K/V differently
(`[k0_v0, k1_v1, k0_v2, k1_v3]` vs `[k0_v0, k0_v1, k1_v2, k1_v3]`) — a wrong choice
there would produce exactly this symptom. Added `GGML_Q4E_BCAST=next` to test the
other grouping: **still garbage**, so it is neither.

### The next step is a tool, not another guess

Every cheap hypothesis is exhausted. The right instrument is the one the GLM-5.3
Full effort used to find its strided-row bug: run `llama-eval-callback` on a short
prompt under one socket and under TP, and diff per-op tensors to find the **first**
op whose outputs disagree. That converts this from guess-and-check into a bisection
over ~1400 ops. Everything needed is in place — a correct stage-0 baseline to diff
against, and a staged knob to re-enable rules one at a time.

Patches: `patches/q4e-llama-model.cpp.patch`, `patches/q4e-ggml-backend-meta.cpp.patch`.
The tree is `/dev/shm/q4e-fast` (tmpfs — rebuild from the patches after a reboot).

## 21. GLM-5.3-Flash: the KDA head-split works, and it settles the question

The last structural lever was the KDA head-split — splitting the 34 linear-attention
layers by head instead of mirroring them. It is now **implemented and correct**.

The earlier attempt aborted at `SSM_CONV ... srcs=[conv_input-0 axis=1, node_48
axis=10]`: the conv weight was mirrored while its input was head-split. The KDA
layers carry the 64 x 128 head layout on **four different axes**, and only one was
being split:

| tensor | shape | head axis |
|---|---|---|
| `attn_q/k/v`, `ssm_f_b`, `ssm_g_b` | [4096, 8192] / [128, 8192] | 1 |
| `ssm_conv1d_{q,k,v}` | [4, 1, **8192**] | **2** |
| `ssm_dt.bias` | [**8192**] | **0** |
| `ssm_beta` [4096, **64**], `ssm_a` [**64**] | per-HEAD, not per-channel | 1 / 0, granularity 1 |

With all four split (plus whole-head granularity), GLM-5.3-Flash runs on four
sockets and answers `17*23 -> 391` correctly.

### And it produced no speedup, which is the answer

| configuration | tok/s | extraction |
|---|---:|---:|
| single socket, no speculation | 3.68 | 33% |
| **single socket + MTP n=1** | **4.37** | 39% |
| 4 sockets, attention mirrored | 2.99 | 21% |
| 4 sockets, KDA head-split (correct) | 3.89 | 16% |
| 4 sockets, head-split + shexp + fast kernels | 4.24 | 16% |

**Four sockets deliver 4x the memory bandwidth and produce 4.24 against one
socket's 4.37 — zero gain.** That is not a tuning gap, it is proof that bandwidth
is not this model's constraint. Every byte-side lever now works correctly and none
of them matter.

### Why 20 tok/s is not reachable for this model on this box

20 tok/s is a 50 ms token. GLM-5.3-Flash runs **~6,700 ops per token** (46 layers x
~145), so the budget is **7.5 us per op** — and a 32-thread fork/join barrier alone
costs 2-5 us. Tensor-parallel makes this strictly worse: it divides each already-tiny
op by four and adds cross-socket synchronisation, which is exactly what the table
above shows. The op count comes from the architecture: 46 layers, a six-stream
hyper-connection residual (so every norm/concat/add runs six times), a lightning
indexer on the MLA half, and a KDA recurrent state on the other.

The honest ceiling for GLM-5.3-Flash here is **~4.4 tok/s**, up from 2.90 at the
start of this work (+51%). The bandwidth ceiling of 10.8 tok/s on one socket is
academic — the model cannot get near it.

### What would actually move it

Not scheduling, not kernels, not NUMA — **op count**. Fusing the six
hyper-connection streams into one wide op and fusing the KDA elementwise chain
would cut ops per layer several-fold. That is surgery on the glm5next graph
builder in `src/models/glm5next.cpp`, with an uncertain payoff, and it is the only
remaining avenue that addresses the real constraint.

**The 20+ tok/s model on this machine is Qwen3.8-27B** (section 19): 27.2-28.2 tok/s
on agentic traffic, because it is dense — big contiguous matrices, few large ops,
59% extraction, and TP that actually scales.

## 22. Qwen3.8-Flash-Next tensor-parallel: bisection complete, one bug named

Section 20 left the bisection half-done. It is now finished. Every stage below is
4-socket TP on `/dev/shm/q4e-fast`; the only variable is which tensor families are
split (`GGML_Q4E_SPLIT`), everything else mirrored, which is always numerically safe.

| stage | split | output | tok/s |
|---|---|---|---:|
| 0 | routed experts + dense FFN | **CORRECT** | 5.52 |
| 12 | + lm_head | **CORRECT** | 5.69 |
| 11 | + the 36 SSM/KDA layers and their recurrent state | **CORRECT** | 7.65 (full bench) |
| **14** | **experts + SSM + lm_head (best correct)** | **CORRECT** | **7.51** |
| 10 | + the 12 full-attention layers and the KV cache | garbage | 5.40 |
| 15 | + GQA-correct attention (q and attn_output only) | garbage | 8.22 |
| 13 | all named families | garbage | 8.19 |
| 1 | all generic rules | garbage | 8.32 |

**The SSM/KDA half splits correctly and is where the win is** — stage 11 alone takes
it from 5.52 to 7.65, past the best single-socket result (7.01). **The sole remaining
bug is the 12 full-attention layers.**

Two things about those layers, both established here:
- They are **GQA with 24 q heads but only TWO kv heads** (`attn_k/v` are
  `[2560, 512]` = 2 x 256). Two heads cannot be divided across four sockets, so the
  generic head-split is unsatisfiable by construction. That explains why stage 10
  is garbage.
- But **the textbook GQA fix is not sufficient**: stage 15 mirrors k/v and the KV
  cache and splits only q (24 heads -> 6 per socket) and `attn_output`, and it is
  *still* garbage. So something else in that path is also wrong. Note
  `attn_q` is `[2560, 12288]` while `attn_output` consumes `6144` — q is twice the
  width attn_output expects, so the q layout is not a plain 24 x 256 and the head
  granularity the generic rule computes is very likely wrong for it. That is the
  next thing to check.

Correctness discipline that made this tractable: mirroring is always numerically
safe, so a staged knob that mirrors everything except one named family turns "which
rule is wrong" into a bisection over four families instead of a hunt through ~1400
ops. It found the answer in four runs after `llama-eval-callback` proved unusable
(it segfaults under the meta backend, dereferencing meta placeholder addresses).

Also fixed here: the earlier stage-10 abort on
`split_states_equal(src_ss[0], src_ss[2])` was my own staging artifact — splitting
attention while mirroring the KV cache. The cache must follow attention.

### Measurement hygiene note (09-04 evening)

The first attempt at a final Flash-Next number returned `mean 3.584, range
0.198-6.97` — a 35x spread, which is not a result. Cause: a GLM-5.3-Flash server
was running at **1264% CPU** for the whole bench (a concurrent GLM-5.3-Flash TP+MTP
test on port 18096), plus dockerd/containerd at ~64% each. The probe taken on the
same server minutes earlier read 7.23.

Two rules this reinforces, both already in the GLM-5.3 Full plan and both worth
obeying literally:
- **Check `ps --sort=-%cpu` immediately before trusting any tok/s number.** A 35x
  range is the signature; a plausible-looking mean with a wide range is the danger.
- Kill by **port**, not by process name, and verify afterwards — several sweeps in
  this session left servers alive because the cleanup pattern did not match, and one
  of them then silently contaminated a later measurement for two hours.
