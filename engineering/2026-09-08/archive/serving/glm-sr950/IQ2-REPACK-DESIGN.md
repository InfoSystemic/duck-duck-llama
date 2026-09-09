# Exact load-time repack for IQ2_XS / IQ3_XXS

Design and production record, 2026-08-25 through 2026-08-26. Companion to
`KERNEL-CALIBRATION.md`. The compact IQ2_XS and IQ3_XXS paths described here
are implemented, correctness-tested, PGO-trained, and enabled in production.

## Implemented result

The final formats preserve the source scale metadata and signed value alphabet,
then interleave three IQ2 or four IQ3 code bitplanes for AVX-512 VNNI. They are
transient RAM formats only; the GGUF files on disk are unchanged.

| Gate | Production value |
| --- | --- |
| `GGML_CPU_NUMA_REPACK` | `1` |
| `GGML_CPU_IQ2_XS_REPACK` | `1` |
| `GGML_CPU_IQ3_XXS_REPACK` | `1` |
| `GGML_CPU_REPACK_LOAD_THREADS` | `8` |

Direct target-forward validation measured maximum absolute error of 1.56e-7
for IQ2_XS and 2.01e-7 for IQ3_XXS. The representation is exact; those values
come from a different floating-point accumulation order. The fused IQ2_XS
target path improved 1.640x in its matched kernel test.

The accepted PGO corpus uses the real tensor-split expert shape `[6144 x 512]`,
16 threads per socket, 1/2/4/8/16-token cases, and four-device Meta-backend
samples. It was collected with `-fprofile-update=atomic`; an earlier corpus
trained on undersized shapes was rejected after it regressed high-batch IQ3.
The corrected PGO build showed no tested kernel regression and improved the
matched deterministic whole-service raw suite by 3.85-3.86%.

## Why this is worth doing

Measured at real GLM-5.2 expert shape `[6144 x 2048]`, M=1, 16 threads, one
socket (95.3 GB/s available):

| type | bpw | ms | effective GB/s |
| --- | ---: | ---: | ---: |
| IQ2_XS (LUT path) | 2.31 | 0.504 | 7.2 |
| Q2_K repacked (VNNI GEMM) | 2.62 | **0.178** | **23.2** |

**2.83x at a modestly larger footprint.** IQ2_XS + IQ3_XXS are 94.4% of the model
(137.7 GB + 89.9 GB of 253.7 GB) and neither can enter the repack path:
`ggml_repack_get_optimal_repack_type()` returns null, `extra` stays null, and
`get_tensor_traits` dereferences it — placing such a tensor in `CPU_REPACK`
segfaults.

## The enabling result: these grids are tiny

Parsed from `ggml/src/ggml-common.h`:

| grid | entries | distinct byte magnitudes | values with sign | bits for exact |
| --- | ---: | --- | ---: | ---: |
| `iq2xxs_grid` | 256 | `{8, 25, 43}` | 6 | 3 |
| **`iq2xs_grid`** | 512 | **`{8, 25, 43}`** | **6** | **3** |
| **`iq3xxs_grid`** | 256 | `{4,12,20,28,36,44,52,62}` | 16 | **4** |
| `iq3s_grid` | 512 | `{1,3,5,7,9,11,13,15}` | 16 | 4 |

The 512 grid entries are *combinations* of 8 bytes, but the alphabet of byte
values is only three for IQ2_XS. A 3-bit IQ2 code or 4-bit IQ3 code plus a
16-entry value table therefore reproduces both formats **exactly**. This is a
lossless re-encoding, not a requantization — there is no quality question to
litigate.

## Format: `IQ2_XS_R8` / `IQ3_XXS_R8` (repack-internal, never on disk)

Keep the original scale metadata and losslessly replace only the grid/sign
encoding. Interleave eight rows in the same spirit as `q4_0_8x8`.

Amortized over each original 256-weight row block:

    IQ2_XS_R8:
      d         : 2 B    original fp16 global scale
      subscales : 8 B    16 original 4-bit scales, one per 16 weights
      codes     : 96 B   256 x 3-bit signed-alphabet indices
      total     : 106 B = 3.3125 bpw

    IQ3_XXS_R8:
      d         : 2 B    original fp16 global scale
      subscales : 4 B    8 original 4-bit scales, one per 32 weights
      codes     : 128 B  256 x 4-bit signed-alphabet indices
      total     : 134 B = 4.1875 bpw

This scale granularity matters: IQ2_XS stores two independent scale nibbles in
each 32-weight group, selected by `l/2` in `dequantize_row_iq2_xs`. Folding one
fp16 scale per 32 would both discard one scale and introduce an extra fp16
rounding. Preserving `d` and the original nibbles is smaller and exact.

Sign bits fold into the code: the table holds the signed values
`{-43,-25,-8,+8,+25,+43}` for IQ2_XS, so sign application disappears entirely
from the inner loop.

## Inner loop: bitplane expansion replaces a 512-entry gather

This is the whole point. Current IQ2_XS `vec_dot` per 32 weights does 4 uint16
index extractions, 4 loads from the 512-entry `iq2xs_grid` (memory-resident,
serialising), then sign application from a 7-bit field.

Replacement, per 64 IQ2 weights:

    codes  = movm(bitplane0) & 1;
    codes |= movm(bitplane1) & 2;
    codes |= movm(bitplane2) & 4;
    values = _mm512_shuffle_epi8(vtable, codes);
    acc    = _mm512_dpbusd_epi32(acc, values, activations);

`vtable` is a broadcast 16-byte register. The three mask expansions and one
in-register `vpshufb` replace the grid loads and sign application while 8-row
interleaving amortises the Q8 activation load.

Apply the original integer subscale before the final floating-point block
scale: IQ2_XS uses two `(2*ls + 1)` factors per 32 weights and a final `0.125`,
while IQ3_XXS uses one `(2*ls + 1)` factor per 32 and a final `0.25`. This keeps
the representation and scale arithmetic identical to the existing kernels.

`dpbusd` needs an unsigned first operand; use the standard llama.cpp trick of
biasing the weight table by +64 into `{21,39,56,72,89,107}` and subtracting
`64 * sum(activations)` once per block, or keep values unsigned and carry the
sign in the accumulator sign-correction already used by `q4_0_8x8`.

## Implementation sketch

1. `ggml/src/ggml-cpu/repack.cpp` — add
   `tensor_traits<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K>` and the IQ3_XXS
   counterpart; register both in `ggml_repack_get_optimal_repack_type()`.
   The existing `MUL_MAT_ID` branch (`ggml_n_dims(op->src[0]) == 3`) already
   admits MoE experts, so no scheduler change is needed.
2. `repack()` — copy the original global scale and subscale nibbles, decode
   grid + signs once, and emit 3-bit IQ2 or 4-bit IQ3 code bitplanes in
   8-row-interleaved order. Runs at load, once, alongside the existing repack
   pass.
3. `gemv`/`gemm` — expand the bitplanes and feed VNNI as above. Both decode
   and multi-row entry points are implemented.
4. Validation gate — env flag, off by default. The repacker must round-trip
   every signed alphabet value and `memcmp` the copied scale metadata. Compare
   kernel outputs against the existing kernels with a tight numeric tolerance:
   the representation is exact, but a different SIMD reduction order need not
   produce bit-identical floating-point sums.

## Memory budget

| portion | GGUF | after exact compact repack |
| --- | ---: | ---: |
| IQ2_XS | 128.34 GiB | 183.84 GiB (3.3125 bpw) |
| IQ3_XXS | 83.84 GiB | 114.63 GiB (4.1875 bpw) |
| everything else | 24.25 GiB | 24.25 GiB |
| **compact-IQ projection** | **236.43 GiB** | **322.73 GiB** |

The enabled expanded Q5_K repack raises the full runtime weight projection to
327.14 GiB. Both projections exclude KV cache and server workspaces. This fits
755 GiB at the selected 256K context with safe headroom. Costs RAM only — no
disk, no requant artifact, and no download.

## Measured whole-service result and ceiling

At 32K context, the accepted PGO binary measured 3.7814 tok/s deterministic
raw decode and 7.0787 tok/s on the confirmed agentic replay run. At the selected
256K production context it measured 6.586-6.645 tok/s on confirmed/measured
replay runs. A later 256K raw run was discarded because an uncoordinated second
model overlapped it. The smaller context is the speed tier; 256K is the
production balance between speed, useful agent context, and recovery headroom.

The implementation is a material gain but does not make GLM-5.2 a measured
12 tok/s model. Active expert bytes, the remaining non-repacked kernels, and
target-forward verification still dominate. Keep modeled bandwidth ceilings
separate from observed application throughput.

## NUMA allocation prerequisite: resolved

Production's tensor-sharded weights previously could not select the singleton
CPU repack buffer. The fork now exposes a per-device repack wrapper whose
allocation delegates to the corresponding CPU-NUMA buffer type. Repacked
weights therefore remain node-local and the Meta backend can execute the four
socket shards without surrendering tensor placement. See
`NUMA-REPACK-DESIGN.md` for the plumbing and validation record.
