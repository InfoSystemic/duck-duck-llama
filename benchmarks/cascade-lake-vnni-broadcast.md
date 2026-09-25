# Cascade Lake: an embedded-broadcast VNNI instruction issues at about half rate

**Measured 2026-09-23** on one core of a Xeon Gold 6242 (Cascade Lake, two AVX-512 FMA units per core). The core's hyperthread sibling was held idle. The kit and its instructions are in [tools/ubench/](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/ubench/README.md).

## The finding

| Instruction | Issue rate per cycle |
| --- | ---: |
| `vpdpbusd zmm, zmm, zmm` (register operands) | ~1.85 |
| `vpdpbusd zmm, zmm, m32{1to16}` (embedded memory broadcast) | ~1.0–1.35 |
| `vpbroadcastd` once, then 2 × `vpdpbusd` from the register | ~1.9 per dot product |
| FMA or VNNI, 24 register accumulators, clock measured under load | ~1.6–1.68 (use this as the practical peak) |

The embedded-broadcast form is what a compiler emits for `_mm512_dpbusd_epi32(acc, w, _mm512_set1_epi32(*p))`, or a separate `vpbroadcastd` per dot product. Either way, every dot product costs a load.

llvm-mca (`-mcpu=cascadelake`) does **not** model this penalty. Trust a measured peak over its estimate.

Two things were ruled out as the cause. A 3.8 KB unrolled loop of broadcast plus two dot products runs as fast as a 114-byte one (1.9 per cycle), so instruction delivery is not the limit. And the AVX-512 heavy clock measured 3.06 GHz on a single core and about 2.7 GHz with all cores busy.

## What it changed

The x16 integer kernels that run MiMo's prompt batches now process 16-row weight groups in pairs. Each activation dword is broadcast into a register once and feeds two register-form dot products. Accumulators start at a precomputed per-block bias rather than zero plus a subtraction.

The float operations are unchanged per output element, so the result is bit-identical to the single-group kernel. That was checked at every width from 1 to 12 columns, on 16-, 48- and 64-row tiles.

Single core, sibling held, old and new libraries interleaved, minimum times:

| Shape | Before | After |
| --- | ---: | ---: |
| Dense Q8_0, 512 columns | 40.3 ms | 32.0 ms (1.26×) |
| MoE gate, MXFP4, 11 tokens per expert | 44.3 ms | 36.5 ms (1.21×) |
| MoE down | 45.6 ms | 42.5 ms (1.07×) |

At decode widths, the paired kernels bring the 2–4-token experts common in 8-row verify steps about 20% closer to the memory wall. For the MoE gate at 2 / 4 tokens per expert, bandwidth went from 76 / 66 to 91 / 81 GB/s.

[Patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/cpu-x16-gemm-paired-groups.patch) · [kernel prototypes and the variants that lost](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/ubench/x16-kernel-proto.cpp).

## Also measured

- **Register pressure.** 12 columns per call is a cliff for the Q8_0 kernel: register spills drop it from 3,384 to 2,152 GOPS. MXFP4 at 11 columns is fine. Calls are therefore split evenly, never as 8 plus a thin tail.
- **Losing variants.** Three groups per pass was no better than two. Activation blocks carrying their own bias and scale (40-byte blocks) ran out of general-purpose registers and were 1.26× *slower*.
- **Where the paired kernel sits now.** It reaches about 0.8 dot products per cycle with weights streaming from L2, and 1.04 with them resident in L1. For comparison, llvm-mca estimates 1.23 and the core can issue 1.85. What remains is the L2 weight stream and the per-block epilogue. Software prefetch of activations (distances 2–16) and of weights (1–8) gained nothing.
- **Bit-exactness and FMA contraction.** GCC's default `-ffp-contract=fast` fuses a multiply and a following add into one FMA, even across intrinsics. Where bit-exactness matters, put an empty `asm` barrier on every product, including the first. That is how the exact MoE weighted-sum fusion was kept exact.

## Method on a shared machine

- **Clean core.** A nice-0 `pause` spinner pinned to the benchmark core's hyperthread sibling ([spin.c](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/ubench/spin.c)) gives a clean core without stopping other jobs. Stop it with `pkill -x spin`, not a pattern match: a pattern also matches the shell that ran the command.
- **Interleaving.** Frequency drifts with the socket's load, so compare only interleaved runs, and report minimums.

## Does it transfer?

It is likely to apply to any AVX-512 VNNI inner loop on Cascade Lake with two FMA units. The x16 kernels behind the GLM, Qwen and DeepSeek results here have not been re-checked for it.

On a Xeon Silver (one FMA unit per core), the register form can issue only about once per cycle anyway, so the penalty may vanish. That has not been measured; see [independent verification](../docs/independent-verification.md).
