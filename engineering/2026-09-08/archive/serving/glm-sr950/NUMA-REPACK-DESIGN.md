# NUMA-local CPU_REPACK — removing the placement/kernel tradeoff

Design and production record, 2026-08-25 through 2026-08-26. Companion to
`KERNEL-CALIBRATION.md` and `IQ2-REPACK-DESIGN.md`.

The per-device repack path is implemented, correctness-tested, PGO-trained,
and enabled in production with `GGML_CPU_NUMA_REPACK=1`. The final 256K process
placed anonymous memory at 96.98/96.97/97.05/97.16 GiB across NUMA nodes 0-3,
within 0.2%, while holding zero process swap. This closes the original
placement-versus-kernel tradeoff for supported tensor-split architectures.

## The historical constraint this removes

On this four-socket box you currently must choose ONE of:

| choice | memory | kernel | measured consequence |
| --- | ---: | ---: | --- |
| `--device CPU-NUMA0..3 --split-mode tensor` | **365 GB/s** local, balanced within ~1% | LUT `vec_dot` | GLM replay = **6.551 tok/s** |
| plain CPU + `CPU_REPACK` | interleaved | **VNNI 8x8 GEMM** | controlled GLM replay = **2.001 tok/s** |

They are mutually exclusive in the unmodified path. The controlled GLM arm had
reasonable 69.5/70.8/71.5/83.7 GB placement and still lost 69.5%, so the global
backend's cross-socket execution and synchronization are material in addition
to page placement. The earlier 1.7 tok/s V4-Flash arm had extreme skew and is
not a valid model-performance number.

## Why they are exclusive — it is an artifact, not a design

`ggml/src/ggml-cpu/repack.cpp:4821`

    ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {
        static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_repack = {
            ...
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
            /* .context = */ new ggml::cpu::repack::extra_buffer_type(),
        };
        return &ggml_backend_cpu_buffer_type_repack;
    }

One **static singleton, hard-bound to CPU device 0**. The scheduler requires a
tensor's `buft->device` to match the backend executing the op, so any tensor
assigned to `CPU-NUMA1/2/3` can never select it. Repack predates this fork's
multi-CPU-device NUMA backend; nothing reconciles them.

And its allocation is NUMA-oblivious (`repack.cpp:4751`):

    static ggml_backend_buffer_t ggml_backend_cpu_repack_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buft, size_t size) {
        ggml_backend_buffer_t buffer =
            ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

Generic host allocation, first-touch placement — so the repacked destination
lands wherever the loading thread happened to run.

Meanwhile `ggml/src/ggml-cpu/ggml-cpu.cpp:322` already provides
`ggml_backend_cpu_numa_buft_context` / `ggml_backend_cpu_numa_buffer_context`
with node-bound alloc/free/set_tensor. **Both halves exist. Neither calls the
other.**

## The change

The implementation replaces the singleton-only design with a reusable wrapper
factory:

1. `ggml_backend_cpu_repack_buffer_type_from(base_buft, device, name)` caches a
   stable repack buffer type for each base-buffer/device pair.
2. Its allocator delegates to `base_buft`. For `CPU-NUMA<N>` this is the
   existing node-bound `mbind` allocator; for ordinary CPU it preserves the
   original generic allocation behavior.
3. Each NUMA device exposes only its own wrapper when
   `GGML_CPU_NUMA_REPACK=1`; with the flag absent, production selection is
   unchanged.
4. CPU device buffer ordering prefers the node-local wrapper before the normal
   node buffer, but only for devices that actually expose one.
5. Repack operation support admits CPU-device activation buffers and requires
   the repacked weight's device to match the executing backend.

No change to `ggml_repack_get_optimal_repack_type()`, the traits, the kernels,
or the `MUL_MAT_ID` branch. This is plumbing, not new math.

## Measured result

Each socket owns one quarter of the supported tensor, repacked into its own
node-local memory, and runs the VNNI path against it. The compact IQ2/IQ3 PGO
engine measured 3.7814 tok/s deterministic raw decode at 32K and 7.0787 tok/s
on the confirmed 32K agentic replay suite. The 256K production replay checks
measured 6.586-6.645 tok/s. A subsequent 256K raw suite overlapped an
uncoordinated second model and was discarded.

The matched PGO versus non-PGO compact-repack raw comparison was +3.85-3.86%.
Direct validation measured maximum absolute output differences of 1.56e-7 for
IQ2_XS and 2.01e-7 for IQ3_XXS. These are the production results; earlier
bandwidth-derived 7/15/20 tok/s figures were hypotheses, not measurements.

## Prerequisite for V4-Flash specifically

`LLAMA_SPLIT_MODE_TENSOR is not implemented for architecture 'deepseek4'`, so
V4-Flash still needs architecture support before it can use per-node repack.
Tensor split is not GLM-only: `qwen35` was subsequently loaded and validated on
`CPU-NUMA0..3`, improving Qwen3.8-27B Q4_0 no-spec decode from 4.33 to 7.58
tok/s. Architecture support must therefore be tested per model family.

## Related non-GLM observations

For a plain global CPU backend, pair `numactl --interleave=all` with
`--numa numactl`; do not combine an external interleave policy with
`--numa distribute`, which installs a competing policy. This rule does not
replace tensor split: Qwen3.8 demonstrated 7.58 tok/s with socket-local shards
versus 4.33 tok/s interleaved on the same Q4_0 model.

DeepSeek-V4-Flash remains a separate unresolved kernel/architecture target.
Its measured decode was 1.88 tok/s across four sockets and 1.80 tok/s on one
socket; neither result supports the earlier 15-20 tok/s estimate. Do not quote
a bandwidth projection as application throughput.
