# Reproducible llama.cpp patch bundles

For the current source map, start with the [reproduction guide](../docs/reproducing.md) and [September 12 complete engine bundles](../engineering/2026-09-12/source-bundles.json). The material below documents earlier source lines and their integration history.

The [September 8 engineering snapshot](../engineering/2026-09-08/README.md) adds eight complete source snapshots and two selected source overlays. Use its pinned-base table for current GLM Full/Flash and Qwen work. The patches below remain historical artifacts; in particular, the old Qwen tensor-split exclusion describes an earlier correctness failure that the later goal engine repairs. Do not stack that exclusion onto the current Qwen source bundle.

Two additional Full snapshots are preserved here: `llama.cpp-a302733-glm-sr950-20260901-session-start.patch` and `llama.cpp-a302733-glm-sr950-20260902-fast-stack.patch`. Both are independent deltas against `a30273376ef669023334fc20ad02ae4ed8196a65`, not patches to stack together. The current Full delta is in the September 8 bundle.

These patches publish the exact source deltas used for the measured CPU and
NUMA work in this repository. Each bundle is pinned to one upstream commit so
it can be audited, reproduced, and rebased deliberately.

| Patch | Pinned llama.cpp base | Scope |
| --- | --- | --- |
| [`llama.cpp-b249-cpu-numa.patch`](llama.cpp-b249-cpu-numa.patch) | `3173a56471c1753650cd806694145ffd6dcace67` (build 249) | Linux per-socket CPU devices, asynchronous execution, Meta fixes, direct F32 collective |
| [`llama.cpp-b249-qwen4exp-mtp.patch`](llama.cpp-b249-qwen4exp-mtp.patch) | `3173a56471c1753650cd806694145ffd6dcace67` (build 249) | Qwen3.8/Qwen4 experimental conversion, architecture, and detached MTP-sidecar support |
| [`llama.cpp-a302733-glm-sr950.patch`](llama.cpp-a302733-glm-sr950.patch) | `a30273376ef669023334fc20ad02ae4ed8196a65` | Integrated CPU-NUMA, GLM tensor-parallel/speculative support, compact-quant and x86 kernel work, tests |
| [`numa-reduce-nontemporal-stores.patch`](numa-reduce-nontemporal-stores.patch) | q4e goal engine (2026-09-04 line) | Non-temporal stores for the three remote copies written by the cross-socket all-reduce. Bit-identical; +5-10% decode and +11-13% prefill on Qwen3.8-Flash-Next, +2-3% on GLM-5.3-Flash |
| [`kv-seq-rm-bounded-scan.patch`](kv-seq-rm-bounded-scan.patch) | q4e goal engine (2026-09-04 line) | `llama_kv_cache::seq_rm` scans the used prefix instead of all cells; the full scan ran once per slot per speculative rollback at 256K context |
| [GLM-5.3-Flash decode series, 09-18 to 09-20](README-20260920-glm5next.md) | glm5n goal engine (2026-09-04 line) over the published 09-11 CPU sources | Thirteen changes in twelve patch files, all in production on 2026-09-20: fused/wider pooled-indexer kernels, flat state copy, pooled-result cache, batch-invariant cell-split MQA attention, selection top-k with tie fallback, gated delta net by state row, one worker team per NUMA device, bounded KV rollback, cache-only MTP catch-up, MTP query rows, MTP draft loop (merge, direct pick, constant-shape batches), coupled draft/verifier sampling, sampler host cost. All but the attention kernel are output-identical; the attention kernel moves output toward float64 because the stock kernel sums V in FP16 |
| [`upstream-cpu-fattn-f32-accumulate.patch`](upstream-cpu-fattn-f32-accumulate.patch) | ggml-org/llama.cpp `b23efaa2ef147f547ee75cbf0c621d61904de80e` | Candidate only, not submitted. CPU flash attention sums V in FP16 with an F16 cache; F32 accumulation cuts the error 75x for +17-21% in the op. See [upstream candidates](../docs/upstream-candidates.md) |
| [`topk-linear-selection.patch`](topk-linear-selection.patch) | q4e goal engine (2026-09-04 line) | **REFUTED IN PRACTICE 2026-09-15, do not promote.** `GGML_Q4E_TOPK=2` makes the selection ~4-7% cheaper per cycle but drops draft acceptance from 89% to 64%, for 7-17% WORSE net throughput. Masked (-INFINITY) scores make the tie tail arbitrary, so the selected key set differs from `std::partial_sort` and the model attends elsewhere. The microbenchmark missed it by feeding pure normals. Retained as a record of the measurement. |
| [`qwen4exp-ssm-mirror-type-independent.patch`](qwen4exp-ssm-mirror-type-independent.patch) | q4e goal engine (2026-09-04 line) | **DISPROVEN 2026-09-14, do not enable.** `GGML_Q4E_SSM_MIRROR=1` was intended to make the tensor-split rule independent of quantisation type, but it breaks a working model: the control arm running the unmodified Q6_K_XL production model with the knob on also failed to load. `handle_gated_delta_net` accepts only all-MIRRORED or all-SPLIT_AXIS_1, so mirroring two projections creates an invalid mix. Retained as a record of the attempt and its refutation |

The two build-249 patches may be applied together. The GLM integration is a
separate source line and must not be stacked on them. Verify downloads against
[`SHA256SUMS`](SHA256SUMS), and see [`ATTRIBUTION.md`](ATTRIBUTION.md) for
provenance.

**Safety fix — apply this one.**
[`qwen4exp-exclude-from-tensor-split.patch`](qwen4exp-exclude-from-tensor-split.patch)
adds `LLM_ARCH_QWEN4EXP` to `llm_arch_supports_sm_tensor()`'s exclusion list.
Without it, `--split-mode tensor` on Qwen3.8-Flash-Next produces **silently wrong
output** — fluent nonsense, no error — while running ~45% faster than the correct
path, so throughput tuning selects for it. With it, that configuration fails at
load with a clear message and the single-node path is unaffected. Verified both
ways. Evidence:
[`benchmarks/qwen4exp-tensor-split-corruption.md`](../benchmarks/qwen4exp-tensor-split-corruption.md).

**Porting the CPU-NUMA backend to a model-support branch that lacks it** —
including how to tell "not compiled in" from "not in the tree", and the
`--list-devices` check that misleads if you forget the environment variable —
is documented separately in [`PORTING.md`](PORTING.md). The build-249 CPU-NUMA
patch has been verified to apply cleanly onto the `glm5next` support line
(base `2e0e57f1`), which is a different lineage from its own base.

## Build-249 CPU-NUMA backend

`llama.cpp-b249-cpu-numa.patch` adds an opt-in Linux CPU device for each NUMA
node. It:

- discovers one hardware thread per physical core while respecting affinity;
- exposes `CPU-NUMA0`, `CPU-NUMA1`, and so on as llama.cpp devices;
- allocates device buffers with strict `mbind()` placement;
- gives each device a persistent, pinned worker pool and asynchronous
  dispatcher;
- fixes Meta host-view mapping and graph-metadata lifetime issues;
- provides a narrow direct host-memory F32 all-reduce, with the corrected
  generic Meta collective as fallback;
- enables device selection in server, common tools, and benchmark binaries.

The feature is Linux-only and disabled unless `GGML_CPU_NUMA_DEVICES=1`.
Runtime documentation is added as `docs/backend/CPU-NUMA.md` in the patched
tree.

Apply and build:

```bash
git checkout 3173a56471c1753650cd806694145ffd6dcace67
git apply --check /path/to/llama.cpp-b249-cpu-numa.patch
git apply /path/to/llama.cpp-b249-cpu-numa.patch

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

The measured four-node runtime contract is:

```bash
export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS=12
export GGML_CPU_NUMA_POLL=100
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
```

Select `CPU-NUMA0` through `CPU-NUMA3`, use tensor split `1,1,1,1`, and treat
the thread count as *per node*. The ready-to-edit Qwen profile is
[`profiles/launch-qwen38-27b.sh`](../profiles/launch-qwen38-27b.sh).

## Build-249 Qwen experimental and MTP support

`llama.cpp-b249-qwen4exp-mtp.patch` is deliberately path-scoped to Qwen
conversion, GGUF mappings, architecture/model loading, and Qwen4 experimental
model code. Apply it to a clean base by itself, or after the CPU-NUMA patch:

```bash
git checkout 3173a56471c1753650cd806694145ffd6dcace67
git apply /path/to/llama.cpp-b249-cpu-numa.patch       # optional
git apply --check /path/to/llama.cpp-b249-qwen4exp-mtp.patch
git apply /path/to/llama.cpp-b249-qwen4exp-mtp.patch
```

Build with the same CMake command above. For MTP, pass a detached sidecar:

```bash
llama-server \
  --model "$MODEL" \
  --spec-type draft-mtp \
  --spec-draft-model "$MTP_MODEL" \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.2
```

The source basis and authors are listed in
[`ATTRIBUTION.md`](ATTRIBUTION.md).

## GLM/SR950 integration

`llama.cpp-a302733-glm-sr950.patch` is the complete tracked delta from the
pinned base. It contains:

- CPU-NUMA devices, Meta correctness fixes, sharded execution, and collectives;
- GLM model, conversion, tensor-parallel, MTP, and server integration;
- exact compact IQ2/IQ3 kernels and expanded Q5_K/Q8_0 x86 paths;
- repack, multi-row speculative-verification, and MoE kernel work;
- focused backend, argument-parser, model-resolution, and server tests.

Apply it only to its own base:

```bash
git checkout a30273376ef669023334fc20ad02ae4ed8196a65
git apply --check /path/to/llama.cpp-a302733-glm-sr950.patch
git apply /path/to/llama.cpp-a302733-glm-sr950.patch

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

Use [`profiles/launch-glm53-full.sh`](../profiles/launch-glm53-full.sh) as the
four-node starting point. The benchmark-selected general profile uses MTP
depth 2 with no confidence gate. The deeper request-scoped profile is intended
only for agentic replay traffic.

## Validation and limits

- All three files pass `git apply --check` against their exact pinned bases.
- Both build-249 patches also apply sequentially to the same clean checkout.
- The build-249 combined source built in Release mode. Of 62 CTest entries, 61
  passed; the remaining tokenizer case could not run because six test fixtures
  were checked out as 132-byte Git LFS pointer files.
- The GLM source delta is whitespace-clean and applies cleanly. Its benchmarked
  executable exercised the production model, MTP, server endpoints, and focused
  kernel paths.
- Strict `mbind()` may fail in containers, restricted services, or when a node
  lacks local memory. It fails explicitly instead of silently losing locality.
- The direct collective accepts only compatible contiguous F32 tensors owned
  by matching CPU-NUMA devices. All other cases use the generic path.
- Every rebase needs a clean build, backend tests, deterministic output
  comparison, and a full-service A/B. Microbench wins have regressed the
  complete service before.

These patches are experimental research artifacts, not upstream-supported
interfaces.
