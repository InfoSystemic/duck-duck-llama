# GLM-5.3 Full unequal tensor split correction

This is an isolated, source-only candidate. The existing engine source, binaries,
launchers, live services, and memory placement are unchanged.

The original `llama-model.cpp` assigns split granularity 1 to GLM's `attn_q_a`,
`attn_kv_a_mqa`, and shared-expert gate/up/down tensors. For the actual Full GGUF,
the first two have input width 6144 and the shared down projection has input width
2048. All five are Q8_0 in the checkpoint, requiring 32-element block boundaries.
Unequal `1.41,1.43,1.43,1.00` splitting violates that requirement for 234 tensors.

The ten-line `llama-model.patch` selects `lcm(source_block_size, 256)` for those
GLM tensor patterns. The 256-element quantum also keeps slices eligible for the
selected Q8_0-to-Q5_K repacking path. Shared gate/up obtains its reference block
size from the same down tensor as before, so all three use identical channels.
The balanced `1,1,1,1` split remains identical on every actual Full tensor.

Validation in `verification.json`:

- 1,809 actual tensors from all eleven Full GGUF shard headers; 9,546,880 header
  bytes inspected and hashed. Tensor payloads are not loaded.
- 32 ratio schedules including the failing split, balanced/default-zero ratios,
  zero-weight devices, highly uneven ratios, and 25 seeded random schedules.
- 57,888 checks of nonnegative slices, source quantization block alignment, and
  conservation of the split dimension.
- 2,432 shared-expert triplet checks and 2,528 MLA head-consistency checks.
- 7,488 checks that affected reduction dimensions retain 256-element alignment.
- The original callback reproduces 234 invalid AXIS_0 splits for the failing
  ratio; the first is `blk.0.attn_kv_a_mqa.weight`.
- Both complete split callbacks are extracted directly from the original and
  candidate source and compiled with real GGML tensor definitions and the
  existing `ggml_blck_size` library. Model/hparams scaffolding supplies only the
  real header metadata, including 78 backbone layers after excluding MTP.
- The complete candidate translation unit also passes `-fsyntax-only` against
  the actual engine headers.

Run the bounded check from the AI-Server root:

```bash
taskset -c 127 tools/deepseek-v41-cpu-reference-0910/venv/bin/python \
  serving/fleet-0912/glmfull-split/prepare_and_check.py
```

`llama-model.parent.cpp` and `llama-model.patched.cpp` are reviewable snapshots.
`llama-model.patch` is the proposed source change; it has not been applied to the
engine. The checker records hashes and compile commands for reproduction.

These results prove the metadata-level alignment fix for the inspected Full
model. They do not prove a whole model loads or produces equivalent logits with
an unequal split. Graph propagation, CPU kernel selection, throughput, and
per-node memory demand still require a separate model trial. Redistributing
existing tmpfs pages can preserve the already exercised balanced split and is
being handled separately by the parent task.
