# DeepSeek-V4.1-Flash: component work, 2026-09-12

These files operate on the actual V4.1 reference path. They do not modify a live
service, load checkpoint weights, replace V4.1 with V4, or claim model throughput.

## Official compressor oracle for the incoming llama.cpp port

`compressor_oracle_0912.py` runs the unchanged publisher `Compressor` through the
existing CPU import shim, with small synthetic weights. The shim replaces GPU
kernels at import time; it does not replace `Compressor.forward` or `RMSNorm`.

The generated `results/compressor-oracle/compressor-oracle.npz` holds inputs,
weights, and expected complete outputs. Its JSON companion records source hashes,
emission positions, coverage, and limits. All 332,928 checked output values match
exactly between full prefill, incremental decode, and reuse after an earlier
complete or incomplete request. Coverage includes ratios 1/2, head widths 32/512,
batch size 2, and prefill lengths 1/2/3/6/7. Inputs and BF16 outputs are exported
as F32 arrays without losing any value; F32 projection weights retain F32 values.

Important integration requirements from the actual reference:

- Ratio 1 uses a BF16 projection and normalization, without a gate or state.
- Ratio 2 uses F32 projection/gate weights and a per-feature softmax over each
  consecutive token pair. It rounds the pooled result to the input dtype before
  applying RMSNorm. Partial groups persist until a pair completes.
- Compressed latent RoPE uses the first token's position in the group. This is
  recorded as metadata; the compressor fixture does not apply RoPE.
- After `Compressor.forward`, `_compress_kv` additionally applies FP4 quantization
  with block size 16 and E4M3 scales. Window KV uses FP8 block size 32/E8M0, while
  indexer Q/K uses FP4 block size 32/E8M0. Passing these fixtures does not establish
  parity for a port that omits those later quantization steps.
- `Attention.forward` normalizes `wq_a(x)` using `q_norm`, but applies no additional
  per-head RMSNorm after `wq_b`. The old local `deepseek4.cpp` unconditionally adds
  that extra normalization at lines 917-920. A genuine V4.1 port must distinguish
  that behavior from V4.

The earlier local port report's lack of a GPU oracle does not block testing these
CPU-compatible operators. Full checkpoint/logit comparison remains separate.

Run from the AI-Server root:

```bash
taskset -c 127 tools/deepseek-v41-cpu-reference-0910/venv/bin/python \
  serving/fleet-0912/deepseek/compressor_oracle_0912.py \
  --output serving/fleet-0912/deepseek/results/compressor-oracle
```

## Experimental native sparse prefill adapter (unselected)

`sparse_prefill_0912.py` adds `NativeSparsePrefill`, an opt-in subclass using the
existing compiled native sparse attention library. It invokes the native kernel
directly for each prefill query, validating shapes once and avoiding repeated
Python tensor views/copies and dispatch guards. The native kernel's 64-position
softmax chunks, MKL reductions, and BF16 probability rounding are unchanged.
Unsupported requests and individual exceptional rows retain the original
fallback. Native call metrics count query rows, including prefill rows.

`results/sparse-prefill-final-check.json` records **45 byte-exact attention cases,
4,998,306 BF16 output values**, input preservation, invalid-index rejection,
idempotent installation, and autograd/stride/NaN/denormal/disabled fallbacks.

Single-core timings are mixed and noisy under current host activity. The direct
adapter's observed medians range from 0.91x to 1.16x across tested shapes; this is
not evidence for a serving promotion. It remains unselected. The final correctness
check intentionally does not repeat timing. Existing earlier timing records retain
their own source hashes and should not be treated as current full-model evidence.

```bash
taskset -c 127 tools/deepseek-v41-cpu-reference-0910/venv/bin/python \
  serving/fleet-0912/deepseek/check_sparse_prefill_0912.py \
  --output serving/fleet-0912/deepseek/results/sparse-prefill-final-check.json \
  --threads 1
```

Before any selection, benchmark the complete native model with varied prompts at
the actual 16-worker setting. Updating a running Goal2 runtime requires a deliberate
integration because its configuration assertions bind the selected adapter object.
None of the existing launchers, selection manifests, or serving sources was edited.
