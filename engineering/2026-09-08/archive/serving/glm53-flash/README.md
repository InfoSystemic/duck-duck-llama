# GLM-5.3-Flash on SR950

Pinned releases:

- Recommended model: `unsloth/GLM-5.3-Flash-GGUF` at `2975ab414d30340466d8c51533c6e91f0cca64c1`
- Recommended quant: `UD-IQ2_XXS` (101,844,951,808 bytes of text weights)
- Higher-quality comparison: `UD-IQ3_XXS` at `ac47690c15c8703615ab7d9c1ef2293d45372757` (120,367,571,715 bytes of text weights)
- Projector: `mmproj-F16.gguf` (1,128,047,200 bytes)
- Runtime: Unsloth llama.cpp PR 27754 at `2e0e57f1008053bae4902a772da85e3eb99d4aff`

The runtime is built in `../../engines/llama.cpp-glm53-flash/build-sr950` and its
`glm5next` architecture test passes on CPU with zero NMSE and a clean roundtrip.

Download and verify the recommended IQ2 model after at least 108 GB is free at
the destination:

```bash
./download-glm53-flash.sh
# Or, while the model SSD is unavailable:
GLM53_DEST=/dev/shm/ai-models/GLM-5.3-Flash ./download-glm53-flash.sh
```

To fetch the higher-quality IQ3 comparison instead, set
`GLM53_QUANT=UD-IQ3_XXS`; it requires at least 130 GB free. Both downloads are
revision-pinned and checked against quant-specific SHA-256 manifests. IQ2 is
the capacity-efficient starting point, but should not replace IQ3 until it
passes the deterministic prompt and quality suites on this host.

Run the local OpenAI-compatible server on port 5830:

```bash
./launch-glm53-flash.sh 5830
```

The launcher auto-selects a complete IQ2 tree before IQ3, or honors an explicit
`GLM53_QUANT`. The safe simultaneous-service profile uses 64K context, F16
cache, required `--flash-attn off`, and CPUs on NUMA node 0. Memory prefers
node 0 but may spill when other workloads reduce local headroom. Override
context with `GLM53_CTX_SIZE` after measuring headroom. The launcher prefers
verified RAM-backed files and refuses a stale `/models` mount whose block
device is absent.
