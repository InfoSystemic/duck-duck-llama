---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- deepseek
- deepseek-v4.1
- llama.cpp
quantized_by: vcruz305
---

# DeepSeek-V4.1-Flash GGUF

llama.cpp / Unsloth-style GGUF of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

This is **V4.1-Flash** (`DeepseekV41ForCausalLM`, Causal Encoder-Decoder, CSA2, Engram, DSpark n=3). It is not V4-Flash-0731.

## Status

Weights are not in this repo yet. Ladder in order: **Q2_K_M → Q3_K_M → Q4_K_M → Q5_K_M**. Each file uploads when that step finishes.

Official checkpoint is mixed **FP8 + FP4 experts** (~475 GiB, 48 shards). High-quality convert must keep expert FP4 bit-exact where GGUF MXFP4 matches, not blindly requant routed experts.

## Files (will land here)

| File | Quant | Notes |
| --- | --- | --- |
| `DeepSeek-V4.1-Flash-Q2_K_M.gguf` | Q2_K_M | first |
| `DeepSeek-V4.1-Flash-Q3_K_M.gguf` | Q3_K_M | |
| `DeepSeek-V4.1-Flash-Q4_K_M.gguf` | Q4_K_M | |
| `DeepSeek-V4.1-Flash-Q5_K_M.gguf` | Q5_K_M | |

Split parts if a single file exceeds Hub limits.

Apache/MIT from upstream. Credit: DeepSeek. GGUF pack: Victor Cruz (`vcruz305`).
