---
library_name: mlx
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - deepseek-v41
  - quantized
  - 2-bit
  - mtp
  - experimental
---

<p align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="260" alt="DeepSeek">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Vontra-2bit-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="Vontra 2-bit">
  <img src="https://img.shields.io/badge/Status-Experimental-E8A317?style=for-the-badge" alt="Experimental">
</p>

<h1 align="center">DeepSeek-V4.1-Flash · MLX 2-bit · Native MTP weights</h1>

<p align="center">Up to <strong>9.5 tokens/sec so far</strong> on a 256 GiB M3 Ultra Mac Studio in short, text-only custom-runtime tests.</p>

<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">Original model</a> ·
  <a href="https://www.deepseek.com/">DeepSeek</a> ·
  <a href="https://github.com/ml-explore/mlx">Apple MLX</a> ·
  <a href="LICENSE">MIT licence</a>
</p>

> **Experimental weights and standalone runtime, not a drop-in oMLX release.** The custom text adapter is included in [runtime/](runtime/README.md). Stock oMLX and standard MLX loaders have not been validated for this source-layout checkpoint. Read the memory requirements and limitations before downloading.

## At a glance

| Item | Value |
| --- | --- |
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Format | Source-layout MLX safetensors |
| Quantisation | Affine 2-bit, group size 64; selected non-matrix parameters remain higher precision |
| Source precision | Official mixed FP4/FP8 checkpoint, with BF16/F32 components; not a BF16-source conversion |
| Conversion type | Standard requantisation, **not calibrated oQ** |
| Weight payload | 238.80 GB / 222.40 GiB, plus headers and tokenizer/config files |
| Tensors / shards | 143,982 tensors / 48 shards |
| Native MTP | All three DSpark stages retained; 3,584 converted MTP tensors |
| Architecture | `deepseek_v41`, with Engram conditional memory |
| Context | Upstream config specifies 1,048,576 tokens; long context is **not validated** here |

## M3 Ultra Studio performance

Measured on 2026-09-10 with MLX 0.32.0, greedy decoding and the upstream chat encoder in `chat` mode. The custom adapter keeps the text backbone in RAM, leaves Engram tables on SSD, and can cache frequently used Engram rows in process memory.

| Short diagnostic | Output tokens | Timed decode steps | Non-MTP, warm Engram cache | Native MTP, serial verification |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic, 14-token prompt | 9 including EOS | 8 | **9.46 tokens/sec** | 8.54 tokens/sec |
| Python function, 16-token prompt | 24, stopped at output cap | 23 | **8.80 tokens/sec** | 8.17 tokens/sec |

**The headline 9.5 tokens/sec rounds the 9.46 result above.** It is a short observed result, not a sustained-speed guarantee, a broad benchmark, or an oMLX result. Warm cache means the same prompt was run previously; real conversations may have different cache hit rates. Initial model loading and prompt processing are excluded from decode speed.

The optimized non-MTP adapter also produced 8.62–8.97 tokens/sec in a separate repeated two-prompt comparison without the explicit Engram row cache. Native MTP produced identical greedy outputs on both diagnostics but was **slower** with serial verification; accelerated block verification is not validated. Observed accepted draft tokens averaged 1.33 and 1.00 per block, respectively, with output limits truncating some proposals.

### Memory and loading

The measured text backbone occupied 160.88 GiB; 57.22 GiB of Engram tables stayed on SSD. The native MTP weights add 4.15 GiB when loaded. A combined benchmark session reached 166.69 GiB peak process RSS, excluding filesystem-cache memory and other system use.

These observations apply to the custom offload strategy on a **256 GiB Mac**, not to loading every shard into GPU memory at once. Keep substantial room for macOS, temporary allocations and context growth. Cold setup took approximately three minutes in the development environment; storage performance will change that figure.

## Recorded text check

Prompt, using the upstream chat encoder and greedy decoding:

```text
What is 2+2? Answer briefly.
```

Actual output, excluding the EOS marker:

```text
2 plus 2 equals 4.
```

The coding diagnostic checked a 24-token prefix only, not a completed or executed Python function. Neither test establishes broad reasoning, coding or chat quality.

## Download

```bash
hf download Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP --local-dir DeepSeek-V4.1-Flash-MLX-2bit-MTP
```

The download includes the standalone runtime, pinned dependencies and tests. On a 256 GiB Apple silicon Mac, from the downloaded directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements.txt
python -m unittest discover -s runtime -p 'test_*.py'
python runtime/generate.py --model . --resident-backbone --prompt "What is 2+2? Answer briefly." --max-tokens 16 --repeat 2
```

The second run reuses the model and Engram row cache. Add `--mtp` to test native DSpark with serial verification, which is currently slower. The command releases memory when it exits; no server is started. See the [runtime guide](runtime/README.md) for the 128-token diagnostic context cap and other restrictions. This does not install support into oMLX.

## Validation and limitations

- All 48 converted shards are retained, including native MTP and vision-related weights.
- The text adapter executed real greedy generation; its reference, deferred-execution and compiled modes matched token-for-token on the two short prompts.
- Native MTP proposals were checked against target-model logits, with rejected tokens excluded from the verified cache.
- All 17 included numerical, cache, verification and command-line runner tests passed; unpublished experimental candidates are excluded.
- No independent whole-model numerical comparison against the higher-precision upstream model has been completed.
- Vision, long-context use, tool calls, production serving and stock oMLX compatibility are not validated.
- Two-bit requantisation can substantially reduce quality relative to the already-quantised source; repetition, incoherence and task failures remain possible.

For the upstream architecture, evaluations, intended use and limitations, see the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

## Licence and attribution

The original model is released under the **MIT licence**, included as [LICENSE](LICENSE). Model design, training and upstream documentation belong to DeepSeek and its contributors; this is an independent community conversion by [Vontra](https://huggingface.co/Vontra), not an official DeepSeek release.

[Follow Vontra for new Apple Silicon releases and fixes.](https://huggingface.co/Vontra)
