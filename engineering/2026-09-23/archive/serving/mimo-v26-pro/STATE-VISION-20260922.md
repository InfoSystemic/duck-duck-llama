# MiMo-V2.6-Pro vision: root cause and fix (2026-09-22 evening)

**Symptom (from STATE-SPECULATION-20260922.md):** with `--mmproj` loaded, most images were answered with `?` repeated.
Small flat-colour images (64-128 px) answered correctly, which read as "broken above 128x128".

**Root cause: F16 overflow inside the ViT, not the window mask.** Tensor-by-tensor comparison against Xiaomi's own
`MiMoVisionTransformer` (imported from the checkpoint's `modeling_mimo_v2.py`) showed every block finite through
layer 26, then in block 27 (the last, a full-attention block):

    ffn_swiglu-27   maxabs 1.077e+05     <- above F16's 65504
    ffn_down-27     nonfinite 6400 of 128000

The CPU `mul_mat` converts its activation to the weight's `vec_dot_type` -- F16 for an F16 weight -- so the SwiGLU
output of ~1e5 becomes inf, the down projection turns those rows into inf/NaN, the merger spreads them into the image
embeddings, and the language model answers `?`. It is content-dependent (high-norm outlier tokens), not a size
boundary: the same textured test image fails at 128x128 too. Flat colours never reached the overflow.

**Fix: an F32 projector**, converted from the original BF16 weights (41 s):

    convert/convert_hf_to_gguf.py /models/mimo-v26-pro/skel-src --mmproj --outtype f32 \
        --outfile /models/mimo-v26-pro/gguf/mmproj-MiMo-V2.6-Pro-RL-F32.gguf        # 5.45 GB

BF16 would also have the range, but this CPU has no AVX512-BF16, so it would be emulated and would round every
activation to 8 mantissa bits. F32 is exact and measured no slower than F16 (576x576: 25.9 s vs 28.5 s including load,
8 threads). `launch-mimo-tp.sh` now defaults `MMPROJ_FILE` to it.

## Verified against the reference

`tools/vision-embd` (C++, runs one image through libmtmd exactly as llama-server does, dumps every ViT tensor) vs
`tools/vision-ref.py` (Xiaomi's module, float32):

| image | patches | F16 projector | F32 projector: final cos / per-token median |
|---|---:|---|---|
| 128x128 textured | 64 | NaN in 3 tokens | 0.99999 / 1.00000 |
| 160x160 | 100 | NaN in 5 tokens | 0.999997 / 1.00000 |
| 224x320 | 280 | -- | 0.99991 / 1.00000 |
| 576x576 | 1296 | NaN in 112 tokens | 0.99986 / 0.99999 |

The window mask, column reordering, 2-D RoPE, GQA mapping and merger all match: with them active (160 and above)
the error stays at float-accumulation level.

## The sink semantics question -- settled in favour of what llama.cpp does

Xiaomi's HF vision code applies each head's sink as an additive bias on **key 0's logit**
(`sink_bias[..., 0] = sinks`), while clip.cpp uses a gpt-oss-style virtual column (extra logit, V = 0). They differ
by cos 0.77 in the final embeddings. The virtual column is correct:
- the same file implements the TEXT model's sinks as the virtual column (`torch.cat([attn_weights, sinks])`, and
  `s_aux` under flash-attention);
- SGLang, Xiaomi's recommended serving stack, runs vision through FA3 with `s_aux`; sglang issue #37983 describes the
  sinks as included "in the online softmax denominator";
- the key-0 bias is what you write when SDPA has no sink argument, and it is not equivalent.
With the reference switched to the virtual column (`REF_SINK=virtual`), llama.cpp matches it to cos 0.99999.

## Also fixed: `--image-max-tokens` was ignored for this projector

`clip.cpp` read `image_min/max_pixels` straight from metadata for `PROJECTOR_TYPE_MIMOVL` and never applied the
`--image-min/max-tokens` overrides. One line (`hparams.set_limit_image_tokens()` after reading them) makes the flags
work; without flags nothing changes. `patches/mimovl-image-token-limits.patch`. Production now caps at 1280 tokens
(~1.3 MP, larger images downscaled): uncapped, the metadata allows 12.8 MP = 12.5K tokens for one image.

## Tools left here
- `tools/vision-embd.cpp` -- `vision-embd <mmproj> <text-model(vocab only)> <image> <out.bin> [fa] [dump_dir] [threads]`;
  `VE_STATS=1` prints max|x| and non-finite counts for every graph node (that is how the overflow was found).
- `tools/vision-ref.py` -- `mkimg` / `ref` (`REF_SINK=virtual` for the SGLang semantics) / `cmp`.

---

# Tool calling: the chat layer, not the model (2026-09-22 21:00)

Smoke tool-call args once came back as `{"city":"Paris\n</invoke>"}`. 20 samples at Xiaomi's recommended T=1.0 /
top_p 0.95: **3 malformed**, two of them writing tool call after tool call INSIDE the argument until max_tokens.
The same seeds through raw `/completion` were all clean compact calls, which put the fault in the chat layer.

MiMo's template contains `<function=`/`<parameter=`, so llama.cpp routes it to the Qwen3-Coder XML handler, whose
parser and lazy grammar REQUIRE `<tool_call>\n<function=f>\n<parameter=a>\nvalue\n</parameter>\n</function>\n`.
MiMo's own template renders -- and the model emits -- the compact
`<tool_call><function=f><parameter=a>value</parameter></function></tool_call>`. Offline harness
(`tools/chat-toolcall-test.cpp`): compact calls parsed to NO tool calls at all; in production the grammar forced the
newline layout, pushing the model off-format. Fix (`patches/chat-xml-toolcall-optional-newlines.patch`): every
structural newline optional, values end at `\n</parameter>` or `</parameter>`; both layouts parse, multi-line values
keep their inner newlines. Upstream b23efaa has the same strict handler (`common/parsers/qwen3-coder.cpp`).
Probe: `tools/toolcall-probe.py [port] [n]`.

## The image cap, exercised (22:00)
A synthetic 3840x2160 screenshot (8.3 MP; uncapped it would be ~8,100 image tokens) was downscaled under
`--image-max-tokens 1280` to 1,256 prompt tokens and described correctly ("red, green, and blue from left to right, and
a dark grey horizontal bar ... below them") in 45 s, measured while other sessions' jobs held two of MiMo's NUMA nodes.

**Live on build-prod-0922e (21:51):** `tools/toolcall-probe.py` 0/20 malformed (was 3/20 on 0922d); streamed
three-argument `edit_file` calls 5/5 exact at T=1.0 (the partial/streaming parse path); smoke tool call clean.
