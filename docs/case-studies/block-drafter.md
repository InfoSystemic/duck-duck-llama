# Making a block drafter work: six defects, one control, one cycle model

MiMo-V2.6-Pro ships a drafter that proposes eight tokens per forward pass. On this CPU server that is the difference between 7.9 and 27 tok/s. Getting there took six fixes, and none of them was in the drafter's weights.

This case study records how each defect was found, because the method transfers to any new drafter. It also covers the control that makes a block drafter pay, and the cycle model that shows where the ceiling is.

Evidence:

- [speculation notes](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/STATE-SPECULATION-20260922.md)
- [feature patches](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/)
- [benchmark](../../benchmarks/mimo-v26-pro-cpu-20260923.md)

## First, the drafter that is not one

The checkpoint contains three MTP blocks, and the engine can draft with them. Drafting with them made decode 42% slower than no speculation.

Acceptance was flat and low across every kind of prompt. Verbatim repetition, where a trained next-token head should accept about 95%, accepted 0.097–0.159. Flat acceptance across prompts looks like a plumbing bug, so one was hypothesised: the hidden state reached the heads before the final norm, where every sibling architecture exports it after. A patch moved the export.

- The target's output stayed byte-identical, so the patch was safe.
- Acceptance did not move, so the hypothesis was [refuted](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/mimo2-hnextn-postnorm.patch).

The model card settles it. The inference drafter is a separate five-layer sliding-window model, the kind called DFlash; the MTP blocks are training heads.

**Lesson: read the model's deployment section before debugging a drafter's acceptance.**

## Six defects between the drafter and a working one

| # | Defect | Symptom | Found by |
| --- | --- | --- | --- |
| 1 | The converter drops `.weight` from the attention-sink tensor name. The source key ends in `_bias`, so nothing is stripped. | The loader asks for `attn_sinks.weight`, but the file has `attn_sinks` | The draft load, after 27 minutes of target load |
| 2 | The drafter's Qwen3-style graph never loads attention sinks. The sibling branch does. | `wrong number of tensors; expected 63, got 58` | The draft load again, after another 27 minutes |
| 3 | The target (`mimo2`) never exports the per-layer hidden states the drafter reads: layers 1, 16, 32, 48 and 70. | Health check passes; the first request aborts in `llama_decode` | An 8-token first request |
| 4 | Exporting the last layer at full width disabled that layer's early row crop. After that, no node referenced the output-row index, so it never got a buffer. Separately, under the NUMA tensor-parallel backend an alias of the layer input has no buffer of its own; the export has to be a dedicated copy node. | Abort in `set_input` on a null buffer | The 4-layer proxy, in one minute |
| 5 | **The mask token's embedding is empty in the target.** DFlash builds its block as the last token plus seven mask tokens, embedded through the *target's* table. In the base checkpoint that row has norm 0.0008. The real embedding ships beside the drafter as `mask_embedding.pt`. | Acceptance 0.014–0.072, worse than the broken MTP heads | Reading the reference `dflash.py` against the target's embedding row |
| 6 | **The drafter's rotary dimension count was never written to the GGUF.** The config has `partial_rotary_factor` 0.5 on 128-dim heads, so only 64 dimensions carry position. The runtime defaulted to all 128, putting every drafter position wrong. | Looks like a weak drafter | Dumping every GGUF key beside every `config.json` key |

Defect 5 is fixed [in place](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/patch-mask-embedding.py): one Q8_0 row of the target's embedding table, 6,528 bytes at a known offset in a 42 GiB shard. The original bytes are saved and restorable. The target never generates that token, so nothing else changes.

After the fix, the row reads back at norm 1.8618, with cosine 0.999944 to the shipped vector.

Fixing defect 5 alone moved verbatim-repeat acceptance from 0.020 to 0.054. That is how it was clear the defect was genuine, and also that it was not the last one. Fixing defect 6 took the same prompt to 1.000.

## Method that transfers

- **Load the draft model standalone before loading a large target.** The draft loads last, so defects 1 and 2 each cost a 27-minute tensor-parallel load before they surfaced.

  Standalone loading takes 0.3 s. The pass condition is that it reaches `dflash requires ctx_other to be set`: every tensor was claimed, and only the shared-embedding wiring is missing, which is correct for a standalone load.

- **Send a tiny first request before measuring.** A healthy `/health` proves only that the weights loaded. Graph defects appear on the first decode.

- **Keep a 1-minute loop for graph changes.** A 4-layer proxy of the same architecture loads in about a minute. The drafter asks for target layer 70, which the proxy does not have. That is metadata, so rewriting one key to layers [0–4] lets the real drafter drive the proxy.

  The text is garbage and says nothing about acceptance or speed. It still exercises the same graph construction, allocation and scheduling. Defect 4 was found this way.

- **Diff every key of the produced GGUF against the source config first.** Two hypotheses failed before this diff found defect 6. One key that differed was a false alarm: the reference code never applies `attention_value_scale`.

- **Treat every non-weight file in a drafter's directory as required.** A converter that reads only safetensors silently drops files like `mask_embedding.pt`, and the resulting failure looks like a weak drafter.

- **Keep the target's parity check running throughout.** The target's greedy output stayed byte-identical through all six fixes. That is how each fix was known to be in the draft path and not in the model.

## The control: `p_min`, not a fixed draft length

A block drafter's acceptance depends heavily on the text:

| Workload | Tokens per verify cycle | Acceptance | tok/s |
| --- | ---: | ---: | ---: |
| Counting | 7.39 | 1.000 | 20.35 |
| Verbatim repeat | 7.39 | 1.000 | 19.19 |
| Code | 5.33 | 0.667 | 13.84 |
| Memorised list | 4.36 | 0.525 | 11.65 |
| Open prose | 2.04 | 0.160 | 5.51 |

No speculation gives 7.94 tok/s. On open prose, then, the drafter loses. The reason is the verify pass: a rejected row still costs a row of expert bandwidth.

Shortening the fixed draft length helps prose (8.04 tok/s at n = 2) but gives up everything the drafter earns on predictable text.

The right control is `--spec-draft-p-min`: cut the block at the first position whose top candidate falls below the threshold. At n = 7 with `p_min` 0.5, open prose goes from 6.23 to 8.80 tok/s (+41%, acceptance 0.20 → 0.73), and code is flat (13.40).

Both overrides were made readable at runtime from a file, so a whole draft-length × `p_min` surface comes out of one 30-minute load. See the [draft-length patch](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/dflash-runtime-draft-n.patch); the `p_min` override is in the [production source state](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/SOURCE-STATE-build-prod-0922j.diff).

The optimum is not a constant: re-tune it whenever the cycle cost changes. With cheaper Q4_K attention, n = 4 with `p_min` 0 won outright. After the attention kernel made the verify pass cheaper, the production setting was re-checked across five workloads and held.

## The cycle model, and what it rules out

The verify cycle fits `cycle_ms = 99.2 + 36.83 · f(rows)` to within 0.7% over draft lengths 2–7 ([fit](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/fit-cycle.py)). `f(rows) = 384 · (1 − (1 − 8/384)^rows) / 8` is the expected number of distinct 8-of-384 expert sets that `rows` tokens touch.

- **The constant, 99.2 ms,** is what one cycle reads once, however many rows it verifies: attention, the router, the output head and one drafter pass.
- **The slope, 36.83 ms,** is one 10.3 GiB expert set at 300 GB/s, which is 79% of this machine's measured 381.6 GB/s.

At that duty cycle, faster serving code has little left to gain. The one variable left is tokens per cycle, and that belongs to the drafter.

## Draft quality is free; target quality is not

Speculative decoding is exact: a drafted token survives only if it equals what the target would have produced. So any approximation confined to the draft path can only cost acceptance, never output quality. That covers a cheaper drafter, a requantised draft output head, or a shortlisted argmax. Judge such changes on acceptance; perplexity cannot see them.

The opposite holds for anything the target reads. On GLM-5.3-Flash, an F16 copy of the router looked bit-identical on a 4-layer check. On the full model it flipped near-tie expert choices, and acceptance fell from 0.775 to 0.763 ([revision i](../../engineering/2026-09-23/archive/serving/fleet-0920-flash18/deploy-0920i/95-f18-0920.conf.proposed)).

One implementation trap applies to MTP-style drafting. There, the draft runs on the target's own weights (`shares_model = !has_draft`). A cheap second copy of a tensor must therefore be bound by *graph*, not passed as a separate draft model. Passing `--spec-draft-model` forces a second full load.
