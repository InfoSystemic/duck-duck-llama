# MiMo speculation: what was measured, what was refuted, and what is running

Written 2026-09-22 by the Claude session working the "MiMo to 16.5 tok/s" goal.

**What changed in your files:** `launch-mimo-tp.sh` gained a `SPEC=dflash` branch and a `BUILD=<dir>` override (both
backwards-compatible, defaults unchanged). `gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE-00001-of-00013.gguf` had ONE 6528-byte row
rewritten — the mask token's embedding, which was empty; the original bytes are in `gguf/token_embd_mask_row.orig` and
`patch-mask-embedding.py --restore` puts them back. Your `build/` tree is untouched; the engine patches are built into
`build-dflashn` and kept as files in `patches/`. `conversion/qwen.py` in the engine tree currently HAS the converter
patch applied (deliberately — see the note at the end about reverting patches while a patched build is in service).

## RESULT: DFlash works. 19-20 tok/s on predictable text, 13.8 on code, against 7.94 with no speculation.

Measured 2026-09-22 03:57 after six fixes (below). Trunk parity byte-identical throughout, so none of them touched the
model: `golden.py --compare` gives IDENTICAL tokens and max |dlogprob(top1)| 0.0000 on all three prompts.

| prompt | tok/s | tokens/cycle | acceptance |
|---|---:|---:|---:|
| counting | **20.35** | 7.385 | **1.000** |
| verbatim repeat | **19.19** | 7.385 | **1.000** |
| code | 13.84 | 5.333 | 0.667 |
| memorised list | 11.65 | 4.364 | 0.525 |
| open prose | 5.51 | 2.043 | 0.160 |
| *(no speculation)* | *7.94* | *1.0* | — |

**The spread is the point.** DFlash drafts a whole 8-token block per forward, and a rejected row still costs a full row
of expert bandwidth in the verify pass. So it is 2.6x on text it can predict and worse than useless on text it cannot.

**The cycle model fits to 0.7%** over draft lengths 2-7 (`fit-cycle.py`): `cycle_ms = 99.2 + 36.83 * f(rows)`,
`f(rows) = 384*(1-(1-8/384)^rows)/8`. The 99.2 ms is everything read once per cycle (attention, router, LM head, one
DFlash forward); 36.83 ms is one expert set of 10.3 GiB = **300 GB/s, 79% of this host's measured 381.6**. Better duty
than GLM-5.3-Flash's 61%, so there is little op-level slack here — only bytes and tokens-per-cycle move the number.

On the hard prompt the draft-length curve is monotone the wrong way (n=7 6.28 ... n=2 8.04), which says a FIXED draft
length is the wrong control. The right one is `--spec-draft-p-min`: the plain-DFlash branch truncates the block at the
first position whose top-candidate probability falls below it. Now runtime-overridable through
`LLAMA_SPEC_DRAFT_PMIN_FILE` so the whole p_min x n surface comes out of one load (`sweep-pmin.sh`).

## Why it took six fixes: `--spec-type draft-mtp` is a dead end for this model

| configuration | tok/s | tokens/cycle | draft acceptance |
|---|---:|---:|---:|
| no speculation (`results/mimo-tp4-nospec.json`) | **7.94** | 1.0 | — |
| MTP depth 3 | 4.58 / 4.18 | 1.25 / 1.13 | **0.082 / 0.042** |
| MTP depth 2 | 5.03 | 1.13 | 0.062 |
| MTP depth 1 | 6.21 | 1.11 | 0.110 |

Speculation through the nextn heads makes decode **42% slower**. `draft-probe.py` (new, beside this file) runs prompts
from verbatim repetition to open prose and the acceptance is flat and low on every one of them — 0.097 on *"repeat the
quick brown fox twenty times"*, where a trained next-token head should accept ~95%. Compare GLM-5.3-Flash at 0.775.

**The reason is in MiMo's own model card** (`/models/mimo-v26-pro/src/README.md`):

> "Multi-Token Prediction (MTP): **5-layer** speculative decoder … **5 SWA layers, window 1024** …
> 5-layer SWA MTP drafter (**DFlash-style**). Predicts **7 subsequent tokens per forward pass**."

The drafter MiMo ships is `src/dflash/` — five layers, window 1024, block size 8. The three `model.mtp.layers.{0,1,2}`
blocks inside the main checkpoint are a *training* artifact (the MTP loss heads), not an inference drafter. There was no
plumbing bug to find. SGLang's recommended recipe in the same README runs EAGLE with `--speculative-num-draft-tokens 4`.

## One hypothesis measured and refuted, so nobody repeats it

`mimo2.cpp` exports `h_nextn` **before** `model.output_norm`, while glm5next, deepseek32, qwen35 and qwen3next all export
it **after** (glm5next even carries a comment recording 1.1% acceptance before that export existed). Patching it to
post-norm and rebuilding changed **nothing**: trunk output byte-identical on all three golden prompts with
max |dlogprob(top1)| 0.0000, and acceptance 0.097 vs 0.159 — no better. Patch kept, with its result written into the
header, at `patches/mimo2-hnextn-postnorm.patch`; built into `engines/llama.cpp-mimo-tp/build-mtpfix` (default ON there,
`LLAMA_MIMO2_HNEXTN_POSTNORM=0` restores upstream). **`src/models/mimo2.cpp` in your tree is untouched.**

Also verified and NOT the problem: `blk.70.nextn.{enorm,hnorm}`, `attn_norm`, `ffn_norm` and `layer_output_norm` in the
GGUF are byte-identical to `model.mtp.layers.0.{enorm,hnorm,input_layernorm,pre_mlp_layernorm,final_layernorm}` in
`model_mtp.safetensors`, so the conversion is faithful; `layer.layer_out_norm` is created for nextn blocks and the name
template matches; `chain_heads` is correctly active (`ctx_other` is only set for GEMMA4_ASSISTANT/EAGLE3/DFLASH, so
`is_mem_shared` is false and all three heads are used, one per draft step).

**One latent bug worth fixing anyway** (it does not explain the above, because it is invisible below 128 tokens of
context): `load_arch_hparams` fills the SWA pattern with `hparams.n_layer()` entries, i.e. 70, so `is_swa_impl[70..72]`
— the MTP blocks — are left at their default. Those blocks carry `attention_sink_bias`, and `config.json` sets
`add_swa_attention_sink_bias: true` / `add_full_attention_sink_bias: false`, so they are SWA layers and are currently
being built as full-attention ones. Use `hparams.n_layer_all`.

## What is running now: DFlash

Converted in about a minute; everything needed was already in the fork.

    hf download XiaomiMiMo/MiMo-V2.6-Pro-RL --include "dflash/*" --local-dir /models/mimo-v26-pro/src   # 5.54 GB
    mkdir -p /models/mimo-v26-pro/dflash-src
    ln -sf /models/mimo-v26-pro/src/dflash/dflash_draft_model.safetensors /models/mimo-v26-pro/dflash-src/model.safetensors
    cp /models/mimo-v26-pro/src/dflash/config.json /models/mimo-v26-pro/dflash-src/
    <engine>/convert_hf_to_gguf.py /models/mimo-v26-pro/dflash-src --target-model-dir /models/mimo-v26-pro/src \
      --outtype q8_0 --outfile /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-DFlash-Q8_0.gguf

**The symlink is not cosmetic.** `ModelBase.get_model_part_names` selects parts with `filename.startswith("model")`, and
the checkpoint file is `dflash_draft_model.safetensors`, so a direct conversion succeeds while silently writing
`n_tensors = 0, metadata only`. Read that line before trusting the output.

Result: `general.architecture=dflash`, block_count 5, `block_size 8`, `target_layers [1,16,32,48,70]`, sliding_window
1024 on all five, `attention.causal=false`, mask token 151675, `fc.weight [30720, 6144]` (five concatenated target hidden
states), 63 tensors, 2.94 GB. It ships no embeddings and no LM head — it borrows the target's through `ctx_other`, which
`llama-context.cpp` wires for `LLM_ARCH_DFLASH`.

`launch-mimo-tp.sh` gained a `SPEC=dflash` branch and a `BUILD=<dir>` override (default `build`, so your existing usage is
unchanged). Launch: `BUILD=build-dflashn SPEC=dflash NMAX=7 ./launch-mimo-tp.sh`.

### Three real defects in the fork, found by loading it

**1. The converter emits the attention sink without a `.weight` suffix.** The HF key is
`self_attn.attention_sink_bias`, which ends in `_bias` rather than `.bias`, so no suffix is stripped and the mapper
writes a bare `blk.N.attn_sinks` — while every loader in the tree asks for `tn(LLM_TENSOR_ATTN_SINKS, "weight", il)`.
`patches/dflash-converter-attn-sinks-suffix.patch`.

**2. `dflash.cpp`'s Qwen3-style branch never loads attention sinks at all.** The DSpark/DSV4 branch above it does; the
Qwen3 branch creates 11 tensors per layer and omits the sink, so a drafter trained with sinks (MiMo sets
`add_swa_attention_sink_bias: true`) fails with `done_getting_tensors: wrong number of tensors; expected 63, got 58` —
one unclaimed sink per layer. The patch loads it `TENSOR_NOT_REQUIRED` and passes it to `build_attn`, so drafters
without sinks are unaffected. `patches/dflash-qwen3-attn-sinks.patch`.

**3. `mimo2.cpp` never exports the trunk's per-layer hidden states.** An EAGLE3/DFlash drafter is fed the residual
stream at several trunk depths, requested through `cparams.embeddings_layer_inp`, and
`llm_graph_result::set_outputs` asserts every requested index is non-null. MiMo's DFlash asks for layers
[1, 16, 32, 48, 70]; this graph exported none of them, so the server passed `/health`, accepted the first request and
aborted inside `llama_decode` with `llama-graph.cpp:1552: GGML_ASSERT(t_layer_inp[il] != nullptr)`. `qwen3.cpp` does it
in one line at the top of the layer loop. Two details matter: the driver indexes the result **by batch row**
(`layer + (i_batch_beg + offset + i) * n_embd`, and the context allocates `n_embd * n_batch` per enabled layer), so the
export must happen **before** any out-ids filtering — which is why requesting the post-trunk state at index `n_layer`
also has to disable the last layer's early crop. `patches/mimo2-export-layer-inputs.patch`.

**`build-dflashn` = the shared `build` plus `dflash-qwen3-attn-sinks.patch`, `mimo2-export-layer-inputs.patch` and `dflash-runtime-draft-n.patch`.** The
source files were reverted after building, so your tree is clean; re-apply the patches before rebuilding that directory.

### Validate the DRAFT model standalone first — it costs 0.3 seconds

Both defects above surfaced **after** a 27-minute tensor-parallel load of the 518 GiB target, because the draft model is
loaded last. Loading the 2.9 GB draft on its own reproduces the tensor error immediately:

    LD_LIBRARY_PATH=<build>/bin <build>/bin/llama-server --port 18199 \
      --model /models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-DFlash-Q8_0.gguf --ctx-size 512 --threads 8

The pass condition is reaching `failed to initialize the context: dflash requires ctx_other to be set` — that error means
every tensor was claimed and only the shared-embeddings wiring is missing, which is expected standalone. Anything about
tensor counts is a real failure. (The same trick caught the MTP sidecar's abort earlier in the day; it is worth making a
habit of.)

## The tools left behind

- `draft-probe.py` — acceptance by prompt predictability. The one number that separates "weak drafter" from "broken
  plumbing"; run it before theorising about either.
- `sweep-dflash.sh` — the whole draft-length curve from ONE load, via `LLAMA_SPEC_DRAFT_N_FILE`
  (`patches/dflash-runtime-draft-n.patch`, built into `build-dflashn`; source reverted). Each draft length costs bytes:
  every extra verify row lights up ~8 more of the 384 experts per layer, while attention, the router and the output head
  are read once per cycle regardless. So the optimum is somewhere below block_size, and that is what the sweep finds.
- `sweep-requant.sh`, `after-mtp3.sh`, `chain-to-165.sh`, `chain-dflash-165.sh` — unattended drivers that gate on
  tokens/cycle before spending another 30-minute load on a bytes lever.
- `spec-detail.py` — splits a rate into verify-cycle rate and tokens accepted per cycle, which is what says whether a
  disappointing number is bandwidth or acceptance.

## Operating notes for this model

- A tensor-parallel load reads 650-700 GiB (more than the 518 GiB model: page cache is evicted and re-read under
  pressure) and takes **29-30 minutes** with `/health` returning 503 throughout. Watch `read_bytes` in `/proc/PID/io`.
- Teardown frees the port at once but takes a minute or two to release the memory. Wait for `free -g` available to come
  back above ~600 GiB before launching the next configuration.
- Stop servers **by listening port** (`ss -ltnpH "sport = :18190"`), never by process name: other sessions run
  llama-servers on this box.
- The only Python here with torch + safetensors + transformers is
  `engines/llama.cpp-dsv41-tuned-0912/.venv/bin/python`.

## If bytes run out, the next lever is the duty cycle

MiMo decodes at **73% of this host's measured 381.6 GB/s** (7.94 tok/s x 35.0 GB/token = 278 GB/s), against
GLM-5.3-Flash's 61%. That is already good, so there is less op-level slack here than there was on Flash — but 27% is
still 27%, and it is the only lever left once attention is requantised and the draft length is optimal (the experts are
MXFP4 at 4.25 bits and cannot go lower). The same tooling works: this fork has `GGML_CPU_OP_PROFILE` and
`GGML_CPU_OP_PROFILE_ARM_FILE` (`ggml/src/ggml-cpu/ggml-cpu.c`), and the aggregation method that matters is to filter to
ONE `cpu=` and ONE `graph=` id before summing — every socket emits the same mirrored graph, and the verify and draft
graphs are separate. Summing across either inflates everything several-fold.

One untried knob that costs nothing: `PMIN` on the launcher (`--spec-draft-p-min`, currently 0.0, so the full block is
always verified). If acceptance falls off sharply with position in the block, a non-zero p_min trims the tail rows
adaptively and saves expert bandwidth on exactly the cycles where the draft was going to be rejected anyway.

### The cheap-canary discipline

Three defects, three 25-30 minute target loads, because each one surfaces later in startup than the last: tensor mapping
(draft load, after the target), then graph construction (first decode). Two habits make that much cheaper:

1. **Load the draft model standalone first** (0.3 s) — catches every tensor-mapping problem.
2. **Send an 8-token canary as the first request** and check the process is still alive before starting a measurement —
   catches every graph-construction problem for the price of one short decode instead of a whole measurement script
   failing request by request against a dead socket.

`/health` returning ok proves only that the weights loaded. It says nothing about whether a graph can be built.

### Leave the patches APPLIED while the campaign is running

I reverted `dflash.cpp` and `common/speculative.cpp` to keep this tree clean, then later rebuilt `build-dflashn` for the
`mimo2.cpp` change — and that rebuild recompiled the reverted files, silently dropping both the attention-sink fix and
the runtime draft-length override. The next load spent 18 minutes reproducing `expected 63, got 58`.

A build directory is not a snapshot: it recompiles whatever is in the working tree. So while a patched build is in use,
keep the patches applied and revert only when the campaign ends — or re-apply them and rebuild before every launch.
Re-applying from `patches/` is one command each and `git apply --check` says immediately whether a patch is already in:

    for p in patches/dflash-qwen3-attn-sinks.patch patches/mimo2-export-layer-inputs.patch patches/dflash-runtime-draft-n.patch; do
      git apply --check "$p" 2>/dev/null && git apply "$p" && echo "applied $p"
    done

Cheap confirmation that a build really carries a change, before spending half an hour on a load:

    strings build-dflashn/bin/libllama-common.so.0 | grep -c LLAMA_SPEC_DRAFT_N_FILE   # 0 means it is not in there

### A 1-minute test loop for graph changes: the 4-layer proxy with a retargeted drafter

Defects 3 and 4 both live in graph construction, which only runs on the first decode — 25 minutes of target load away.
`proxy/mimo-v26-proxy-4L.gguf` is the same `mimo2` architecture with `n_layer() = 4`, and it loads in about a minute.
The drafter cannot be pointed at it as shipped, because `llama_context::set_embeddings_layer_inp` asserts
`lid <= n_layer()` and DFlash asks for layers up to 70 — but that is metadata, not weights. Rewriting one GGUF key gives
a drafter that drives the proxy:

    dflash.target_layers  [1, 16, 32, 48, 70]  ->  [0, 1, 2, 3, 4]

Five layers either way, so `fc.weight [30720, 6144]` still matches `n_embd_inp_enc = 5 * n_embd`. Result:
`gguf/MiMo-DFlash-proxytest.gguf`. Launch:

    BUILD=build-dflashn SPEC=dflash NMAX=4 PORT=18198 CTX=2048 \
      MODEL=/models/mimo-v26-pro/proxy/mimo-v26-proxy-4L.gguf \
      DFLASH_MODEL=/models/mimo-v26-pro/gguf/MiMo-DFlash-proxytest.gguf ./launch-mimo-tp.sh

The text it produces is garbage — a 4-layer truncation of a 70-layer model — and that is fine. It cannot measure
acceptance or speed (the proxy exaggerates per-graph costs and its draft acceptance is meaningless), but it exercises
exactly the same graph construction, allocation and scheduling path, so it answers "does this abort?" in one minute
instead of twenty-five. Use it for every graph change before spending a real load.

**Defect 4, found that way:** exporting `t_layer_inp[n_layer]` at full width meant disabling the last layer's early
out-ids crop — which left `inp_out_ids` referenced by no node at all, so the allocator never gave it a buffer and
`llm_graph_input_out_ids::set_input` aborted in `ggml_backend_buffer_is_host(null)`. The graph must filter by out-ids
exactly once; moving the crop below the export restores that. Also: exporting an *alias* of `inpL` is not enough under
the Meta backend (that node belongs to a device subgraph and has no buffer of its own) — export a dedicated
`ggml_cont` node and expand it, the way glm5next does.

## Defect 5, and the one that actually mattered: the mask token's embedding is EMPTY in the target

DFlash builds its draft block as `[last_token, <mask> * (block_size-1)]` and takes the embeddings for those ids from the
**target** model's table -- `dflash.py` line 319: `noise_embedding = target.model.embed_tokens(block_output_ids)`.

The base checkpoint's row for that token is empty. Measured against shard 1's `token_embd.weight` row 151675:

| | norm |
|---|---:|
| `dflash/mask_embedding.pt` (shipped separately) | **1.8617** |
| `token_embd.weight[151675]` in the GGUF | **0.0008** |

cosine between them: **-0.02**. That is why the repo ships `mask_embedding.pt` as its own file -- loading it into the
target's embedding table is part of the intended setup, not a workaround. Without it the drafter reads a **zero vector**
at seven of every eight block positions, and acceptance collapsed to 0.014-0.072 (worse than the MTP heads, and the
resulting 3.1-4.1 tok/s is well under the 7.94 no-speculation baseline).

`patch-mask-embedding.py` writes it in place: one row of a Q8_0 [6144 x 152576] tensor is 192 blocks x 34 bytes = 6528
bytes at a known offset inside the 42 GiB shard, so no re-conversion and no 42 GiB copy. The original bytes go to
`gguf/token_embd_mask_row.orig` first and `--restore` puts them back. Q8_0 round-trip error 1.06%; after writing, the row
reads back at norm 1.8618, cosine 0.999944 to the shipped vector. The target never generates that token, so nothing else
about the model changes.

**The general lesson:** when a drafter ships extra files beside its weights, they are not optional. `dflash/` contains
`config.json`, `dflash.py`, `model.safetensors.index.json`, the weights -- and `mask_embedding.pt`. A conversion that
reads only the safetensors silently drops the last one, and the failure looks exactly like a bad drafter rather than a
missing file.

### A speculative server can hang on shutdown holding all of RAM

The DFlash run did not exit on SIGTERM. It stopped listening immediately, then sat in `futex_wait_queue` with RSS flat
at 523 GiB for minutes — blocking the next load, which needs the same memory. The non-speculative runs all released
within a minute or two, so the extra context appears to be what hangs.

So: after stopping by port, **watch RSS actually fall**, and escalate if it does not.

    p=$(ss -ltnpH "sport = :18190" | grep -o 'pid=[0-9]*' | cut -d= -f2)   # capture BEFORE the kill
    kill "$p"; sleep 30
    awk '/^VmRSS/{print $2}' /proc/$p/status    # sample twice: flat means hung, not draining

Before escalating to `kill -9`, confirm the process is yours — `readlink /proc/$p/exe` and the `--model` argument — and
that it is no longer listening. Other sessions run llama-servers on this box, which is why the standing rule is to stop
by listening port; escalating by PID is only safe once you have identified that exact process.

## Defect 6: the drafter's rope dimension count was never emitted

`dflash/config.json` sets `partial_rotary_factor: 0.5` on `head_dim: 128` — only **64 of 128 dimensions carry
positional information**. Nothing in the converter path emits `rope.dimension_count` for `DFlashDraftModel`, and
`llama-model.cpp` then defaults `n_rot_full` to the full `n_embd_head_k`, so llama.cpp roped **all 128**.

Every position in the drafter was therefore encoded wrong while the target stayed untouched — which presents exactly as
"the drafter is weak" rather than as a bug. `patches/dflash-converter-rope-dims-and-sinks.patch` adds:

    head_dim = hparams["head_dim"] or hidden_size // num_attention_heads
    rope_dim = int(head_dim * hparams.get("partial_rotary_factor", 1.0))      # 128 * 0.5 = 64
    gguf_writer.add_rope_dimension_count(rope_dim)

(The same patch carries the `.attn_sinks` -> `.attn_sinks.weight` rename, so it supersedes the sink-only patch.)

**The method that found it, after two rounds of guessing failed:** dump EVERY key in the produced GGUF next to EVERY key
in the source `config.json` and read the difference, instead of forming a hypothesis about which one is wrong. Two keys
were missing; `attention_value_scale: 0.612` turned out to be a false alarm (the reference `Qwen3DFlashAttention.forward`
never applies it, and `dflash.cpp` has no value-scale path), but `partial_rotary_factor` was real.

Worth knowing about the reference file: `dflash.py`'s attention computes K and V from BOTH the projected target hidden
states and the noise block, concatenates them, and attends over `ctx_len + q_len` with `is_causal = False`. It also does
NOT use the attention sink bias, even though the checkpoint ships `self_attn.attention_sink_bias` per layer and
`config.json` sets `add_swa_attention_sink_bias: true`. The shipped `dflash.py` looks like a simplified export, so the
sinks are loaded and applied here on the strength of the weights and the config; if acceptance is still short after the
rope fix, that is the next thing to A/B.

## The production setting: `--spec-draft-n-max 7 --spec-draft-p-min 0.5`

Measured surface, one load, `LLAMA_SPEC_DRAFT_PMIN_FILE` sweeping p_min against draft length on an open-prose prompt and
a code prompt (`sweep-pmin.sh`, results in `results/dflash-pmin-sweep.md`):

| prompt (n=7) | p_min 0.0 | 0.3 | **0.5** | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| open prose | 6.23 | 7.73 | **8.80** | 7.91 | 7.69 |
| code | 13.45 | 13.36 | 13.40 | 13.07 | 14.01 |

p_min 0.5 lifts the worst case **41%** (6.23 -> 8.80, acceptance 0.203 -> 0.727) for about 6% on code, and code is flat
across the whole surface (13.07-14.21), so there is nothing to lose. Pushing further costs more in tokens per cycle than
it saves in bytes: at p_min 0.9 acceptance is 1.000 but tokens/cycle falls to 1.31 and prose drops back to 7.69.

**A fixed draft length is the wrong control.** At p_min 0 the n-sweep is monotone the wrong way (n=7 6.28 down to
n=2 8.04) purely because a rejected row still costs a full row of expert bandwidth in the verify pass. p_min fixes that
per request instead of per launch: full block when the drafter is confident, truncated the instant it is not.

## Final numbers, and the Q4_K trade

Two configurations, both measured on the same prompts. **The difference between them is a quality decision, not a
tuning one.**

### Exact — output byte-identical to non-speculative decoding
`SPEC=dflash NMAX=7` with `--spec-draft-p-min 0.5`. `golden.py --compare` against the non-speculative golden:
IDENTICAL tokens on all three prompts, max |dlogprob(top1)| **0.0000**.

| workload | tok/s | acceptance |
|---|---:|---:|
| counting / verbatim repetition | **19.2 - 20.4** | 1.000 |
| code | 13.4 | 0.67 |
| memorised list | 11.7 | 0.53 |
| open prose | 8.8 | 0.73 |
| *(no speculation, flat)* | *7.94* | — |

### Q4_K attention — faster, but it CHANGES OUTPUT
`GGML_CPU_ATTN_REQUANT=q4_K`, best setting `NMAX=4`, `p_min 0.0`. **The quality gate FAILED**: prompt 1 diverges from
the Q8_0 golden at token 11 of 48, max |dlogprob(top1)| 0.181 (prompts 0 and 2 stayed identical at 0.0102 / 0.0031).
That is a genuine output change, not rounding.

| workload | tok/s | acceptance |
|---|---:|---:|
| counting / verbatim repetition | **21.3 - 22.9** | 1.000 |
| code | **15.3** | 0.75 |
| memorised list | 12.6 | 0.67 |
| open prose | 10.0 | 0.39 |

Note that with the cheaper cycle, p_min matters much less — a wasted row costs less, so `n=4 p_min 0.0` beats the
carefully tuned `n=7 p_min 0.5` that was optimal at Q8_0. Retune p_min whenever the cycle cost changes.

### What this means for a target number

**There is no single "MiMo tok/s".** The rate spans 2.4x across workloads because DFlash drafts a whole block: it
accepts 100% of eight tokens on predictable text and ~40-60% on open prose. Quoting one figure without the workload is
meaningless for this model in a way it is not for GLM-5.3-Flash.

16.5 tok/s is **reached on predictable and structured text** (19-23) and **not reached on code** (13.4 exact, 15.3 with
the quality trade) or prose (8.8 / 10.0). The remaining gap is not recoverable by tuning: the fitted cycle model puts
this host at **79% of its measured 381.6 GB/s memory wall**, the experts are already MXFP4 at 4.25 bits, and everything
else in the cycle is read once. Only tokens-per-cycle is left, and that is a property of the drafter.

## Two limits found while verifying the production config

### 1. The 256K context is a memory figure, not a usable one

> **Update added at publication (2026-09-24).** The cause of this depth loss was found the same evening: for
> query batches of 2-63 rows (every speculative verify) the CPU flash-attention kernel streamed the whole KV
> range once per query row and Q head, and summed V in FP16. A grouped split-KV kernel fixed both: decode at
> 64K rose from 1.98 to 6.36 tok/s and at 16K from 5.29 to 8.56, with the verify cycle nearly flat from 4K to
> 64K. See [STATE-FA-GQA-20260922.md](STATE-FA-GQA-20260922.md).

`depth-probe.py`, one append-only session, 2026-09-22:

| depth | new tokens prefilled | prefill tok/s | **decode tok/s** | draft acceptance |
|---:|---:|---:|---:|---:|
| 4,057 | 4,067 | 35.5 | **7.68** | 0.528 |
| 16,201 | 12,155 | 31.4 | **5.29** | 0.477 |
| 65,437 | 49,247 | 21.0 | **1.98** | 0.429 |

Decode falls **3.9x** between 4K and 64K. Only 10 of 73 layers keep full KV, and the bytes those add come to roughly
1.3x of the per-token read at 64K — so most of the loss is not KV bandwidth. Attention at depth costs more than its
bytes, and draft acceptance decays as well (0.53 -> 0.43), which multiplies into the rate a second time.

The server keeps `CTX=262144` because a larger window costs only KV memory and lets long documents through when the
speed is acceptable. But **32K is the practical ceiling for interactive work**, which is what the Codex wrapper is now
set to. Prefill is the other half of the same story: 49K new tokens took 39 minutes.

### 2. Vision is broken above ~128x128, and the boundary is exact

> **Correction added at publication (2026-09-24).** The diagnosis in this section is wrong. Flat test images never
> produce the outlier activations that break the vision tower, so the size boundary only looked exact; a
> textured 128x128 image fails as well. The cause was F16 overflow in the last ViT block (a SwiGLU output of
> ~1.1e5 against F16's 65504), fixed with an F32 projector; the windowed-attention path was correct. See
> [STATE-VISION-20260922.md](STATE-VISION-20260922.md). The audio concern at the end of this section did not
> materialise either: audio transcribes a known clip 12/12 words ([audio.txt](results/deploy-0922b/audio.txt)).

With `MMPROJ=1` the projector loads and requests complete, but the answer is `?` repeated. It is **size-dependent**,
which is far more useful than "broken" (`vision-probe.py`):

| image | patch grid | after merge | result |
|---|---|---|---|
| 64x64 | 4x4 | 2x2 = 4 tok | **'Red'** correct |
| 96x96 | 6x6 | 3x3 = 9 tok | **'Red'** correct |
| 128x128 | 8x8 | 4x4 = 16 tok | **'Red'** correct |
| 160x160 | 10x10 | 5x5 = 25 tok | `'????...'` |
| 192x192 and above | | | `'????...'` |

Text-only controls answer correctly with thinking both on and off, so the language model, chat template and sampler are
all fine — this is the vision tower or its projector.

The suspect is the windowed-attention path. `clip.cpp` computes
`grid_window = attn_window_size / patch_size / merge_ratio` = 64 / 16 / 2 = **2 merged tokens per window**, and the
boundary sits exactly between a 4x4 merged grid (2 windows per axis, works) and 5x5 (3 windows per axis, fails). The
per-layer `wa_pattern_mode` is `[-1, 0, 0, 0, 0, 1, 1, 1, 1, ...]` repeating over 28 blocks, and `v.blk.N.attn_sinks`
is present for 24 of 28 blocks — which is consistent with the 4 `-1` (full-attention) layers legitimately having no
sink, matching the text model's `add_swa_attention_sink_bias: true` / `add_full_attention_sink_bias: false`. Do not
"fix" that tensor count before checking the pattern.

**Workaround until it is fixed: downscale images to 128x128 or smaller before sending.** Low resolution, but correct
answers instead of garbage.

Related, and the same class as the mask-token bug: `token_embd[151669]`, the AUDIO placeholder, is empty (norm 0.0008)
where the image placeholder 151655 is a real vector (0.73). Expect the audio path to fail the same way, and look for a
separately shipped embedding as `mask_embedding.pt` was shipped for DFlash.
