# MiMo-V2.6-Pro-RL on smeagol (2026-09-21)

> **How to run it (2026-09-23).** `systemctl --user start mimo-v26-pro.service`, or
> `./launch-mimo-production.sh` in the foreground. Pinned build `engines/llama.cpp-mimo-tp/build-prod-0922j`:
> grouped-query flash attention (`GGML_CPU_FA_GQA=2`), the bit-identical multi-column VNNI GEMM (`GGML_CPU_X16_GEMM=1`)
> -- since 0922i running 16-row groups in pairs --, the bit-identical MoE weighted-sum fusion
> (`GGML_CPU_MOE_WEIGHTED_SUM_FUSION=2`), 1024-token prompt micro-batches, the XML tool-call parser fix, exact
> attention weights, MiMo's own DFlash drafter at `n_max 7 / p_min 0.5`, 256K context, a 32 GiB host-RAM prompt
> cache, the **F32** vision projector and a 1280-token cap per image. Every value is justified in the script's
> header. Vision, audio and tool calls verified on every deploy (`verify-deploy-0922*.sh`, `results/deploy-0922*/`).
> Read **`STATE-FA-GQA-20260922.md`** (kernels), **`STATE-VISION-20260922.md`** (vision) and
> **`STATE-SPECULATION-20260922.md`** (drafter) before changing any of it.
>
> Measured on a quiet box (`results/deploy-0922b/`, `results/deploy-0922c/`): **22.8 tok/s verbatim, 26.9 counting,
> 16.3 code, 13.2 memorised list, 8.8 open prose** (09-22 morning: 19.2 / 20.4 / 13.4 / 11.7 / 8.8); prefill
> **60-69 tok/s** at short context (was 35-39); depth curve decode 18.5 / 8.6 / 9.3 / **6.4** tok/s at 4K / 16K / 32K /
> 64K (was 7.7 / 5.3 / - / 2.0). Smoke 7/7; real-image OCR 4/4; audio 12/12 words; tool calls 0/20 malformed.
> **0922j is LIVE since 09-23 07:31** (full-model golden identical to 0922g; tool calls 0/20, smoke 7/7, vision 4/4,
> audio 12/12; `results/deploy-0922j/`). An 8K-token prompt prefilled at 61.5 tok/s with other sessions' jobs on every
> node (load 66). Builds 0922d-0922j are faster again with output bit-identical to 0922c: long-context attention ~2x over 0922c
> (0922f 1.2-1.5x, 0922j another 1.25-1.4x), prompt matmuls 1.2-1.4x single-core; 4-layer proxy prefill 0922g -> 0922j
> 873-889 -> 1050-1128 tok/s. Not re-measured on the full model: other sessions' jobs have held every NUMA node since
> 09-22 evening.
>
> **If the service sits in "activating (start-pre)"**, it is waiting for memory, not broken: `wait-for-ram.sh`
> needs 575 GiB host-wide AND 150 GiB free/reclaimable on EVERY NUMA node (MiMo binds ~136 GiB of weights to each).
> `journalctl --user -u mimo-v26-pro` names the short node and the largest processes. A reload that loses its node
> mid-way is killed by `load-watchdog.sh` within ~20 s instead of being OOM-killed 30 minutes later (09-23 05:58).
> The spread across workloads is inherent -- DFlash drafts whole 8-token blocks -- so quote the workload with the
> number. Other sessions' jobs pinned to MiMo's NUMA nodes cost ~30% while they run: check `uptime` before measuring.
>
> Output is no longer byte-identical to the 09-22 morning build: the old flash-attention kernel summed V in FP16
> (rel.err 1e-3 at short context, 4e-2 at 64K); the new one is F32 (1e-7). Golden: 2 of 3 prompts token-identical,
> prompt 1 flips at token 11, the same near-tie Q4_K attention flipped.
>
> Two traps worth knowing before you start: `--spec-type draft-mtp` is a **dead end** (the three `model.mtp.layers.*`
> heads are a training artifact, not the drafter: 4-16% acceptance, slower than no speculation), and the service is
> **exclusive with GLM-5.3-Flash and GLM-5.3 Full** on RAM, which is why its unit carries `Conflicts=`.
>
> Reachable as `mimo`, `mimo-v2.6-pro` or `MiMo-V2.6-Pro-RL` from the harnesses (route + Codex catalog + Paseo
> provider lists, all added 09-22; backups alongside each file).

Xiaomi's flagship, published on HF 2026-09-21 ~20:12 UTC: 1.02T-param sparse MoE / 42B active, 70 layers
(60 SWA window 128 + 10 global), 384 routed experts (8 active), 1M context, 3 MTP layers, vision + audio encoders.
Checkpoint: `XiaomiMiMo/MiMo-V2.6-Pro-RL` @ `54b10491b1811c76aa9681a9d0ff872396a4064c` (573 GB, 534 GiB):
routed experts MXFP4 (QAT'd; U8 codes + U8 E8M0 scales), attention FP8 e4m3 128x128 blocks, o_proj/embeddings BF16.

## What is on disk

| path | what |
|---|---|
| `/models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE-000NN-of-00013.gguf` | 556.2 GB / 518 GiB. Experts bit-exact MXFP4, everything else Q8_0 (FP8/BF16 -> Q8_0), norms/router F32. 3 nextn (MTP) layers included. |
| `/models/mimo-v26-pro/gguf/mmproj-MiMo-V2.6-Pro-RL-F32.gguf` | **5.45 GB, the one production uses**: vision (28-block MiMo ViT) + audio encoder, F32 from the BF16 source (09-22) |
| `/models/mimo-v26-pro/gguf/mmproj-MiMo-V2.6-Pro-RL-F16.gguf` | 2.8 GB, same content in F16 -- **broken on CPU**: ViT block 27 overflows F16 -> NaN -> `????` (STATE-VISION-20260922.md) |
| `/models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-DFlash-Q8_0.gguf` | 2.9 GB, MiMo's own DFlash drafter (STATE-SPECULATION-20260922.md) |
| `/models/mimo-v26-pro/src/` | non-expert source shards (38 GiB) -- only needed to rebuild; safe to delete |
| `hf-manifest-54b10491.json` | sha256 + size of every file in the HF revision |

No GGUF of V2.6 existed anywhere at conversion time, and upstream llama.cpp could not convert it: the `mimo2`
runtime already runs the architecture (the text backbone config is identical to V2.5-Pro), but `conversion/mimo.py`
only dequantizes FP8 `*_scale_inv` tensors -- the MXFP4 expert pairs (`.weight` U8 + `.weight_scale` U8) would have
been cast to float32 codes. `convert/` is a snapshot of upstream's converter (identical to master 09-21) with:

- `MXFP4ExpertStack` + `MimoV2Model._write_mxfp4_experts`: experts leave `model_tensors` before the FP8 path and the
  f32 cast, and are written raw as ggml MXFP4 through the existing lossless `ModelBase.repack_mxfp4_blocks` shuffle.
- `MIMO_MXFP4_SKELETON=1`: reserve each stacked expert tensor as a sparse hole instead of reading it.
- mmproj: skip the audio tokenizer's `decoder.*` (305M-param speech decoder + vocoder; input needs the encoder only).

Nibble order was established on real weights before converting: with element 2i in the low nibble, per-neuron
`|up_i|` vs `|down_:,i|` correlates +0.54..+0.95 (SwiGLU + weight decay balance them); swapped, ~0.00. Every
32-block peaks at code 6 or 7, as MXFP4 quantization produces. gguf-py's MXFP4 dequant of the repacked bytes equals
an independent E2M1 x 2^(e-127) reference exactly.

## How it was built (disk-bounded: the 534 GiB checkpoint never existed on disk at once)

1. `fetch-nonexpert.sh` -- config/tokenizer + `ep0_shard0` (all non-expert text tensors + experts 0-2), `ep0_shard1`
   (vision/audio), `model_mtp`, `audio_tokenizer`; sha256-verified. 38 GiB.
2. `run-skeleton.sh` -- patched converter over a dir without the index (only the local shards get indexed):
   writes the 13 splits with every non-expert tensor and 207 sparse holes. 12 min, 24 GiB real.
3. `run-fill.sh` (`fill_experts.py`) -- for each of the 128 `ep` shards: download to the NVMe root disk, sha256 vs the
   manifest, repack its 621 expert tensors, `pwrite` each at `tensor_offset + expert * 6,684,672`, fdatasync, read a
   random one back (bit-exact + dequant check), mark `fill-state/<shard>.done`, delete the shard. Resumable.
   `fill_experts.py --verify` = every shard done + no hole left in any split (SEEK_HOLE scan).
4. mmproj: `convert_hf_to_gguf.py /models/mimo-v26-pro/skel-src --mmproj --outtype f16`.

All CPU work ran in the background pen (`taskset 15,31,47,63,79,95,111,127`, nice/ionice) next to the live
GLM-5.3-Flash server.

## Running it -- RAM-EXCLUSIVE

556 GB of weights + KV do not fit beside GLM-5.3-Flash (282 GiB anon) or even beside its 186 GiB tmpfs payload
alone. Running MiMo means: `glm53-flash-production` stopped AND the Flash tmpfs payload evicted.

- `window-mimo-smoke.sh` -- the whole round trip, Flash restored by an EXIT trap: preflight (fill verified + Flash
  payload matches its manifest) -> wait for Flash idle -> `systemctl --user stop glm53-flash-production` -> delete the
  6 tmpfs shards (mount kept) -> pre-warm the GGUF into page cache (interleaved) -> `launch-mimo-v26.sh` -> 
  `smoke-mimo.py` (plain, then MTP) -> `restage_flash_q4.py` -> `systemctl --user start glm53-flash-production`.
- `launch-mimo-v26.sh` -- upstream llama.cpp master (`engines/llama.cpp-mimo-v26`, `build-mimo.sh`), port 18190,
  60 worker cores, `numactl --interleave=all`, `oom_score_adj 1000`; `SPEC=mtp` adds `--spec-type draft-mtp`
  (the nextn layers inside the GGUF; no separate draft file).
- `restage_flash_q4.py` -- re-downloads the Flash Q4 payload into its existing tmpfs from the pinned revision
  (`unsloth/GLM-5.3-Flash-GGUF` @ `621d456e`, sha256 per shard). Use it instead of
  `fleet-0903/stage_flash_q4_0910.py`, which asserts a 09-10 Qwen peer pid and no longer runs. Run it under
  `numactl --interleave=all` (fixes the node-3 first-touch skew). Needs the mount to exist: remounting needs sudo.

Flash is a systemd user unit now: always stop/start it with `systemctl --user ... glm53-flash-production.service`.
`restart-native.sh` would launch it WITHOUT the unit's 09-19/09-20 drop-in libraries and env (silent config drift).
The unit's `ConditionPathExists` on tmpfs shard 1 keeps it from starting while the payload is evicted.

## Space made for it (2026-09-21)

- GLM-5.3 Full UD-Q4_K_XL shards deleted (435 GiB). Its local-only `MTP/GLM-5.3-MTP-UD-Q4_K_XL.gguf` was kept.
  Restore: `serving/glm-sr950/download-glm53-q4.sh` (manifest `glm53-hf-tree-7837428d.json`).
- DeepSeek-V4-Flash-0731 UD-Q4_K_XL deleted (145 GiB). Restore note: `serving/dsv4-sr950/DSV4-FLASH-0731-DELETED-20260921.md`.
- Both manifests also copied to `/models/ai-server-recovery-rail/`.

## Results

(see below, appended by the smoke window)
