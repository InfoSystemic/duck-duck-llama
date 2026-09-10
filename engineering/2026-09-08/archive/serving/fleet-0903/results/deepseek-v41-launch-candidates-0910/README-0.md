---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
  - deepseek
  - minirun
  - mlx
  - apple-silicon
  - macos
  - ios
---

# DeepSeek-V4.1-Flash-minirun

The weights of [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at
revision `df42c109f1defefcbfcedbe7d905718a12266e40`, repacked byte for byte into the container format
[Minirun](https://minirun.dev) reads. Minirun streams this model from an external SSD on
Mac and iPhone through a memory budget you set, rather than loading it into
memory, so the model does not have to fit in the machine. The reader is open
source: [`Sources/StorageCore/Container/`](https://github.com/nanguoyu/minirun-app/tree/main/Sources/StorageCore/Container) in
[`nanguoyu/minirun-app`](https://github.com/nanguoyu/minirun-app).

## What this is

A **byte-preserving repack**. No requantization, no retraining, no numerical
change of any kind:

- the routed expert weights are the upstream FP4 packed values and their
  `ue8m0` scales, copied verbatim and only reordered;
- the remaining quantized matrices are the upstream FP8 `e4m3` values and their
  `ue8m0` 32x32 block scales, copied verbatim;
- the two conditional-memory tables are the upstream FP8 `e4m3` rows and their
  own `ue8m0` scales, copied verbatim and regrouped so that a row and its
  scales sit together;
- all other tensors are the upstream BF16/F32 bytes, copied verbatim.

Every byte of weight data in this repository is a byte of the source
checkpoint at the pinned revision above, in a different order. The only bytes
that are not are container headers and zero padding: where a container's scale
region is shorter than its alignment unit, and at the tail of each row page.

Contents: 44 directories, 557 data files,
517.3 GB total, plus a per-directory manifest giving
each file's shape, element width and offsets.

## Layout

Three kinds of data file, all described completely by the per-directory
`manifest.json` beside them.

| Suffix | Holds | Addressed by |
| --- | --- | --- |
| `.mxfp4tile` | routed experts, two per tile | tile index; `expert_order` maps an expert id to its tile and half |
| `.fp8tile` | one FP8 matrix and its 32x32 scale grid | one tile |
| `.engrampage` | a slice of a conditional-memory table | row number |

An `.engrampage` file is the one worth describing here, because it is not a
matrix. The 2 conditional-memory tables hold
768,022,850 rows between them, stored as fixed
4096-byte pages of 15 rows each: every
row's 256 value bytes, then every row's
8 scale bytes, then zero padding. A row and its own
scales are therefore always inside one aligned 4096-byte read,
which the two separate upstream runs could not offer. The manifest gives the
page size, the rows per page, the offset of the scale region inside a page, and
which slice of the table each file covers, so a row's address is arithmetic:

```
part  = the file whose row_base <= row < row_base + rows
page  = (row - row_base) // rows_per_page
at    = first_page_offset + page * page_bytes
value = at + (row - row_base) % rows_per_page * row_weight_bytes
scale = at + scale_offset_in_page + (row - row_base) % rows_per_page * row_scale_bytes
```

## Source files

The model configuration, the tokenizer, the tokenizer-normalisation code and its
test vectors, the upstream card and the technical report are verbatim files from
the same pinned source revision. Every one of their byte counts, SHA-256
identities and original upstream paths is recorded in `index.json` under
`source_files`.

They are all repository root files, including
[`inference-config.json`](inference-config.json) — the argument file DeepSeek's own reference code
loads, upstream `inference/config.json`. A directory would have made them look
like undeclared model payload to anything that counts this repository's files
against what `index.json` declares, so the upstream directory became a name
prefix and the original path is in the table instead.

## Run it with Minirun

**Get the app.** On a Mac, download [`Minirun.dmg`](https://downloads.minirun.dev/Minirun.dmg). On an iPhone,
build it from [`nanguoyu/minirun-app`](https://github.com/nanguoyu/minirun-app); the README there has the
steps.

**Point it at this repository.** In **Settings → Storage**, use *Add a folder…*
to register a folder on an external NVMe drive. In **Settings → Models**, open
*Find Models*, select this repository and press *Download* -- or point Minirun
at a copy you already have. Run *Verify all files*.

**What Minirun does with it today.** It downloads this repository, verifies
every file against the published tree, and keeps it on the drive you chose.
Chat support for this model is in progress; this build has no reader for it
yet, and will say so rather than pretend.

**Requirements.** An Apple-silicon Mac on macOS 15 or later; an iPhone 15 Pro or
later on iOS 18 or later; an external NVMe drive with room for
517 GB.

[minirun.dev](https://minirun.dev) · [Docs](https://minirun.dev/docs) · [This model](https://minirun.dev/models/deepseek-v41-flash) ·
[GitHub](https://github.com/nanguoyu/minirun-app)

## Provenance

| | |
| --- | --- |
| Source model | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Source revision | `df42c109f1defefcbfcedbe7d905718a12266e40` |
| Relationship | byte-preserving repack (no requantization) |

## License

This repository redistributes model weights owned by DeepSeek under the **MIT
License**, reproduced verbatim in [`LICENSE`](LICENSE) and copied unmodified
from the source repository at the pinned revision above.

Copyright (c) 2023 DeepSeek.

The MIT License permits use, copying, modification and redistribution,
including commercially, provided the copyright notice and the permission
notice are included in all copies or substantial portions of the Software.
The weights are provided "as is", without warranty of any kind. Refer to
[`LICENSE`](LICENSE) for the governing text; the summary above is not a
substitute for it.
