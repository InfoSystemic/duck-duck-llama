#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pyyaml>=6"]
# ///
"""Extract a self-contained MiMo-V2.6 NextN/MTP draft model from sharded GGUF.

Adapted 2026-09-21 from the GLM-5.3-Flash extractor (duck-duck-llama
engineering/2026-09-08/archive/serving/glm53-flash/extract-glm5next-mtp-gguf.py). Two differences: MiMo declares
mimo2.nextn_predict_layers = 3, so all three trained heads are kept rather than one, and the architecture check
accepts mimo2. The output keeps the source metadata and tokenizer and carries only those blocks plus the shared
token embedding, final norm and LM head. Unsharded so the draft lands on one NUMA backend.

Speculative decoding needs this as a separate file: draft-mtp requires a non-null draft context, and the driver
auto-detects MTP by finding blk.<block_count-1>.nextn.eh_proj.weight in the draft GGUF.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path


# gguf-py: prefer an engine tree beside this workspace, else rely on PYTHONPATH / an installed gguf
for _cand in (Path(__file__).resolve().parents[2] / "engines" / "llama.cpp-mimo-tp" / "gguf-py",
              Path(__file__).resolve().parents[2] / "engines" / "llama.cpp-mimo-v26" / "gguf-py"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand)); break

import gguf  # noqa: E402


LOG = logging.getLogger("extract-glm-mtp-gguf")
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
GLOBAL_TENSORS = {"token_embd.weight", "output_norm.weight", "output.weight"}
SPLIT_KEYS = {
    gguf.Keys.Split.LLM_KV_SPLIT_NO,
    gguf.Keys.Split.LLM_KV_SPLIT_COUNT,
    gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT,
}


def field_contents(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field else None


def source_shards(first: Path) -> list[Path]:
    match = SHARD_RE.match(first.name)
    if not match:
        return [first]

    count = int(match.group("count"))
    stem = match.group("stem")
    shards = [first.with_name(f"{stem}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source shard(s): " + ", ".join(missing))
    return shards


def wanted_tensor(name: str, mtp_layers) -> bool:
    if name in GLOBAL_TENSORS:
        return True
    return any(name.startswith(f"blk.{i}.") for i in mtp_layers)


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, output_name: str) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in SPLIT_KEYS:
            continue
        if field.name == gguf.Keys.General.NAME:
            continue

        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)

    writer.add_string(gguf.Keys.General.NAME, output_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="first source GGUF shard")
    parser.add_argument("output", type=Path, help="unsharded MTP-only GGUF to create")
    parser.add_argument(
        "--globals-from",
        type=Path,
        help="GGUF providing token_embd.weight, output_norm.weight, and output.weight",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    source = args.source.resolve()
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")

    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")

    layer_shards = source_shards(source)
    global_shards = source_shards(args.globals_from.resolve()) if args.globals_from else layer_shards
    shards = list(dict.fromkeys([*layer_shards, *global_shards]))
    # Keep the readers alive until the writer has consumed metadata and tensor
    # views. GGUFReader uses mmap, so mapping all shards does not read all model
    # weights into RAM.
    readers = {shard: gguf.GGUFReader(shard, "r") for shard in shards}
    metadata_reader = readers[layer_shards[0]]
    architecture = field_contents(metadata_reader, gguf.Keys.General.ARCHITECTURE)
    block_count = int(field_contents(metadata_reader, f"{architecture}.block_count"))
    nextn_count = int(field_contents(metadata_reader, f"{architecture}.nextn_predict_layers") or 0)
    if architecture != "mimo2" or nextn_count < 1 or block_count <= nextn_count:
        raise ValueError(
            f"Expected mimo2 with at least one NextN layer; got architecture={architecture!r}, "
            f"block_count={block_count}, nextn_predict_layers={nextn_count}"
        )
    # MiMo ships nextn_count trained heads, one per draft step, in the LAST blocks. The speculative driver reads
    # llama_model_n_layer_nextn() off this file and enters its chain_heads mode when that is > 1, so keep them all.
    mtp_layers = list(range(block_count - nextn_count, block_count))
    mtp_layer = mtp_layers[-1]

    writer = gguf.GGUFWriter(partial, arch=architecture, endianess=metadata_reader.endianess)
    alignment = field_contents(metadata_reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = int(alignment)
    copy_metadata(metadata_reader, writer, f"{field_contents(metadata_reader, gguf.Keys.General.NAME)}-MTP")

    selected: list[tuple[Path, object]] = []
    selected_names: set[str] = set()
    total_bytes = 0

    def select_tensors(source_paths: list[Path], predicate) -> None:
        nonlocal total_bytes
        for shard in source_paths:
            reader = readers[shard]
            for tensor in reader.tensors:
                if not predicate(tensor.name):
                    continue
                if tensor.name in selected_names:
                    raise ValueError(f"Duplicate selected tensor: {tensor.name}")
                writer.add_tensor_info(
                    tensor.name,
                    tensor.data.shape,
                    tensor.data.dtype,
                    tensor.data.nbytes,
                    tensor.tensor_type,
                )
                selected.append((shard, tensor))
                selected_names.add(tensor.name)
                total_bytes += tensor.n_bytes

    if args.globals_from:
        select_tensors(layer_shards, lambda name: any(name.startswith(f"blk.{i}.") for i in mtp_layers))
        select_tensors(global_shards, lambda name: name in GLOBAL_TENSORS)
    else:
        # Preserve the original source traversal and tensor ordering.
        select_tensors(layer_shards, lambda name: wanted_tensor(name, mtp_layers))

    found = selected_names
    required = {"token_embd.weight", "output_norm.weight", f"blk.{mtp_layer}.nextn.eh_proj.weight"}
    missing = required - found
    if missing:
        raise ValueError("Required MTP tensor(s) missing: " + ", ".join(sorted(missing)))

    LOG.info(
        "Selected %d tensors (%.3f GiB) from %d source shard(s)",
        len(selected),
        total_bytes / 2**30,
        len(shards),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    written = 0
    for shard, tensor in selected:
        writer.write_tensor_data(tensor.data, tensor_endianess=readers[shard].endianess)
        written += tensor.n_bytes
        LOG.info("[%5.1f%%] %s", 100.0 * written / total_bytes, tensor.name)

    writer.close()
    os.replace(partial, output)
    LOG.info("Created %s (%.3f GiB)", output, output.stat().st_size / 2**30)


if __name__ == "__main__":
    main()
