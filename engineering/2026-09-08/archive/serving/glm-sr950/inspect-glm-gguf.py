#!/usr/bin/env python3
"""Inspect a sharded GLM GGUF without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AI_SERVER = Path(__file__).resolve().parents[2]
GGUF_PY = AI_SERVER / "engines" / "llama.cpp-sr950-glm" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

try:
    from gguf.gguf_reader import GGUFReader  # noqa: E402
except ModuleNotFoundError as error:
    fallback_python = AI_SERVER / "serving" / "litellm-proxy-venv" / "bin" / "python3"
    if error.name == "numpy" and fallback_python.is_file() and Path(sys.executable) != fallback_python:
        os.execv(str(fallback_python), [str(fallback_python), *sys.argv])
    raise


SHARD_RE = re.compile(r"^(?P<prefix>.+)-00001-of-(?P<count>[0-9]{5})[.]gguf$")
INTERESTING_METADATA = re.compile(
    r"^(general[.](architecture|name|file_type)|tokenizer[.]chat_template)$"
    r"|[.](block_count|context_length|embedding_length|feed_forward_length)$"
    r"|expert|nextn|mtp",
    re.IGNORECASE,
)


def resolve_shards(first_shard: Path) -> list[Path]:
    match = SHARD_RE.match(first_shard.name)
    if not match:
        return [first_shard]

    count = int(match.group("count"))
    prefix = match.group("prefix")
    shards = [
        first_shard.parent / f"{prefix}-{index:05d}-of-{count:05d}.gguf"
        for index in range(1, count + 1)
    ]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing GGUF shard(s): " + ", ".join(missing))
    return shards


def json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return str(value)


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def inspect(first_shard: Path, tensor_pattern: re.Pattern[str] | None = None) -> dict[str, Any]:
    shards = resolve_shards(first_shard.resolve())
    metadata: dict[str, Any] = {}
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    expert_layouts: Counter[str] = Counter()
    matched_tensors: list[dict[str, Any]] = []
    totals = Counter()

    for shard in shards:
        reader = GGUFReader(str(shard), "r")
        for name, field in reader.fields.items():
            if name not in metadata and INTERESTING_METADATA.search(name):
                metadata[name] = json_value(field.contents())

        for tensor in reader.tensors:
            type_name = tensor.tensor_type.name
            totals["tensors"] += 1
            totals["elements"] += int(tensor.n_elements)
            totals["bytes"] += int(tensor.n_bytes)
            type_counts[type_name]["tensors"] += 1
            type_counts[type_name]["elements"] += int(tensor.n_elements)
            type_counts[type_name]["bytes"] += int(tensor.n_bytes)

            if tensor_pattern is not None and tensor_pattern.search(tensor.name):
                matched_tensors.append(
                    {
                        "name": tensor.name,
                        "type": type_name,
                        "shape": [int(dim) for dim in tensor.shape],
                        "elements": int(tensor.n_elements),
                        "bytes": int(tensor.n_bytes),
                    }
                )

            if "exps" in tensor.name or "expert" in tensor.name:
                shape = "x".join(str(int(dim)) for dim in tensor.shape)
                expert_layouts[f"{type_name} {shape}"] += 1

    def project_repack(specs: tuple[tuple[str, float], ...]) -> tuple[float, int]:
        projected = float(totals["bytes"])
        elements = 0
        for type_name, bpw in specs:
            item = type_counts.get(type_name)
            if not item:
                continue
            projected -= item["bytes"]
            projected += item["elements"] * bpw / 8.0
            elements += item["elements"]
        return projected, elements

    compact_bytes, compact_elements = project_repack(
        (("IQ2_XS", 3.3125), ("IQ3_XXS", 4.1875))
    )
    runtime_bytes, runtime_elements = project_repack(
        (("IQ2_XS", 3.3125), ("IQ3_XXS", 4.1875), ("Q5_K", 8.625))
    )

    architecture = metadata.get("general.architecture")
    warnings: list[str] = []
    if architecture != "glm-dsa":
        warnings.append(
            f"architecture is {architecture!r}, not the validated 'glm-dsa' path"
        )
    if compact_elements == 0:
        warnings.append("no IQ2_XS/IQ3_XXS tensors; compact repack gates will not help")
    if runtime_bytes > 700 * 1024**3:
        warnings.append("projected runtime-repacked weights exceed 700 GiB and leave unsafe host headroom")

    by_type = []
    for type_name, item in sorted(
        type_counts.items(), key=lambda pair: pair[1]["bytes"], reverse=True
    ):
        by_type.append(
            {
                "type": type_name,
                "tensors": item["tensors"],
                "elements": item["elements"],
                "bytes": item["bytes"],
                "percent_elements": 100.0 * item["elements"] / totals["elements"],
                "percent_bytes": 100.0 * item["bytes"] / totals["bytes"],
            }
        )

    return {
        "first_shard": str(shards[0]),
        "shards": [str(path) for path in shards],
        "shard_count": len(shards),
        "metadata": metadata,
        "totals": dict(totals),
        "tensor_types": by_type,
        "expert_layouts": [
            {"layout": layout, "tensors": count}
            for layout, count in expert_layouts.most_common()
        ],
        "matched_tensors": matched_tensors,
        "compact_repack": {
            "eligible_elements": compact_elements,
            "eligible_percent_elements": (
                100.0 * compact_elements / totals["elements"]
                if totals["elements"]
                else 0.0
            ),
            "projected_weight_bytes": round(compact_bytes),
            "projected_growth_bytes": round(compact_bytes - totals["bytes"]),
        },
        "runtime_repack": {
            "eligible_elements": runtime_elements,
            "eligible_percent_elements": (
                100.0 * runtime_elements / totals["elements"]
                if totals["elements"]
                else 0.0
            ),
            "projected_weight_bytes": round(runtime_bytes),
            "projected_growth_bytes": round(runtime_bytes - totals["bytes"]),
        },
        "warnings": warnings,
    }


def print_text(result: dict[str, Any]) -> None:
    totals = result["totals"]
    compact = result["compact_repack"]
    runtime = result["runtime_repack"]
    print(f"GGUF: {result['first_shard']}")
    print(f"Shards: {result['shard_count']}")
    print(f"Architecture: {result['metadata'].get('general.architecture', 'unknown')}")
    print(f"Parameters: {totals['elements']:,}")
    print(f"GGUF tensor bytes: {human_bytes(totals['bytes'])}")
    print(
        "Compact-repack coverage: "
        f"{compact['eligible_percent_elements']:.2f}% of parameters"
    )
    print(
        "Projected compact weight bytes: "
        f"{human_bytes(compact['projected_weight_bytes'])} "
        f"({human_bytes(compact['projected_growth_bytes'])} growth)"
    )
    print(
        "Projected enabled-repack weight bytes: "
        f"{human_bytes(runtime['projected_weight_bytes'])} "
        f"({human_bytes(runtime['projected_growth_bytes'])} growth; "
        f"{runtime['eligible_percent_elements']:.2f}% coverage)"
    )
    print("\nTensor types:")
    print("type        tensors       parameters   param %          bytes   byte %")
    for item in result["tensor_types"]:
        print(
            f"{item['type']:<10} {item['tensors']:>8} "
            f"{item['elements']:>16,} {item['percent_elements']:>8.2f} "
            f"{human_bytes(item['bytes']):>14} {item['percent_bytes']:>8.2f}"
        )
    print("\nExpert tensor layouts:")
    for item in result["expert_layouts"]:
        print(f"{item['tensors']:>5}  {item['layout']}")
    if result["matched_tensors"]:
        print("\nMatched tensors:")
        for item in result["matched_tensors"]:
            shape = "x".join(str(dim) for dim in item["shape"])
            print(
                f"{item['type']:<10} {shape:<24} "
                f"{human_bytes(item['bytes']):>12}  {item['name']}"
            )
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect GLM GGUF metadata, tensor types, and compact-repack coverage"
    )
    parser.add_argument("first_shard", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--tensor-regex",
        help="also list tensor names, types, and shapes matching this regular expression",
    )
    args = parser.parse_args()

    try:
        tensor_pattern = re.compile(args.tensor_regex) if args.tensor_regex else None
        result = inspect(args.first_shard, tensor_pattern)
    except (FileNotFoundError, OSError, ValueError, re.error) as error:
        parser.error(str(error))

    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_text(result)


if __name__ == "__main__":
    main()
