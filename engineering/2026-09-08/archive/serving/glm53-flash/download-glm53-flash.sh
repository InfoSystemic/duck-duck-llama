#!/usr/bin/env bash
set -euo pipefail

REPO=unsloth/GLM-5.3-Flash-GGUF
QUANT="${GLM53_QUANT:-UD-IQ2_XXS}"
DEST="${GLM53_DEST:-/models/gguf/GLM-5.3-Flash}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$QUANT" in
  UD-IQ2_XXS)
    REVISION=2975ab414d30340466d8c51533c6e91f0cca64c1
    NEED_BYTES=108000000000
    MANIFEST="$SCRIPT_DIR/SHA256SUMS.UD-IQ2_XXS"
    ;;
  UD-IQ3_XXS)
    REVISION=ac47690c15c8703615ab7d9c1ef2293d45372757
    NEED_BYTES=130000000000
    MANIFEST="$SCRIPT_DIR/SHA256SUMS"
    ;;
  *)
    printf 'Unsupported GLM53_QUANT: %s\n' "$QUANT" >&2
    exit 1
    ;;
esac

if [[ "$DEST" == /models || "$DEST" == /models/* ]]; then
  model_source=$(findmnt -n -o SOURCE --target /models 2>/dev/null || true)
  if [[ -z "$model_source" || ! -b "$model_source" ]]; then
    printf 'Refusing to write to /models: its backing block device is not present.\n' >&2
    exit 1
  fi
fi

DEST_PARENT="${DEST%/*}"
mkdir -p "$DEST_PARENT"
available_bytes=$(df --output=avail -B1 "$DEST_PARENT" | tail -n 1 | tr -d ' ')
if (( available_bytes < NEED_BYTES )); then
  printf 'Need at least %s free bytes at %s; only %s are available.\n' "$NEED_BYTES" "$DEST_PARENT" "$available_bytes" >&2
  exit 1
fi

mkdir -p "$DEST"
/home/kwebb/.local/bin/hf download "$REPO" \
  --revision "$REVISION" \
  --include "${QUANT}/*" --include "mmproj-F16.gguf" \
  --local-dir "$DEST"

cd "$DEST"
sha256sum -c "$MANIFEST"
chmod a-w \
  "$DEST/$QUANT"/*.gguf \
  "$DEST/mmproj-F16.gguf"

printf 'Verified GLM-5.3-Flash at %s\n' "$DEST"
du -sh "$DEST"
