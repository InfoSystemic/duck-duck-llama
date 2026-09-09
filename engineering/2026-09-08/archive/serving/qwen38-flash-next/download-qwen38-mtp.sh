#!/usr/bin/env bash
set -euo pipefail

REPO=drluoto/Qwen3.8-Flash-Next-MTP-GGUF
REVISION=67de7592b670ef454a903574d5e2aa6c8e1d6b46
FILENAME=mtp-Qwen3.8-Flash-Next-Q8_0.gguf
DEST="${QWEN38_MTP_DEST:-/models/gguf/Qwen3.8-Flash-Next/MTP}"
NEED_BYTES=6000000000
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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
  --include "$FILENAME" \
  --local-dir "$DEST"

cd "$DEST"
sha256sum -c "$SCRIPT_DIR/SHA256SUMS"
chmod a-w "$DEST/$FILENAME"

printf 'Verified Qwen3.8-Flash-Next MTP sidecar at %s/%s\n' "$DEST" "$FILENAME"
du -h "$DEST/$FILENAME"
