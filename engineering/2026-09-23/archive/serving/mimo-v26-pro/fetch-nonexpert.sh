#!/bin/bash
# Fetch the non-expert part of MiMo-V2.6-Pro-RL (pinned revision) into /models staging,
# then verify every LFS file against the HF sha256 manifest.
# ep0_shard0 = all attention/embed/router/dense tensors + experts 0-2; ep0_shard1 = vision/audio;
# model_mtp = the 3 nextn layers; audio_tokenizer = the mmproj audio encoder.
set -euo pipefail
REPO=XiaomiMiMo/MiMo-V2.6-Pro-RL
REV=54b10491b1811c76aa9681a9d0ff872396a4064c
DST=/models/mimo-v26-pro/src
HERE=$(cd "$(dirname "$0")" && pwd)

hf download "$REPO" --revision "$REV" --local-dir "$DST" \
  --include "*.json" "*.jinja" "*.txt" "*.py" "*.md" "tokenizer*" "vocab.json" "merges.txt" \
            "model_pp0_ep0_shard0.safetensors" "model_pp0_ep0_shard1.safetensors" \
            "model_mtp.safetensors" "audio_tokenizer/*" "dflash/*.json" "dflash/*.py"

python3 - "$DST" "$HERE/hf-manifest-54b10491.json" <<'EOF'
import hashlib, json, os, sys
dst, man = sys.argv[1], json.load(open(sys.argv[2]))
bad = 0
for f in ["model_pp0_ep0_shard0.safetensors", "model_pp0_ep0_shard1.safetensors",
          "model_mtp.safetensors", "audio_tokenizer/model.safetensors"]:
    h = hashlib.sha256()
    with open(os.path.join(dst, f), "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    ok = h.hexdigest() == man["sha256"][f]
    bad += not ok
    print(("OK  " if ok else "BAD ") + f, flush=True)
sys.exit(1 if bad else 0)
EOF
echo "FETCH-NONEXPERT DONE"
