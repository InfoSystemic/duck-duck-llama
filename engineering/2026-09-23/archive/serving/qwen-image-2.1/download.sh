#!/usr/bin/env bash
# Fetch Qwen-Image-2.1 (unsloth GGUF) + its VAE + Qwen3-VL-8B text encoder, pinned revisions.
# Runs on smeagol's spare cores so it cannot land on llama-server worker cores.
set -euo pipefail
DEST=${DEST:-/models/gguf/Qwen-Image-2.1}
SPARE=15,31,47,63,79,95,111,127
dl() { taskset -c "$SPARE" nice -n 10 ionice -c3 hf download "$@" --local-dir "$DEST"; }
dl unsloth/Qwen-Image-2.1-GGUF        qwen-image-2.1-Q8_0.gguf                                  --revision 2c31ccd392b367a6637841a143813320a02dff55
dl unsloth/Qwen-Image-2.1-FP8         vae/qwen_image_2.1_vae_bf16.safetensors                   --revision 9e5206424833a00b340bf5396b37486d1306e61f
dl unsloth/Qwen3-VL-8B-Instruct-GGUF  Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf mmproj-F16.gguf      --revision b93a7ee713758252c555be4210c00540df954dc2
echo "DOWNLOAD DONE"
ls -la "$DEST" "$DEST/vae"
