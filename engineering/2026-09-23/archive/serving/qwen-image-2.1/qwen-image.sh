#!/usr/bin/env bash
# Qwen-Image-2.1 (unsloth Q8_0 GGUF) on smeagol via stable-diffusion.cpp, CPU only.
#
#   qwen-image.sh -p "a red fox reading a newspaper"             # text-to-image, 1024x1024
#   qwen-image.sh -p "make it night" -r in.png -o edit.png        # edit (adds the vision tower)
#   qwen-image.sh -p "..." -W 1536 -H 864 --steps 30 -s 7         # any sd-cli flag overrides a default
#
# MODE=gated (default): borrows the physical llama-server worker cores of NODES (15 per
#   socket, default sockets 0-2), but yield-gate.py SIGSTOPs the job whenever a llama-server or
#   other /models/ process is busy, or anything else runs on those cores or their hyperthread
#   siblings, and resumes it 5-10 s after they go idle. So a decode or another session's
#   benchmark always wins, and two image jobs queue behind each other. If another session holds a socket (e.g. a bench pinned to 16-30),
#   choose NODES that leave it alone: the job's threads and memory both stay on NODES.
# MODE=spare: the 8 spare CPUs only (15,31,47,63,79,95,111,127). Never touches worker cores,
#   but they are shared with the CD pipelines and docker, so an image takes hours.
# The job is marked OOM-first either way, so a RAM squeeze kills it, never a model server.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SD=${SD:-$HOME/InfoSystemic/AI-Server/engines/stable-diffusion.cpp/build/bin/sd-cli}
M=${QWEN_IMAGE_DIR:-/models/gguf/Qwen-Image-2.1}
DIT=${DIT:-$M/qwen-image-2.1-Q8_0.gguf}
SPARE=15,31,47,63,79,95,111,127
MODE=${MODE:-gated}
NODES=${NODES:-0,1,2}  # measured: 1 socket 39 s/step, 3 sockets 20 s/step, a 4th adds ~0
OUT_DIR=${OUT_DIR:-$HERE/out}
# On this Cascade Lake, llamafile's Q8_0 GEMM is bound by per-32-block scale math; its F16 GEMM
# is ~1.4x faster end to end (28 -> 20 s/step at 512x512), so upcast the DiT at load. Q8_0 -> F16
# is exact, it only costs RAM (7.3 -> 13.6 GB). The VAE is always F16 (sd.cpp hard-codes it).
TYPE_RULES=${TYPE_RULES:-'^model\.diffusion_model\.=f16'}

case $MODE in
    gated) worker=()
           for n in ${NODES//,/ }; do worker+=("$((n * 16))-$((n * 16 + 14))"); done
           CPUS=${CPUS:-$(IFS=,; echo "${worker[*]}")}; THREADS=${THREADS:-$((15 * ${#worker[@]}))}
           # One node: strictly local memory. Several: interleave (falls back when a node is full).
           if [[ $NODES == *,* ]]; then mem=--interleave="$NODES"; else mem=--membind="$NODES"; fi
           launch=(env GATE_CPUS="$CPUS" taskset -c "$SPARE" python3 "$HERE/yield-gate.py"
                   numactl "$mem") ;;
    spare) CPUS=${CPUS:-$SPARE}; THREADS=${THREADS:-8}; launch=() ;;
    *) echo "MODE must be gated or spare" >&2; exit 2 ;;
esac

vision=()
for a in "$@"; do
    case $a in -r|--ref-image) vision=(--llm_vision "$M/mmproj-F16.gguf") ;; esac
done

mkdir -p "$OUT_DIR"
echo 1000 > /proc/self/oom_score_adj

# Defaults first: sd-cli keeps the last value of a repeated flag, so caller flags win.
exec "${launch[@]}" taskset -c "$CPUS" nice -n 10 "$SD" \
    --diffusion-model "$DIT" \
    --vae "$M/vae/qwen_image_2.1_vae_bf16.safetensors" \
    --llm "$M/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf" \
    "${vision[@]}" \
    --tensor-type-rules "$TYPE_RULES" \
    --cfg-scale 6.0 --sampling-method euler --steps 20 -W 1024 -H 1024 -s -1 \
    -t "$THREADS" \
    -o "$OUT_DIR/qwen-image-$(date +%Y%m%d-%H%M%S).png" \
    "$@"
