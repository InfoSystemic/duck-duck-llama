#!/usr/bin/env python3
"""Re-verify the Qwen-Image-2.1 weights against the SHA-256 that Hugging Face records for the pinned revisions."""
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
api = HfApi()
D = os.environ.get("QWEN_IMAGE_DIR", "/models/gguf/Qwen-Image-2.1")
want = [("unsloth/Qwen-Image-2.1-GGUF","2c31ccd392b367a6637841a143813320a02dff55",["qwen-image-2.1-Q8_0.gguf"]),
        ("unsloth/Qwen-Image-2.1-FP8","9e5206424833a00b340bf5396b37486d1306e61f",["vae/qwen_image_2.1_vae_bf16.safetensors"]),
        ("unsloth/Qwen3-VL-8B-Instruct-GGUF","b93a7ee713758252c555be4210c00540df954dc2",["Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf","mmproj-F16.gguf"])]
jobs = []
for repo, rev, files in want:
    for p in api.get_paths_info(repo, files, revision=rev):
        jobs.append((p.path, p.lfs.sha256 if p.lfs else None, p.size))
def sha(path):
    h = hashlib.sha256()
    with open(os.path.join(D, path), "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(lambda j: (j, sha(j[0])), jobs))
ok = True
for (path, exp, size), got in res:
    good = got == exp and os.path.getsize(os.path.join(D, path)) == size
    ok &= good
    print(("OK  " if good else "BAD ") + path, got)
print("ALL OK" if ok else "MISMATCH")
