#!/usr/bin/env python3
"""vision-ref.py -- ground truth for MiMo-V2.6-Pro's vision tower, to compare llama.cpp's mmproj against.

  mkimg <dir>                      write test images (PNG for llama.cpp + .npy of the same pixels for the reference)
  ref   <img.npy> <out.npz>        run Xiaomi's own MiMoVisionTransformer (imported from the checkpoint's
                                   modeling_mimo_v2.py) on the image; save final embeddings + every block output
  cmp   <llama.bin> <ref.npz> [dump_dir]
                                   compare llama.cpp's projector output (vision-embd) against the reference, and,
                                   given vision-embd's dump dir, every layer_out-N as well

Run with the dsv41 venv (the only one with torch + safetensors). Sizes must be multiples of 32 so neither side resizes.
"""
import json, os, struct, sys, types, zlib
import numpy as np

SRC = "/models/mimo-v26-pro/src"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def write_png(path, arr):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def mkimg(d):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    for h, w in [(128, 128), (160, 160), (224, 224), (320, 224), (576, 576)]:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.full((h, w, 3), 235, dtype=np.float32)
        circ = (xx - 0.25 * w) ** 2 + (yy - 0.3 * h) ** 2 < (0.2 * min(h, w)) ** 2
        img[circ] = (220, 30, 30)
        sq = (xx > 0.55 * w) & (xx < 0.95 * w) & (yy > 0.1 * h) & (yy < 0.5 * h)
        img[sq] = (30, 60, 220)
        tri = (yy > 0.55 * h) & (yy < 0.95 * h) & (np.abs(xx - 0.5 * w) < (yy - 0.55 * h) * 0.75 * w / h)
        img[tri] = (30, 180, 60)
        img += rng.normal(0, 8, img.shape)  # texture so no two patches are identical
        arr = np.clip(img, 0, 255).astype(np.uint8)
        name = f"shapes{w}x{h}"
        write_png(os.path.join(d, name + ".png"), arr)
        np.save(os.path.join(d, name + ".npy"), arr)
        print(name)


def load_vit():
    import torch
    from safetensors import safe_open
    sys.path.insert(0, SRC)
    # modeling_mimo_v2 imports the configuration module relatively; load both as a package-less module pair
    import importlib.util
    pkg = types.ModuleType("mimo_ref"); pkg.__path__ = [SRC]; sys.modules["mimo_ref"] = pkg
    for mod in ("configuration_mimo_v2", "modeling_mimo_v2"):
        spec = importlib.util.spec_from_file_location(f"mimo_ref.{mod}", os.path.join(SRC, mod + ".py"))
        m = importlib.util.module_from_spec(spec); sys.modules[f"mimo_ref.{mod}"] = m; spec.loader.exec_module(m)
    M = sys.modules["mimo_ref.modeling_mimo_v2"]
    vc = json.load(open(os.path.join(SRC, "config.json")))["vision_config"]
    cfg = types.SimpleNamespace(**vc)
    vit = M.MiMoVisionTransformer(cfg).float().eval()
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    want = {k: v for k, v in idx.items() if k.startswith("visual.")}
    sd = {}
    for shard in sorted(set(want.values())):
        with safe_open(os.path.join(SRC, shard), framework="pt") as f:
            for k, v in want.items():
                if v == shard:
                    sd[k[len("visual."):]] = f.get_tensor(k).float()
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    # the checkpoint ships no merger biases / ln_q bias: HF init leaves them at zero
    with torch.no_grad():
        for name in missing:
            p = dict(vit.named_parameters())[name]
            assert name.endswith("bias"), f"missing non-bias param {name}"
            p.zero_()
    print("missing (zeroed):", missing, "unexpected:", unexpected, file=sys.stderr)
    return vit


def patch_virtual_sinks():
    import torch
    import torch.nn.functional as F
    M = sys.modules["mimo_ref.modeling_mimo_v2"]

    def fwd(self, hidden_states, cu_seqlens, position_embeddings, full_attn=False):
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q_dim = self.num_heads * self.head_dim; kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[:, :q_dim].view(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_dim:q_dim + kv_dim].view(seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim:].view(seq_len, self.num_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        q, k = M._apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1); k = k.repeat_interleave(self.num_kv_groups, 1).transpose(0, 1)
        v = v.repeat_interleave(self.num_kv_groups, 1).transpose(0, 1)
        logits = (q @ k.transpose(1, 2)) * self.scaling                        # [H, L, L]
        if not full_attn:
            m = self._build_window_mask(seq_len, q.device, q.dtype)
            if m is not None:
                logits = logits + m
        if self.sinks is not None:
            sink = self.sinks.view(-1, 1, 1).expand(-1, seq_len, 1)
            logits = torch.cat([logits, sink], -1)
            p = logits.softmax(-1)[..., :-1]
        else:
            p = logits.softmax(-1)
        o = (p @ v).transpose(0, 1).reshape(seq_len, -1)
        return self.proj(o)
    M.MiMoVisionAttention.forward = fwd


def ref(npy, out):
    import torch
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
    arr = np.load(npy)
    h, w, _ = arr.shape
    P, T, m = 16, 2, 2
    x = (arr.astype(np.float32) / 255.0 - MEAN) / STD             # [h, w, 3]
    x = x.transpose(2, 0, 1)                                       # [C, h, w]
    x = np.stack([x, x], 0)                                        # [T, C, h, w] (image repeated temporally)
    gh, gw = h // P, w // P
    # Qwen2VLImageProcessor: (t, T, C, gh/m, m, P, gw/m, m, P) -> (t, gh/m, gw/m, m, m, C, T, P, P)
    x = x.reshape(1, T, 3, gh // m, m, P, gw // m, m, P).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    pix = torch.from_numpy(np.ascontiguousarray(x.reshape(gh * gw, 3 * T * P * P)))
    grid = torch.tensor([[1, gh, gw]])
    vit = load_vit()
    if os.environ.get("REF_SINK") == "virtual":
        # A/B: llama.cpp's gpt-oss-style sink (an extra logit column with V = 0) instead of the reference's
        # additive bias on key 0's logit
        patch_virtual_sinks()
    outs = {}
    hooks = [blk.register_forward_hook(lambda mod, i, o, n=n: outs.__setitem__(f"layer_out-{n}", o.detach().numpy().copy()))
             for n, blk in enumerate(vit.blocks)]
    hooks.append(vit.patch_embed.register_forward_hook(lambda mod, i, o: outs.__setitem__("patch_embed", o.detach().numpy().copy())))
    with torch.no_grad():
        emb = vit(pix, grid)
    for hk in hooks:
        hk.remove()
    np.savez(out, embd=emb.numpy(), **outs)
    print(f"ref {npy}: grid {gh}x{gw} -> embd {tuple(emb.shape)} rms {emb.pow(2).mean().sqrt():.5f}")


def read_bin(path):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(32), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=np.float32)
    return hdr, data


def stats(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    cos = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)
    return rel, cos


def cmp(llama_bin, ref_npz, dump_dir=None):
    r = np.load(ref_npz)
    hdr, e = read_bin(llama_bin)
    n_embd, n_tok = int(hdr[0]), int(hdr[1])
    e = e.reshape(n_tok, n_embd)
    re = r["embd"]
    print(f"final embd: llama {e.shape} ref {re.shape}")
    if e.shape == re.shape:
        rel, cos = stats(e, re)
        per_tok = [stats(e[i], re[i])[1] for i in range(n_tok)]
        print(f"  rel err {rel:.4e}  cos {cos:.6f}  per-token cos min {min(per_tok):.5f} "
              f"(token {int(np.argmin(per_tok))}) median {np.median(per_tok):.5f}")
    if dump_dir:
        for n in ["patch_embed"] + [f"layer_out-{i}" for i in range(64)]:
            p = os.path.join(dump_dir, n + ".bin")
            if not os.path.exists(p) or n not in r:
                continue
            _, d = read_bin(p)
            ref_t = r[n].reshape(-1)
            if d.size != ref_t.size:
                print(f"  {n}: size mismatch {d.size} vs {ref_t.size}"); continue
            rel, cos = stats(d, ref_t)
            print(f"  {n:14s} rel {rel:.4e}  cos {cos:.6f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "mkimg":
        mkimg(sys.argv[2])
    elif cmd == "ref":
        ref(sys.argv[2], sys.argv[3])
    elif cmd == "cmp":
        cmp(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
