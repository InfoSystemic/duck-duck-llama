#!/usr/bin/env python3
"""patch-mask-embedding.py -- write DFlash's learned mask-token embedding into the target GGUF's token_embd row.

Why this is needed. MiMo's DFlash drafter builds its input block as [last_token, <mask> * (block_size-1)] and takes the
embeddings for those ids from the TARGET model's table -- `dflash.py` does exactly that:

    noise_embedding = target.model.embed_tokens(block_output_ids)

But the base checkpoint's row for the mask token is empty (measured: norm 0.0008 against the shipped vector's 1.8617,
cosine -0.02), which is precisely why the repo ships `dflash/mask_embedding.pt` as a separate file. Loading it into the
target's embedding table is part of the intended setup, not a hack. Without it the drafter sees a zero vector at seven of
every eight block positions and draft acceptance collapses: measured 0.014-0.072 against a no-speculation baseline that
is already faster.

Why an in-place patch rather than a re-conversion: one row of a Q8_0 [6144 x 152576] tensor is 192 blocks x 34 bytes =
6528 bytes inside a 42 GiB shard. The original bytes are saved beside the file first, so this is reversible.

Only the mask token's row changes, and the target never generates that token, so nothing else about the model moves.

    ./patch-mask-embedding.py            # report what it would do
    ./patch-mask-embedding.py --write    # back up the row, patch, verify
    ./patch-mask-embedding.py --restore  # put the original bytes back
"""
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

SHARD = Path('/models/mimo-v26-pro/gguf/MiMo-V2.6-Pro-RL-MXFP4_MOE-00001-of-00013.gguf')
MASK_PT = Path('/models/mimo-v26-pro/src/dflash/mask_embedding.pt')
BACKUP = Path('/models/mimo-v26-pro/gguf/token_embd_mask_row.orig')
QK = 32          # Q8_0 block size
BLOCK_BYTES = 34  # f16 scale + 32 int8


def quantize_q8_0(vec: np.ndarray) -> bytes:
    """Same rule as ggml's quantize_row_q8_0: per 32 values, d = max|x| / 127, q = round(x / d)."""
    out = bytearray()
    for i in range(0, len(vec), QK):
        blk = vec[i:i + QK].astype(np.float32)
        amax = float(np.max(np.abs(blk)))
        d = amax / 127.0
        q = np.zeros(QK, dtype=np.int8) if d == 0 else np.clip(np.rint(blk / d), -127, 127).astype(np.int8)
        out += np.float16(d).tobytes() + q.tobytes()
    return bytes(out)


def dequantize_q8_0(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    d = a[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    q = a[:, 2:].view(np.int8).astype(np.float32)
    return (q * d[:, None]).ravel()[:n]


def locate():
    r = GGUFReader(str(SHARD))
    t = next(t for t in r.tensors if t.name == 'token_embd.weight')
    n_embd, n_vocab = (int(x) for x in t.shape)
    assert t.tensor_type.name == 'Q8_0', f'expected Q8_0, found {t.tensor_type.name}'
    row_bytes = (n_embd // QK) * BLOCK_BYTES
    return int(t.data_offset), n_embd, n_vocab, row_bytes


def main():
    d = torch.load(MASK_PT, map_location='cpu', weights_only=True)
    mask_id, emb = int(d['mask_token_id']), d['embedding'].float().numpy().ravel()
    base, n_embd, n_vocab, row_bytes = locate()
    assert emb.shape[0] == n_embd, f'{emb.shape[0]} != n_embd {n_embd}'
    assert mask_id < n_vocab, f'mask id {mask_id} outside vocab {n_vocab}'
    off = base + mask_id * row_bytes

    with SHARD.open('rb') as fh:
        fh.seek(off)
        current = fh.read(row_bytes)
    cur_vec = dequantize_q8_0(current, n_embd)
    print(f'token_embd.weight  ne=[{n_embd}, {n_vocab}]  data at {base}  row {mask_id} at {off} ({row_bytes} B)')
    print(f'current row   norm {np.linalg.norm(cur_vec):8.4f}')
    print(f'mask_embedding norm {np.linalg.norm(emb):8.4f}')

    new = quantize_q8_0(emb)
    assert len(new) == row_bytes, f'{len(new)} != {row_bytes}'
    check = dequantize_q8_0(new, n_embd)
    rel = float(np.linalg.norm(check - emb) / (np.linalg.norm(emb) + 1e-12))
    print(f'quantisation round-trip relative error {rel:.5f}')

    if '--restore' in sys.argv:
        if not BACKUP.is_file():
            sys.exit('no backup to restore')
        with SHARD.open('r+b') as fh:
            fh.seek(off); fh.write(BACKUP.read_bytes()); fh.flush()
        print('restored the original row')
        return
    if '--write' not in sys.argv:
        print('\ndry run. pass --write to patch (the original row is saved to the backup path first).')
        return
    if not BACKUP.is_file():
        BACKUP.write_bytes(current)
        print(f'original row saved to {BACKUP}')
    with SHARD.open('r+b') as fh:
        fh.seek(off); fh.write(new); fh.flush()
    with SHARD.open('rb') as fh:
        fh.seek(off); after = fh.read(row_bytes)
    v = dequantize_q8_0(after, n_embd)
    cos = float(np.dot(v, emb) / (np.linalg.norm(v) * np.linalg.norm(emb) + 1e-12))
    print(f'patched. row now norm {np.linalg.norm(v):.4f}, cosine to the shipped vector {cos:.6f}')


main()
