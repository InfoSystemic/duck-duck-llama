#!/usr/bin/env python3
"""Compute exact active bytes-per-token for a sharded GGUF on Hugging Face,
without downloading the tensor payloads.

GGUF puts its full tensor table (name, shape, dtype) in a header at the front of
every shard, so an HTTP range request for the first few MB of each shard is
enough to price the whole model. For a 467 GB download that decision deserves a
measurement rather than an estimate.

Decode at batch 1 is bandwidth-bound, so
    tok/s = effective_bandwidth / active_bytes_per_token
and only 8 of 256 routed experts are touched per token.

Usage:
  ./remote-gguf-bytes-per-token.py --repo unsloth/GLM-5.3-GGUF --dir UD-Q4_K_XL
  ./remote-gguf-bytes-per-token.py --local /dev/shm/.../GLM-5.3-UD-Q3_K_XL-00001-of-00009.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from collections import defaultdict

# bytes per block, elements per block -- from ggml.c type traits
TYPE_INFO = {
    0:  ("F32",      4,   1),
    1:  ("F16",      2,   1),
    2:  ("Q4_0",     18,  32),
    3:  ("Q4_1",     20,  32),
    6:  ("Q5_0",     22,  32),
    7:  ("Q5_1",     24,  32),
    8:  ("Q8_0",     34,  32),
    9:  ("Q8_1",     36,  32),
    10: ("Q2_K",     84,  256),
    11: ("Q3_K",     110, 256),
    12: ("Q4_K",     144, 256),
    13: ("Q5_K",     176, 256),
    14: ("Q6_K",     210, 256),
    15: ("Q8_K",     292, 256),
    16: ("IQ2_XXS",  66,  256),
    17: ("IQ2_XS",   74,  256),
    18: ("IQ3_XXS",  98,  256),
    19: ("IQ1_S",    50,  256),
    20: ("IQ4_NL",   18,  32),
    21: ("IQ3_S",    110, 256),
    22: ("IQ2_S",    82,  256),
    23: ("IQ4_XS",   136, 256),
    24: ("I8",       1,   1),
    25: ("I16",      2,   1),
    26: ("I32",      4,   1),
    27: ("I64",      8,   1),
    28: ("F64",      8,   1),
    29: ("IQ1_M",    56,  256),
    30: ("BF16",     2,   1),
    39: ("MXFP4",    17,  32),
}


class Buf:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0

    def need(self, n: int):
        if self.o + n > len(self.d):
            raise EOFError(f"header truncated at {self.o}+{n} > {len(self.d)}")

    def u32(self) -> int:
        self.need(4); v = struct.unpack_from("<I", self.d, self.o)[0]; self.o += 4; return v

    def u64(self) -> int:
        self.need(8); v = struct.unpack_from("<Q", self.d, self.o)[0]; self.o += 8; return v

    def i64(self) -> int:
        self.need(8); v = struct.unpack_from("<q", self.d, self.o)[0]; self.o += 8; return v

    def string(self) -> str:
        n = self.u64()
        self.need(n)
        s = self.d[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def skip_value(self, t: int):
        # GGUF metadata value types
        simple = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in simple:
            self.need(simple[t]); self.o += simple[t]
        elif t == 8:      # string
            self.string()
        elif t == 9:      # array
            et = self.u32()
            n = self.u64()
            for _ in range(n):
                self.skip_value(et)
        else:
            raise ValueError(f"unknown metadata value type {t}")


def parse_header(data: bytes):
    b = Buf(data)
    magic = b.d[0:4]
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file (magic={magic!r})")
    b.o = 4
    version = b.u32()
    n_tensors = b.u64()
    n_kv = b.u64()

    meta = {}
    for _ in range(n_kv):
        key = b.string()
        vt = b.u32()
        if vt == 4 and key.count(".") >= 1:      # uint32 - keep, cheap
            meta[key] = b.u32()
        elif vt == 5:
            meta[key] = struct.unpack_from("<i", b.d, b.o)[0]; b.o += 4
        elif vt == 10:
            meta[key] = b.u64()
        elif vt == 11:
            meta[key] = b.i64()
        elif vt == 8:
            v = b.string()
            if len(v) < 200:
                meta[key] = v
        else:
            b.skip_value(vt)

    tensors = []
    for _ in range(n_tensors):
        name = b.string()
        nd = b.u32()
        dims = [b.u64() for _ in range(nd)]
        ttype = b.u32()
        b.u64()   # offset
        tensors.append((name, dims, ttype))
    return version, meta, tensors


def tensor_bytes(dims, ttype) -> int:
    nelem = 1
    for d in dims:
        nelem *= d
    if ttype not in TYPE_INFO:
        raise ValueError(f"unknown tensor type id {ttype}")
    _, blk_bytes, blk_elems = TYPE_INFO[ttype]
    return nelem * blk_bytes // blk_elems


def fetch_head(url: str, nbytes: int) -> bytes:
    out = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "120",
         "-H", f"Range: bytes=0-{nbytes - 1}", url],
        capture_output=True, check=True)
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="unsloth/GLM-5.3-GGUF")
    ap.add_argument("--dir", default="UD-Q4_K_XL")
    ap.add_argument("--local", help="path to shard 1 of a local sharded GGUF instead")
    ap.add_argument("--experts-used", type=int, default=8)
    ap.add_argument("--head-bytes", type=int, default=24 * 1024 * 1024)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    all_tensors = []
    meta = {}

    if args.local:
        m = re.match(r"^(.+)-00001-of-(\d{5})\.gguf$", args.local)
        if not m:
            print("--local must point at shard -00001-of-000NN.gguf", file=sys.stderr)
            return 2
        stem, count = m.group(1), int(m.group(2))
        for i in range(1, count + 1):
            path = f"{stem}-{i:05d}-of-{count:05d}.gguf"
            with open(path, "rb") as fh:
                data = fh.read(args.head_bytes)
            _, mm, ts = parse_header(data)
            if i == 1:
                meta = mm
            all_tensors.extend(ts)
            print(f"  shard {i:2d}/{count}: {len(ts):5d} tensors", file=sys.stderr)
    else:
        api = f"https://huggingface.co/api/models/{args.repo}?blobs=true"
        info = json.loads(subprocess.run(
            ["curl", "-sS", "-L", "--max-time", "60", api],
            capture_output=True, check=True).stdout)
        names = sorted(f["rfilename"] for f in info["siblings"]
                       if f["rfilename"].startswith(args.dir + "/")
                       and f["rfilename"].endswith(".gguf"))
        if not names:
            print(f"no .gguf under {args.dir}/ in {args.repo}", file=sys.stderr)
            return 2
        for n in names:
            url = f"https://huggingface.co/{args.repo}/resolve/main/{n}"
            data = fetch_head(url, args.head_bytes)
            _, mm, ts = parse_header(data)
            if not meta:
                meta = mm
            all_tensors.extend(ts)
            print(f"  {n}: {len(ts):5d} tensors", file=sys.stderr)

    # ---- classify -------------------------------------------------------
    n_expert = None
    n_used = args.experts_used
    for k, v in meta.items():
        if k.endswith("expert_count"):
            n_expert = int(v)
        if k.endswith("expert_used_count"):
            n_used = int(v)
    if not n_expert:
        n_expert = 256

    block_of = re.compile(r"^blk\.(\d+)\.")
    mtp_block = None
    blocks = set()
    for name, _, _ in all_tensors:
        m = block_of.match(name)
        if m:
            blocks.add(int(m.group(1)))
        if "nextn" in name:
            m2 = block_of.match(name)
            if m2:
                mtp_block = int(m2.group(1))
    n_layers = len(blocks) - (1 if mtp_block is not None else 0)

    per_type = defaultdict(int)
    total = 0
    active = 0
    cat = defaultdict(int)
    mtp_bytes = 0

    for name, dims, ttype in all_tensors:
        nb = tensor_bytes(dims, ttype)
        total += nb
        per_type[TYPE_INFO[ttype][0]] += nb

        m = block_of.match(name)
        blk = int(m.group(1)) if m else None

        if blk is not None and blk == mtp_block:
            mtp_bytes += nb
            continue

        is_routed = "_exps." in name and "shexp" not in name
        if is_routed:
            a = nb * n_used // n_expert
            cat["routed experts (%d/%d)" % (n_used, n_expert)] += a
        elif name.startswith("token_embd"):
            a = 0                      # one row per token, negligible
            cat["token_embd (skipped)"] += 0
        elif blk is None:
            a = nb                     # output_norm, output/lm_head
            cat["global (lm_head, norms)"] += a
        elif "attn" in name:
            a = nb
            cat["attention (dense)"] += a
        elif "shexp" in name:
            a = nb
            cat["shared expert"] += a
        else:
            a = nb
            cat["gate/norm/other dense"] += a
        active += a

    print()
    print(f"architecture      {meta.get('general.architecture')}")
    print(f"layers            {n_layers}   (MTP/NextN block: {mtp_block})")
    print(f"experts           {n_expert}, used {n_used}")
    print(f"tensors           {len(all_tensors)}")
    print()
    print(f"file payload      {total/1e9:10.1f} GB")
    print(f"MTP block         {mtp_bytes/1e9:10.1f} GB  (draft model source)")
    print()
    print("tensor type mix:")
    for t, v in sorted(per_type.items(), key=lambda x: -x[1]):
        print(f"  {t:10s} {v/1e9:9.1f} GB  ({100*v/total:5.1f}%)")
    print()
    print("ACTIVE BYTES PER TOKEN:")
    for c, v in sorted(cat.items(), key=lambda x: -x[1]):
        print(f"  {c:34s} {v/1e9:8.2f} GB  ({100*v/active:5.1f}%)")
    print(f"  {'TOTAL':34s} {active/1e9:8.2f} GB")
    print()
    print("tok/s ceiling at measured socket-local bandwidth:")
    for bw in (379, 300, 200, 150, 120):
        pct = 100 * bw / 379
        print(f"  {bw:3d} GB/s ({pct:5.1f}% of 379) -> {bw/(active/1e9):6.2f} tok/s raw"
              f"   |  x1.87 MTP = {1.87*bw/(active/1e9):6.2f}")
    print()
    need10 = 10 * active / 1e9
    print(f"10 tok/s RAW needs {need10:.0f} GB/s  ({100*need10/379:.0f}% of 379 GB/s ceiling)")
    print(f"10 tok/s with 1.87x MTP needs {need10/1.87:.0f} GB/s "
          f"({100*need10/1.87/379:.0f}% of ceiling)")
    print(f"10 tok/s with 2.50x MTP needs {need10/2.50:.0f} GB/s "
          f"({100*need10/2.50/379:.0f}% of ceiling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
