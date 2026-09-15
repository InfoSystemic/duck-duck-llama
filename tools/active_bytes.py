#!/usr/bin/env python3
"""Active-bytes decomposition for a GGUF model: what a forward pass actually STREAMS from DRAM.

For a bandwidth-bound CPU deployment this number is the denominator of every ceiling you will quote, so it has to be
right. Two ways it goes wrong, both of which produced wrong published ceilings for this fleet:

1. GATHERED lookup tables counted as streamed weights. Qwen3.8-Flash-Next carries `per_layer_token_embd` at 55 GB and
   DeepSeek-V4.1 carries `engram_embd` at 203 GB. Both are indexed per token / per n-gram, so a forward pass touches a
   handful of rows. Counting them as streamed gives a byte floor of ~7 tok/s for a model that measurably runs at 26.

2. A wrong block size for one tensor type, which is silent. This file now RECONSTRUCTS the model size from the tensor
   table and prints it against the on-disk size. If they disagree the type table is wrong and every number below it is
   wrong. Unknown type ids raise rather than defaulting to a guess -- ids 31-33 in particular are Q4_0_4_4/4_8/8_8 in
   some forks and TQ1_0/TQ2_0 in mainline, so guessing is not safe.

It also models the SPECULATIVE case, which a per-token figure cannot: a verify pass over b drafted tokens reads the
shared weights ONCE and the expert tensors for the UNION of the experts those b tokens route to. Bytes per ACCEPTED
token therefore fall well below bytes per pass, which is why speculation helps a bandwidth-bound MoE at all.

Usage:
  active_bytes.py '<glob of shards>' [experts_used_override] [accepted_tokens_per_cycle]
"""
import struct, sys, glob, os
from collections import defaultdict

# ggml_type enum ids. Verified against ggml/include/ggml.h. Deliberately incomplete: unknown ids raise.
TYPES = {0:'f32',1:'f16',2:'q4_0',3:'q4_1',6:'q5_0',7:'q5_1',8:'q8_0',9:'q8_1',10:'q2_K',11:'q3_K',12:'q4_K',
         13:'q5_K',14:'q6_K',15:'q8_K',16:'iq2_xxs',17:'iq2_xs',18:'iq3_xxs',19:'iq1_s',20:'iq4_nl',21:'iq3_s',
         22:'iq2_s',23:'iq4_xs',24:'i8',25:'i16',26:'i32',27:'i64',28:'f64',29:'iq1_m',30:'bf16',39:'mxfp4'}
BLK = {'f32':(1,4),'f16':(1,2),'bf16':(1,2),'f64':(1,8),'i8':(1,1),'i16':(1,2),'i32':(1,4),'i64':(1,8),
       'q4_0':(32,18),'q4_1':(32,20),'q5_0':(32,22),'q5_1':(32,24),'q8_0':(32,34),'q8_1':(32,36),
       'q2_K':(256,84),'q3_K':(256,110),'q4_K':(256,144),'q5_K':(256,176),'q6_K':(256,210),'q8_K':(256,292),
       'iq4_nl':(32,18),'iq4_xs':(256,136),'iq3_s':(256,110),'iq2_s':(256,82),'iq1_s':(256,50),'iq1_m':(256,56),
       'iq2_xxs':(256,66),'iq2_xs':(256,74),'iq3_xxs':(256,98),'mxfp4':(32,17)}

def _u32(f): return struct.unpack('<I', f.read(4))[0]
def _u64(f): return struct.unpack('<Q', f.read(8))[0]
def _str(f): return f.read(_u64(f)).decode('utf-8', 'replace')

def _val(f, t):
    if t == 4:  return struct.unpack('<i', f.read(4))[0]
    if t == 5:  return _u32(f)
    if t == 10: return struct.unpack('<q', f.read(8))[0]
    if t == 11: return _u64(f)
    if t in (0, 1, 7): return f.read(1)[0]
    if t in (2, 3): return struct.unpack('<H', f.read(2))[0]
    if t == 6:  return struct.unpack('<f', f.read(4))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    if t == 8:  return _str(f)
    if t == 9:
        et = _u32(f); n = _u64(f); return [_val(f, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF metadata value type {t}")

def parse(path):
    kv, ts = {}, []
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF': raise SystemExit(f"{path}: not a GGUF file")
        _u32(f); nt = _u64(f); nkv = _u64(f)
        for _ in range(nkv):
            k = _str(f); kv[k] = _val(f, _u32(f))
        for _ in range(nt):
            name = _str(f); nd = _u32(f); dims = [_u64(f) for _ in range(nd)]; tid = _u32(f); _u64(f)
            n = 1
            for d in dims: n *= d
            tn = TYPES.get(tid)
            if tn is None or tn not in BLK:
                raise SystemExit(f"unknown ggml type id {tid} on tensor {name}: add it to TYPES/BLK rather than "
                                 f"guessing, a wrong block size silently corrupts every number this tool prints")
            bs, bb = BLK[tn]
            ts.append((name, dims, tn, n // bs * bb))
    return kv, ts

# Indexed by id, not multiplied: a pass touches a few rows, so these are NOT part of the streamed working set.
GATHERED = ('token_embd', 'per_layer_token_embd', 'engram_embd')
def classify(name):
    if '_exps' in name: return 'expert'
    if any(g in name for g in GATHERED): return 'gather'
    if name.startswith('output.'): return 'head'
    return 'shared'

def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    paths = sorted(glob.glob(sys.argv[1]))
    if not paths: raise SystemExit(f"no files match {sys.argv[1]}")
    # '' or '-' means "take it from the model metadata"; only override when you actually know better
    opt = lambda i, f: (f(sys.argv[i]) if len(sys.argv) > i and sys.argv[i] not in ('', '-') else None)
    used_override = opt(2, int)
    acc = opt(3, float)

    kv, ts = {}, []
    for p in paths:
        k, t = parse(p); kv.update(k); ts += t

    agg, bytype = defaultdict(int), defaultdict(int)
    for n, d, t, b in ts:
        c = classify(n); agg[c] += b; bytype[(c, t)] += b

    recon = sum(agg.values()); disk = sum(os.path.getsize(p) for p in paths)
    ok = abs(recon - disk) / disk < 0.01
    arch = kv.get('general.architecture', '?')
    find = lambda suf: next((v for k, v in kv.items() if k.endswith(suf)), None)
    n_exp = find('.expert_count'); n_used = used_override or find('.expert_used_count'); n_lay = find('.block_count')
    exp = [t for t in ts if classify(t[0]) == 'expert']
    moe_layers = len({t[0].split('.')[1] for t in exp}) if exp else 0
    per_expert = agg['expert'] / (moe_layers * n_exp) if exp and n_exp else 0

    print(f"{arch}: {len(paths)} shard(s), {len(ts)} tensors")
    print(f"  reconstructed {recon/1e9:8.1f} GB vs on-disk {disk/1e9:8.1f} GB  ->  "
          f"{'OK' if ok else 'MISMATCH: the type table is wrong, do not trust anything below'}")
    print(f"  layers {n_lay}, experts {n_exp}, routed/token {n_used}, MoE layers {moe_layers}\n")
    for k in ('shared', 'head', 'gather', 'expert'):
        note = {'gather': '   <- indexed per token, NOT streamed',
                'expert': f'   <- one expert = {per_expert/1e6:.2f} MB'}.get(k, '')
        print(f"  {k:8s} {agg[k]/1e9:9.2f} GB{note}")
    print("  by quant: " + "  ".join(f"{c}/{t} {v/1e9:.1f}"
          for (c, t), v in sorted(bytype.items(), key=lambda x: -x[1])[:6]))

    if not (n_exp and n_used and per_expert):
        print("\n  dense model or no MoE metadata; active bytes/token = shared + head "
              f"= {(agg['shared']+agg['head'])/1e9:.2f} GB")
        return
    fixed = agg['shared'] + agg['head']
    act1 = fixed + per_expert * n_used * moe_layers
    print(f"\n  ACTIVE bytes, batch 1           = {act1/1e9:6.2f} GB  "
          f"(shared+head {100*fixed/act1:.0f}%, experts {100-100*fixed/act1:.0f}%)")
    for bw in (200e9, 381e9):
        print(f"  byte floor at {bw/1e9:.0f} GB/s          = {bw/act1:6.1f} tok/s")
    print(f"\n  speculative verify (expert union assumes INDEPENDENT routing -> an upper bound on bytes;")
    print(f"  adjacent tokens route similarly, so the true union is smaller and the real floor higher)")
    print(f"  {'batch':>6} {'union exp/layer':>16} {'GB/pass':>9} {'GB/acc tok':>11} {'floor @381':>11}")
    for b in (1, 2, 3, 5, 7):
        u = n_exp * (1 - (1 - n_used / n_exp) ** b)
        tot = fixed + per_expert * u * moe_layers
        per = tot / (acc if (acc and b == 5) else b)
        print(f"  {b:>6} {u:>16.1f} {tot/1e9:>9.2f} {per/1e9:>11.2f} {381e9/per:>11.0f}")
    if acc: print(f"\n  (the batch-5 row divides by the MEASURED {acc} accepted tokens per cycle, not by 5)")

if __name__ == '__main__':
    main()
