#!/usr/bin/env python3
"""Write a truncated GGUF (first N blocks) of any model, for fast iteration.

Generalised from /dev/shm/make_trunc_glm.py, which was hard-coded to GLM-5.3
Full. Two things have to be handled that the original did not need:

  * per-layer arrays. GLM-5.3-Flash stores `attention.head_count_kv` as one
    entry per layer (0 = linear-attention layer, non-zero = full attention).
    Copying all 46 entries into an 8-layer model makes the loader disagree with
    itself about which layers are recurrent. Any array whose length equals the
    original block count is truncated too.
  * the MTP layer. `nextn_predict_layers` is set to 0, because blk.<last>.nextn.*
    is dropped along with every other block past the cut.

usage: make_trunc.py <n_blocks> <out.gguf> <shard-glob> [gguf-py path]

Set KEEP_NEXTN=1 to also carry the MTP/NextN block over, renumbered to sit right
after the kept trunk layers. `block_count` INCLUDES the nextn block (glm5next has
block_count 46 = 45 trunk + 1 nextn), so keeping it means n_blocks+1 blocks and
nextn_predict_layers stays 1. That is what makes a truncated model usable as a
speculative-decoding harness.
"""
import sys, re, glob

import os
keep_nextn = os.environ.get('KEEP_NEXTN', '0') != '0'
n_blocks = int(sys.argv[1])
out      = sys.argv[2]
pattern  = sys.argv[3]
sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4
                else '/home/kwebb/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/gguf-py')

import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf import GGUFValueType, GGMLQuantizationType as Q

shards  = sorted(glob.glob(pattern))
if not shards:
    sys.exit("no shards matched: %s" % pattern)
readers = [GGUFReader(s) for s in shards]
r0      = readers[0]

arch_field = r0.fields['general.architecture']
arch = bytes(arch_field.parts[arch_field.data[0]]).decode('utf-8')
n_blocks_orig = int(r0.fields['%s.block_count' % arch].parts[-1][0])
print("arch=%s blocks=%d -> %d" % (arch, n_blocks_orig, n_blocks))

w = GGUFWriter(out, arch)
skip = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count',
        'split.no', 'split.count', 'split.tensors.count', 'general.architecture'}

scalar_add = {
    GGUFValueType.UINT32: w.add_uint32, GGUFValueType.INT32:  w.add_int32,
    GGUFValueType.FLOAT32: w.add_float32, GGUFValueType.BOOL: w.add_bool,
    GGUFValueType.UINT64: w.add_uint64, GGUFValueType.INT64:  w.add_int64,
    GGUFValueType.FLOAT64: w.add_float64, GGUFValueType.UINT8: w.add_uint8,
    GGUFValueType.INT8:   w.add_int8,   GGUFValueType.UINT16: w.add_uint16,
    GGUFValueType.INT16:  w.add_int16,
}

for k, f in r0.fields.items():
    if k in skip:
        continue
    vt = f.types[0]
    if k == '%s.block_count' % arch:
        w.add_uint32(k, n_blocks + (1 if keep_nextn else 0)); continue
    if k.endswith('nextn_predict_layers'):
        w.add_uint32(k, 1 if keep_nextn else 0); continue
    if vt == GGUFValueType.ARRAY:
        sub  = f.types[1]
        vals = [bytes(f.parts[i]).decode('utf-8') if sub == GGUFValueType.STRING
                else f.parts[i].tolist() for i in f.data]
        if len(vals) == n_blocks_orig:          # one entry per layer
            # keep the nextn block's own entry (the last one) when carrying it over
            vals = vals[:n_blocks] + ([vals[-1]] if keep_nextn else [])
        # Preserve the element type explicitly. add_array() infers it from the
        # first value and picks INT32 for any Python int, which overflows on
        # genuinely 64-bit arrays (qwen4exp.ple.layer_multipliers holds values
        # around 2.4e13).
        if sub == GGUFValueType.STRING:
            w.add_key_value(k, vals, GGUFValueType.ARRAY, sub_type=sub)
        else:
            w.add_key_value(k, [v[0] if isinstance(v, list) else v for v in vals],
                            GGUFValueType.ARRAY, sub_type=sub)
        continue
    val = f.parts[f.data[0]]
    if vt == GGUFValueType.STRING:
        w.add_string(k, bytes(val).decode('utf-8'))
    elif vt in scalar_add:
        scalar_add[vt](k, int(val[0]) if vt != GGUFValueType.FLOAT32
                          and vt != GGUFValueType.FLOAT64 else float(val[0]))
    else:
        print('skipped kv', k, vt, file=sys.stderr)

kept = 0
for r in readers:
    for t in r.tensors:
        m = re.match(r'blk\.(\d+)\.', t.name)
        name = t.name
        if m:
            il = int(m.group(1))
            if il >= n_blocks:
                if not (keep_nextn and il == n_blocks_orig - 1):
                    continue
                name = re.sub(r'^blk\.\d+\.', 'blk.%d.' % n_blocks, t.name)
        data = np.asarray(t.data)
        # BF16 must go through raw_dtype: older gguf-py's add_tensor() only
        # accepts F16/F32/F64/I* without it, and qwen4exp keeps its indexer
        # projections in BF16.
        if t.tensor_type in (Q.F32, Q.F16):
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_dtype=t.tensor_type)
        kept += 1

w.write_header_to_file(); w.write_kv_data_to_file()
w.write_tensors_to_file(progress=False); w.close()
print("wrote %s (%d tensors)" % (out, kept))
