#!/usr/bin/env python3
"""Write a truncated GLM-5.3 GGUF (first N blocks) for fast kernel iteration."""
import sys, re, glob
sys.path.insert(0, '/home/user/InfoSystemic/AI-Server/engines/llama.cpp-sr950-glm/gguf-py')
import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf import GGUFValueType

n_blocks = int(sys.argv[1]); out = sys.argv[2]
shards = sorted(glob.glob('/models/GLM-5.3-GGUF/UD-Q4_K_XL/GLM-5.3-UD-Q4_K_XL-*.gguf'))
readers = [GGUFReader(s) for s in shards]
r0 = readers[0]
arch = 'glm-dsa'
w = GGUFWriter(out, arch)
skip = {'GGUF.version','GGUF.tensor_count','GGUF.kv_count','split.no','split.count','split.tensors.count','general.architecture'}
for k, f in r0.fields.items():
    if k in skip: continue
    vt = f.types[0]
    if k == f'{arch}.block_count':
        w.add_uint32(k, n_blocks); continue
    if vt == GGUFValueType.ARRAY:
        sub = f.types[1]
        vals = [f.parts[i].tolist() if sub != GGUFValueType.STRING else bytes(f.parts[i]).decode('utf-8') for i in f.data]
        if sub == GGUFValueType.STRING:
            w.add_array(k, vals)
        else:
            w.add_array(k, [v[0] if isinstance(v, list) else v for v in vals])
        continue
    val = f.parts[f.data[0]]
    if vt == GGUFValueType.STRING:
        w.add_string(k, bytes(val).decode('utf-8'))
    elif vt == GGUFValueType.UINT32: w.add_uint32(k, int(val[0]))
    elif vt == GGUFValueType.INT32: w.add_int32(k, int(val[0]))
    elif vt == GGUFValueType.FLOAT32: w.add_float32(k, float(val[0]))
    elif vt == GGUFValueType.BOOL: w.add_bool(k, bool(val[0]))
    elif vt == GGUFValueType.UINT64: w.add_uint64(k, int(val[0]))
    elif vt == GGUFValueType.INT64: w.add_int64(k, int(val[0]))
    elif vt == GGUFValueType.FLOAT64: w.add_float64(k, float(val[0]))
    elif vt == GGUFValueType.UINT8: w.add_uint8(k, int(val[0]))
    elif vt == GGUFValueType.INT8: w.add_int8(k, int(val[0]))
    elif vt == GGUFValueType.UINT16: w.add_uint16(k, int(val[0]))
    elif vt == GGUFValueType.INT16: w.add_int16(k, int(val[0]))
    else: print('skip', k, vt, file=sys.stderr)
keep = []
for r in readers:
    for t in r.tensors:
        m = re.match(r'blk\.(\d+)\.', t.name)
        if m and int(m.group(1)) >= n_blocks: continue
        keep.append(t)
total = 0
from gguf import GGMLQuantizationType as Q
for t in keep:
    data = np.asarray(t.data)
    if t.tensor_type in (Q.F32, Q.F16, Q.BF16):
        w.add_tensor(t.name, data)
    else:
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    total += int(t.n_bytes)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
print('wrote', out, 'tensors', len(keep), 'bytes', total)
