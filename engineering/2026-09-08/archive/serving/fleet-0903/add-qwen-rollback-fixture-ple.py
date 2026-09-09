#!/usr/bin/env python3
"""Add small PLE weights to the architecture test's generated Qwen fixture."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'engines/llama.cpp-q4e-goal-0904/gguf-py'))
import gguf
stage = Path('/dev/shm/flash-goal-0904-qwen-rollback-fixtures')
reader = gguf.GGUFReader(stage/'qwen4exp-moe.gguf')
output = stage/'qwen4exp-ple-moe.gguf'
assert not output.exists()
writer = gguf.GGUFWriter(output, arch='qwen4exp')
spec = importlib.util.spec_from_file_location('extract_mtp', root/'serving/glm53-flash/extract-glm5next-mtp-gguf.py')
extract = importlib.util.module_from_spec(spec); spec.loader.exec_module(extract)
for field in reader.fields.values():
    if field.name.startswith('GGUF.') or field.name in ('general.architecture','general.name'):
        continue
    value=field.contents()
    if field.types[0] == gguf.GGUFValueType.ARRAY and len(value) == 0:
        continue
    writer.add_key_value(field.name,value,field.types[0],
                         sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
writer.add_string('general.name','Synthetic Qwen rollback fixture with PLE')
for key, value in {'ngram_size':3, 'heads_per_ngram':1, 'conv_kernel':3, 'eos_token_id':0}.items():
    writer.add_uint32('qwen4exp.ple.'+key,value)
writer.add_uint32('qwen4exp.embedding_length_per_layer_input',128)
for key, values, kind in [('layers',[0],gguf.GGUFValueType.UINT32),
                           ('layer_multipliers',[17,31,43],gguf.GGUFValueType.UINT64),
                           ('head_offsets',[0,67],gguf.GGUFValueType.UINT64),
                           ('head_vocab_sizes',[67,71],gguf.GGUFValueType.UINT64)]:
    writer.add_key_value('qwen4exp.ple.'+key, values, gguf.GGUFValueType.ARRAY, sub_type=kind)
tensors=[]
for t in reader.tensors:
    assert t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    values=np.array(t.data,copy=True)
    if 'norm' in t.name:
        values.fill(1)
    tensors.append((t.name,values))
rng=np.random.default_rng(904)
for name, shape in [('per_layer_token_embd.weight',(138,128)),
                     ('blk.0.ple_key.weight',(1024,256)), ('blk.0.ple_value.weight',(256,256)),
                     ('blk.0.ple_norm_key.weight',(1024,)), ('blk.0.ple_norm_query.weight',(1024,)),
                     ('blk.0.ple_norm_conv.weight',(1024,)), ('blk.0.ple_conv1d.weight',(1024,3))]:
    values=np.ones(shape,dtype=np.float32) if 'norm' in name else rng.normal(0,0.02,shape).astype(np.float32)
    tensors.append((name,values))
for name, values in tensors:
    writer.add_tensor(name,values)
writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
print(f'Created {output}: {output.stat().st_size:,} bytes')
