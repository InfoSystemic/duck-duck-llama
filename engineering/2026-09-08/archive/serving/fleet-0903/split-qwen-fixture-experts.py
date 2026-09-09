#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'engines/llama.cpp-q4e-goal-0904/gguf-py'))
import gguf
stage=Path('/dev/shm/flash-goal-0904-qwen-rollback-fixtures')
for name in ['qwen4exp-moe','qwen4exp-ple-moe']:
    r=gguf.GGUFReader(stage/(name+'.gguf'))
    output=stage/(name+'-split-experts.gguf')
    assert not output.exists()
    w=gguf.GGUFWriter(output,arch='qwen4exp')
    for f in r.fields.values():
        if f.name.startswith('GGUF.') or f.name=='general.architecture':continue
        value=f.contents()
        if f.types[0]==gguf.GGUFValueType.ARRAY and len(value)==0:continue
        w.add_key_value(f.name,value,f.types[0],sub_type=f.types[-1] if f.types[0]==gguf.GGUFValueType.ARRAY else None)
    for t in r.tensors:
        data=np.array(t.data,copy=True)
        if 'norm' in t.name or t.name.endswith(('.scale','.input_scale')):
            data.fill(1)
        if 'ffn_gate_up_exps.weight' in t.name:
            half=data.shape[1]//2
            w.add_tensor(t.name.replace('gate_up','gate'),np.ascontiguousarray(data[:,:half,:]))
            w.add_tensor(t.name.replace('gate_up','up'),np.ascontiguousarray(data[:,half:,:]))
        else:
            w.add_tensor(t.name,data)
    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
    print(output)
