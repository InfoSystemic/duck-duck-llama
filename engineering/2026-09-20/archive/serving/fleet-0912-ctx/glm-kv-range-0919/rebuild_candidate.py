#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,time
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=json.loads((HERE/'parent-rebuild.json').read_text())
assert parent['byte_identical'] and sha(HERE/'build/libllama.parent.so.0.3.0')==parent['expected_sha256']
for row in json.loads((HERE/'parent-final-objects.json').read_text()):
    assert sha(row['path'])==row['sha256'],row['path']
command=next(c[:] for c in parent['commands'] if c[-1].endswith('/llama-kv-cache.cpp'))
command[command.index('-c')+1]=str(HERE/'candidate/llama-kv-cache.cpp')
command[command.index('-o')+1]=str(HERE/'candidate/llama-kv-cache.cpp.o')
command=[arg.replace(str(HERE/'src'),str(HERE/'candidate')) if arg.startswith('-fmacro-prefix-map=') else arg for arg in command]
engine=Path(json.loads((HERE/'parent-inputs.json').read_text())['engine'])
print('Compiling isolated KV rollback candidate',flush=True)
with (HERE/'logs/candidate-compile.log').open('w') as log:
    subprocess.run(['nice','-n','10']+command,cwd=engine/'build-goal',stdout=log,stderr=subprocess.STDOUT,check=True)
link=parent['link_command'][:]
link[link.index(str(HERE/'objects/llama-kv-cache.cpp.o'))]=str(HERE/'candidate/llama-kv-cache.cpp.o')
link[link.index('-o')+1]=str(HERE/'candidate-lib/libllama.so.0.3.0')
subprocess.run(link,cwd=engine/'build-goal/src',check=True)
out={'compiled_at':time.time(),'parent_sha256':parent['expected_sha256'],'candidate_sha256':sha(HERE/'candidate-lib/libllama.so.0.3.0'),'source_sha256':sha(HERE/'candidate/llama-kv-cache.cpp'),'object_sha256':sha(HERE/'candidate/llama-kv-cache.cpp.o'),'changed_objects':['llama-kv-cache.cpp.o'],'compile':command,'link':link,'flag':'LLAMA_KV_SEQ_RM_USED_PREFIX=1 (default off)'}
(HERE/'candidate-build.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['candidate_sha256'],flush=True)
