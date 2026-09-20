#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,subprocess,time
HERE=Path(__file__).resolve().parent
meta=json.loads((HERE/'parent-inputs.json').read_text())
ENGINE=Path(meta['engine'])
commands=json.loads((HERE/'compile-commands-original.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'started':time.time(),'commands':[],'sources':{}}
for rel in ['llama-model.cpp','llama-kv-cache.cpp','models/deepseek4.cpp']:
    original=ENGINE/'src'/rel
    saved=HERE/'src'/rel
    command=next(c for c in commands if c['file']==str(original))
    argv=shlex.split(command['command'])
    argv.insert(1, '-iquote'+str(original.parent))
    argv.insert(1, '-fmacro-prefix-map='+str(ENGINE/'src')+'/./llama-kv-cache.h='+str(ENGINE/'src/llama-kv-cache.h'))
    argv[argv.index('-o')+1]=str(HERE/'objects'/Path(rel+'.o'))
    argv[argv.index('-c')+1]=str(saved)
    argv.insert(1,'-fmacro-prefix-map='+str(HERE/'src')+'='+str(ENGINE/'src'))
    print('Compiling parent',rel,flush=True)
    with (HERE/'logs'/(Path(rel).name+'.log')).open('w') as log:
        subprocess.run(['nice','-n','10']+argv,cwd=command['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
    report['sources'][rel]={'sha256':sha(saved),'object_sha256':sha(HERE/'objects'/Path(rel+'.o'))}
    report['commands'].append(argv)
link=shlex.split((HERE/'link-original.txt').read_text())
link[link.index('-o')+1]=str(HERE/'build/libllama.parent.so.0.3.0')
for i,arg in enumerate(link):
    if arg.endswith('.o'):
        link[i]=str(HERE/'objects'/Path(arg).relative_to('CMakeFiles/llama.dir'))
print('Linking private parent',flush=True)
subprocess.run(link,cwd=ENGINE/'build-goal/src',check=True)
report['link_command']=link
report['parent_sha256']=sha(HERE/'build/libllama.parent.so.0.3.0')
report['expected_sha256']=meta['deployed_sha256']
report['byte_identical']=report['parent_sha256']==report['expected_sha256']
report['finished']=time.time()
(HERE/'parent-rebuild.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['parent_sha256','expected_sha256','byte_identical']}),flush=True)
