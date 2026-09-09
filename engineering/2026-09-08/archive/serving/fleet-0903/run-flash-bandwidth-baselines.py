#!/usr/bin/env python3
"""Measure one Flash server at a time after the preceding bandwidth stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from inference_contention_guard import InferenceContentionGuard, wait_for_idle

base=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=('raw','mtp'),default='raw')
options=parser.parse_args()
reference=base/('results/glm53-full-bandwidth-baseline-0905c/result.json' if options.mode=='raw'
                else 'results/qwen-flash-bandwidth-raw-baseline-0905-lifecycle/result.json')
while True:
    try:
        prior=json.loads(reference.read_text())
        if 'finished' in prior:
            assert not prior.get('error'), 'Preceding bandwidth stage failed'
            if options.mode=='raw':
                assert len(prior['measurements'])==4, 'Full measurement did not complete'
            else:
                assert prior.get('measurement_exit')==0, 'Raw Flash measurements did not complete'
            break
    except (FileNotFoundError,ValueError):pass
    time.sleep(5)

def snapshot():
    result={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            name=(proc/'comm').read_text().strip()
            if name!='llama-server' and not name.startswith(('glm-mtp-head','qwen-mtp-head')):continue
            f=(proc/'stat').read_text().rsplit(')',1)[1].split()
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            port=argv[argv.index(b'--port')+1].decode() if b'--port' in argv else None
            result[proc.name]=(int(f[11])+int(f[12]),f[19],port)
        except (OSError,ValueError,IndexError):pass
    return result

for model,manifest_name,port in [('glm','glm-flash-validated.json',18138),('qwen','qwen-flash-20tps.json',18139)]:
    if options.mode=='mtp':port+=2
    label=f'{model}-flash-bandwidth-{options.mode}-baseline-0905'
    lifecycle=base/'results'/(label+'-lifecycle')
    lifecycle.mkdir(exist_ok=False)
    state=dict(model=model,label=label,mode=options.mode,started=time.time())
    manifest=json.loads((base/manifest_name).read_text())
    argv=manifest['command'];command=[];i=0
    while i<len(argv):
        if options.mode=='raw' and argv[i].startswith('--spec-'):
            i+=2
        else:
            command.append(argv[i]);i+=1
    command[command.index('--port')+1]=str(port)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_GRAPH_PHASE','LLAMA_MTP_DRAFT_N_FILE','OMP_','GOMP_','REPACK_TEST_'))}
    env.update(manifest['runtime_env'])
    env['LD_LIBRARY_PATH']=str(Path(command[0]).parent)
    state.update(command=command,runtime_env=manifest['runtime_env'])
    state['idle_before_load']=wait_for_idle(snapshot,lifecycle/'waiting-for-idle.json')
    with (lifecycle/'server.log').open('w') as log:
        server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=server.pid
        (lifecycle/'server.pid').write_text(str(server.pid)+'\n')
        guard=InferenceContentionGuard(server,snapshot,lifecycle/'contention-guard.json');guard.start()
        try:
            started=time.monotonic()
            while True:
                assert server.poll() is None, 'Flash server exited during load'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2):pass
                    break
                except (OSError,ValueError):
                    if time.monotonic()-started>900:raise TimeoutError('Flash load exceeded 900s')
                    time.sleep(2)
            state['load_seconds']=time.monotonic()-started
            print(model,'healthy',round(state['load_seconds'],1),'seconds',flush=True)
            alias=command[command.index('--alias')+1]
            run=subprocess.run(['python3','-u',str(base/'measure-model-bandwidth.py'),label,
                                '--port',str(port),'--pid',str(server.pid),'--alias',alias,
                                '--drafts','0' if options.mode=='raw' else '2',
                                '--tokens','512','--skip-idle-gate'],timeout=900)
            state['measurement_exit']=run.returncode
            assert run.returncode==0,'Flash bandwidth measurement failed'
        except Exception as e:
            state['error']=repr(e)
            raise
        finally:
            state['contention']=guard.stop()
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill();server.wait()
            state.update(finished=time.time(),server_exit=server.returncode)
            (lifecycle/'result.json').write_text(json.dumps(state,indent=2)+'\n')
