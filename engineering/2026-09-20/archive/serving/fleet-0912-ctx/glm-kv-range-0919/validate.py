#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,resource,subprocess,sys,time,urllib.request
HERE=Path(__file__).resolve().parent
BENCH=Path('/home/user/sr950-strategy/codex-bench-0919')
sys.path.insert(0,str(BENCH))
from capture_runtime import snapshot
from pool_window import pid_for
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=snapshot(pid_for('glm53-flash-production.service'),18131)
assert original['pid']>0
assert any('glm-kpool-wide-0919/deploy/libggml-cpu' in path and sha=='3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3' for path,sha in original['mapped_libraries_sha256'].items())
assert not any(s['is_processing'] for s in original['slots'])
assert sha(HERE/'reference/libllama.so.0.3.0')=='8c794722eccdcc27aac88def7f1a1d549926a0d6aa3dd00586d6fdca5c8eb08a'
(HERE/'validation-runtime-before.json').write_text(json.dumps(original,indent=2)+'\n')
report={'started':time.time(),'production_pid':original['pid'],'arms':[]}

def limits():
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def idle():
    with urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=5) as r: slots=json.load(r)
    assert not any(s['is_processing'] for s in slots),'Production busy; stop private test'

for label,folder,flag in [('production','reference','0'),('candidate-off','candidate-lib','0'),('candidate-on','candidate-lib','1'),('candidate-switch','candidate-lib','1')]:
    idle()
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(HERE/folder)+':'+original['environment']['LD_LIBRARY_PATH']
    env['LLAMA_KV_SEQ_RM_USED_PREFIX']=flag
    env['LLAMA_KV_SEQ_RM_PROBE']='1'
    env.pop('LLAMA_KV_SEQ_RM_CONTROL_FILE',None)
    env.pop('TEST_KV_MODE_CONTROL',None)
    if label=='candidate-switch':
        control=HERE/'build/test-control.u32'
        control.write_bytes(bytes(4))
        env['LLAMA_KV_SEQ_RM_CONTROL_FILE']=str(control)
        env['TEST_KV_MODE_CONTROL']=str(control)
    row={'label':label,'flag':flag,'library_sha256':sha(HERE/folder/'libllama.so.0.3.0')}
    output=HERE/'build'/f'state-{label}.bin'
    print('Checking cache-state parity:',label,flush=True)
    with (HERE/'logs'/f'parity-{label}.stdout').open('w') as out, (HERE/'logs'/f'parity-{label}.stderr').open('w') as err:
        completed=subprocess.run([str(HERE/'build/test_kv_range'),'parity',str(output)],env=env,stdout=out,stderr=err,preexec_fn=limits,timeout=180)
    row['parity_returncode']=completed.returncode
    report['arms'].append(row)
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert completed.returncode==0, f'{label} parity failed; inspect saved stderr'
    row['parity']=json.loads((HERE/'logs'/f'parity-{label}.stdout').read_text())
    row['state_bytes']=output.stat().st_size
    row['state_sha256']=sha(output)
    if label!='production':
        assert output.read_bytes()==(HERE/'build/state-production.bin').read_bytes(),f'{label} state differs'
    row['exact_parity']=True
    if label.startswith('candidate'):
        probe=(HERE/'logs'/f'parity-{label}.stderr').read_text()
        required=[0,1] if label=='candidate-switch' else [int(flag)]
        assert all(f'KV_SEQ_RM_MODE bounded={mode}' in probe for mode in required)
        row['engagement_verified']=True
    if label=='candidate-switch':
        (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,row['parity'],row['state_bytes'],'bytes exact; both modes observed',flush=True)
        continue
    idle()
    print('Measuring rollback scan:',label,flush=True)
    with (HERE/'logs'/f'bench-{label}.stdout').open('w') as out, (HERE/'logs'/f'bench-{label}.stderr').open('w') as err:
        subprocess.run([str(HERE/'build/test_kv_range'),'bench','unused'],env=env,stdout=out,stderr=err,preexec_fn=limits,timeout=120,check=True)
    row['benchmark']=json.loads((HERE/'logs'/f'bench-{label}.stdout').read_text())
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(label,row['parity'],row['state_bytes'],'bytes exact',flush=True)
final=snapshot(pid_for('glm53-flash-production.service'),18131)
for key in ['pid','proc_start_ticks','command','environment','mapped_libraries_sha256']:
    assert final[key]==original[key],key
(HERE/'validation-runtime-after.json').write_text(json.dumps(final,indent=2)+'\n')
report['production_unchanged']=True
report['passed']=True
report['max_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
report['finished']=time.time()
(HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('All cache-state parity arms passed; production unchanged',flush=True)
