#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time,urllib.request
HERE=Path(__file__).resolve().parent
BENCH=Path('/home/user/sr950-strategy/codex-bench-0919')
sys.path.insert(0,str(BENCH))
from capture_runtime import snapshot
from pool_window import pid_for
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def slots():
    with urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=5) as r:return json.load(r)
def idle():assert not any(s['is_processing'] for s in slots()),'Production became busy; stop private test'
def limits():
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arms',default='production,candidate-off,candidate-on,candidate-switch');ap.add_argument('--workers',type=int,choices=(1,15),default=1);a=ap.parse_args()
    original=snapshot(pid_for('glm53-flash-production.service'),18131);idle()
    expected=json.loads((BENCH/'kv-range-final-audit.json').read_text())
    for k in ['pid','proc_start_ticks','command','environment','mapped_libraries_sha256']:assert original[k]==expected[k],k
    assert not Path('/dev/shm/flash-optrace.arm').exists() and not Path('/dev/shm/glm-graph-phase.arm').exists()
    cmdline=original['command'];model=Path(cmdline[cmdline.index('--spec-draft-model')+1])
    assert model.stat().st_size==9258572704
    report_path=HERE/f'validation-w{a.workers}.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'started_at':time.time(),'workers':a.workers,'arms':[],'candidate_sha256':sha(HERE/'build/libllama.so.0.3.0'),'test_source_sha256':sha(HERE/'test_mtp_kv_only.cpp'),'test_sha256':sha(HERE/'build/test_mtp_kv_only')}
    assert report['candidate_sha256']==sha(HERE/'build/libllama.so.0.3.0') and report['test_sha256']==sha(HERE/'build/test_mtp_kv_only')
    save(HERE/f'validation-w{a.workers}-runtime-before.json',original)
    labels={'production':('reference','0'),'candidate-off':('build','0'),'candidate-on':('build','1'),'candidate-switch':('build','1')}
    try:
        for label in a.arms.split(','):
            assert label not in [x['label'] for x in report['arms']], 'Do not overwrite an arm'
            folder,flag=labels[label];idle()
            out_path=HERE/'build'/f'gate-w{a.workers}-{label}.bin';stdout_path=HERE/'logs'/f'gate-w{a.workers}-{label}.stdout';stderr_path=HERE/'logs'/f'gate-w{a.workers}-{label}.stderr'
            env=dict(os.environ,**original['environment'])
            env['LD_LIBRARY_PATH']=str(HERE/folder)+':'+original['environment']['LD_LIBRARY_PATH']
            env['GGML_GLM5N_MTP_KV_ONLY']=flag;env['GGML_GLM5N_MTP_KV_ONLY_PROBE']='1'
            env['GGML_CPU_NUMA_THREADS']=str(a.workers);env['GGML_CPU_REPACK_LOAD_THREADS']='4'
            env.pop('GGML_CPU_NUMA_THREADS_FILE',None);env.pop('LLAMA_GRAPH_PHASE_ARM_FILE',None)
            env.pop('GGML_GLM5N_MTP_KV_ONLY_CONTROL_FILE',None);env.pop('TEST_MTP_KV_CONTROL_FILE',None)
            if label=='candidate-switch':
                control=HERE/'build'/f'mode-w{a.workers}.u32';control.write_bytes(bytes(4))
                env['GGML_GLM5N_MTP_KV_ONLY_CONTROL_FILE']=str(control)
                env['TEST_MTP_KV_CONTROL_FILE']=str(control)
            for key in list(env):
                if key.startswith('GGML_CPU_OP_PROFILE'):env.pop(key)
            command=[str(HERE/'build/test_mtp_kv_only'),str(model),str(out_path),'4']
            row={'label':label,'library_sha256':sha(HERE/folder/'libllama.so.0.3.0'),'flag':flag,'command':command,'test_environment':{k:v for k,v in env.items() if k.startswith(('GGML_','LLAMA_')) or k=='LD_LIBRARY_PATH'},'max_rss_kib':0,'started_at':time.time()}
            report['arms'].append(row);save(report_path,report)
            print('Real MTP sidecar gate:',label,flush=True)
            with stdout_path.open('w') as out,stderr_path.open('w') as err:
                p=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limits,start_new_session=True)
                start=time.monotonic();last=start
                try:
                    while p.poll() is None:
                        idle()
                        assert pid_for('glm53-flash-production.service')==original['pid'],'Production PID changed'
                        status=Path(f'/proc/{p.pid}/status')
                        if status.exists():
                            rss=next((int(x.split()[1]) for x in status.read_text().splitlines() if x.startswith('VmRSS:')),0)
                            row['max_rss_kib']=max(row['max_rss_kib'],rss)
                            assert rss<32*1024*1024,'Private test exceeded 32 GiB RSS'
                        assert time.monotonic()-start<600,'Private MTP gate timed out'
                        if time.monotonic()-last>=30:
                            print(label,'active',round(time.monotonic()-start),'s, RSS',round(row['max_rss_kib']/1024),'MiB',flush=True);last=time.monotonic()
                        time.sleep(.5)
                finally:
                    if p.poll() is None:
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            row['returncode']=p.returncode;row['wall_seconds']=time.monotonic()-start;save(report_path,report)
            assert p.returncode==0,f'{label} failed; see {stderr_path}'
            row['result']=json.loads(stdout_path.read_text())
            row['output_bytes']=out_path.stat().st_size;row['output_sha256']=sha(out_path)
            if label!='production':assert out_path.read_bytes()==(HERE/f'build/gate-w{a.workers}-production.bin').read_bytes(),f'{label} output/cache mismatch'
            row['exact_parity']=True
            if label in ['candidate-on','candidate-switch']:
                probe=stderr_path.read_text();row['cache_only_graphs']=probe.count('GLM_MTP_KV_ONLY tokens=')
                assert row['cache_only_graphs']>0,'Candidate branch did not execute'
            print(label,'passed:',row['result'],'peak RSS MiB',round(row['max_rss_kib']/1024),flush=True)
            save(report_path,report)
        report['passed']=len(report['arms'])==4 and all(x.get('exact_parity') for x in report['arms'])
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        after=snapshot(pid_for('glm53-flash-production.service'),18131)
        for k in ['pid','proc_start_ticks','command','environment','mapped_libraries_sha256']:assert original[k]==after[k],k
        idle();report['production_unchanged_and_idle']=True
        save(HERE/f'validation-w{a.workers}-runtime-after.json',after)
        report['finished_at']=time.time();save(report_path,report)
if __name__=='__main__':main()
