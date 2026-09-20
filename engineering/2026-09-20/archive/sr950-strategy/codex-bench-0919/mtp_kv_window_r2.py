#!/usr/bin/env python3
"""Isolated GLM MTP cache-only catch-up A/B with exact production restoration on exit."""
import argparse,hashlib,json,os,signal,struct,subprocess,time
from pathlib import Path
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port
from thread_window import native,idle
import kpool_window as benchmark

HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
ROOT=FLEET/'glm-mtp-kv-only-0919'
BASELINE=FLEET/'glm-kv-range-0919/deploy-r2'
CAND=ROOT/'deploy/libllama.so.0.3.0'
CPU=FLEET/'glm-kpool-wide-0919/deploy/libggml-cpu.so.0.22.0'
LAUNCHER=FLEET/'launch-glm-flash-native.sh'
PRODUCTION='glm53-flash-production.service'
UNIT='glm53-flash-mtp-kv-only-r2-test-0919.service'
PORT=18141
PRODUCTION_PORT=18131
CONTROL=HERE/'mtp-kv-only-r2-window-control.u32'
ARM=Path('/dev/shm/glm-mtp-kv-r2-phase.arm')
REPORT=HERE/'mtp-kv-only-r2-window-report.json'
LOG=HERE/'mtp-kv-only-r2-candidate-server.log'
MANIFEST=HERE/'mtp-kv-only-r2-window-manifest.json'
benchmark.CONTROL=CONTROL

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')
def run(args):subprocess.run(args,cwd=HERE,check=True)
def codex(tag,warm=False,port=PORT):
    previous=benchmark.PORT
    benchmark.PORT=port
    try:return benchmark.codex(tag,warm=warm)
    finally:benchmark.PORT=previous

def command_at_port(command,port):
    result=command[:]
    result[result.index('--port')+1]=str(port)
    return result
def shape(row):
    m=row['server_metrics']
    return {key:m[key] for key in ['prompt_tokens','cached_prompt_tokens','generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']}

def mode(value):
    idle(PORT)
    assert not ARM.exists()
    fd=os.open(CONTROL,os.O_WRONLY|os.O_NOFOLLOW)
    try:assert os.pwrite(fd,struct.pack('<I',value),0)==4
    finally:os.close(fd)
    assert CONTROL.read_bytes()==struct.pack('<I',value)
    print('MTP cache-only mode:',value,flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production-pid',type=int,required=True)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    manifest=json.loads(MANIFEST.read_text())
    for path,sha in manifest['files'].items():assert digest(path)==sha,path
    assert digest(CAND)==manifest['candidate_sha256']
    for workers in [1,15]:
        gate=json.loads((ROOT/f'validation-w{workers}.json').read_text())
        assert gate['passed'] and len(gate['arms'])==4 and all(r['exact_parity'] for r in gate['arms'])
        assert any(r['label']=='candidate-switch' and r['cache_only_graphs']>0 for r in gate['arms'])
        assert all(r['library_sha256']==digest(CAND) for r in gate['arms'] if r['label'].startswith('candidate'))
    parent=json.loads((ROOT/'reference-build.json').read_text())
    assert parent['object_exact'] and parent['reference_library_sha256']==parent['expected_library_sha256']
    assert pid_for(PRODUCTION)==args.production_pid
    original=snapshot(args.production_pid,PRODUCTION_PORT);idle(PRODUCTION_PORT)
    assert any(Path(k).name=='libllama.so.0.3.0' and v==parent['expected_library_sha256'] for k,v in original['mapped_libraries_sha256'].items())
    assert original['mapped_libraries_sha256'][str(CPU)]==digest(CPU)
    assert original['environment']['GGML_CPU_GLM_POOL_WIDE']=='1'
    assert not ARM.exists() and not pid_for(UNIT) and not open_port(PORT)
    if not args.execute:
        print(json.dumps({'preflight_passed':True,'production_unchanged':True,'candidate_sha256':digest(CAND),'estimated_window_minutes':'15-20','automatic_restore':True},indent=2));return
    assert not any(p.exists() for p in [REPORT,LOG,CONTROL]),'Refusing to overwrite an experiment'
    report={'candidate_sha256':digest(CAND),'production_pid_before':args.production_pid,'promoted':False,
        'known_cache_consistency_issue_unresolved':True,'arms':{},'profiles':{},'mode_changes':[],'test_port':PORT,'production_port':PRODUCTION_PORT}
    save(HERE/'mtp-kv-only-r2-window-original.json',original)
    stopped=False
    def checkpoint():save(REPORT,report)
    def change(value):
        mode(value);report['mode_changes'].append({'epoch':time.time(),'kv_only':bool(value)});checkpoint()
    def stop_signal(signum,frame):raise KeyboardInterrupt(f'Signal {signum}; restore production')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        report['production_codex_before']=codex('mtp-kv-only-r2-production-before',port=PRODUCTION_PORT);checkpoint()
        idle(PRODUCTION_PORT)
        current=snapshot(args.production_pid,PRODUCTION_PORT)
        for key in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:
            assert current[key]==original[key],key
        fd=os.open(CONTROL,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:assert os.write(fd,struct.pack('<I',0))==4
        finally:os.close(fd)
        stopped=True;report['window_started_epoch']=time.time();checkpoint()
        run(['systemctl','--user','stop',PRODUCTION]);wait_dead(args.production_pid)
        assert not open_port(PRODUCTION_PORT) and not open_port(PORT)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>380*1024*1024
        run(['systemd-run','--user','--unit='+UNIT,'--collect','--property=MemoryMax=380G','--property=MemorySwapMax=0',
             '--property=TimeoutStopSec=120','--property=RuntimeMaxSec=1800',
             '--property=ExecStopPost=/usr/bin/systemctl --user start '+PRODUCTION,
             '--property=StandardOutput=append:'+str(LOG),'--property=StandardError=append:'+str(LOG),
             '--setenv=PORT='+str(PORT),'--setenv=LIB_PREPEND='+str(CAND.parent)+':'+str(BASELINE)+':'+str(CPU.parent),
             '--setenv=GGML_CPU_GLM_POOL_FUSION=1','--setenv=GGML_CPU_CPY_FLAT=1','--setenv=GGML_CPU_GLM_POOL_WIDE=1',
             '--setenv=LLAMA_KV_SEQ_RM_USED_PREFIX=1','--setenv=GGML_GLM5N_MTP_KV_ONLY=1',
             '--setenv=GGML_GLM5N_MTP_KV_ONLY_CONTROL_FILE='+str(CONTROL),'--setenv=LLAMA_GRAPH_PHASE_ARM_FILE='+str(ARM),str(LAUNCHER)])
        wait_healthy(PORT,UNIT)
        candidate=snapshot(pid_for(UNIT),PORT);save(HERE/'mtp-kv-only-r2-window-candidate.json',candidate)
        assert candidate['mapped_libraries_sha256'][str(CAND)]==digest(CAND)
        oldlibs={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items() if Path(k).name!='libllama.so.0.3.0'}
        newlibs={Path(k).name:v for k,v in candidate['mapped_libraries_sha256'].items() if Path(k).name!='libllama.so.0.3.0'}
        assert oldlibs==newlibs
        assert candidate['command']==command_at_port(original['command'],PORT)
        env=dict(original['environment'])
        env.update(GGML_GLM5N_MTP_KV_ONLY='1',GGML_GLM5N_MTP_KV_ONLY_CONTROL_FILE=str(CONTROL),LLAMA_GRAPH_PHASE_ARM_FILE=str(ARM))
        env['LD_LIBRARY_PATH']=str(CAND.parent)+':'+env['LD_LIBRARY_PATH']
        assert candidate['environment']==env,'Unexpected inference flag change'
        report['candidate_pid']=candidate['pid'];checkpoint()
        for value in [0,1]:
            change(value)
            row=native(PORT,'mtp-kv-only-r2-native-'+str(value));assert row['all_parity']
            report['arms']['native-'+str(value)]=row;checkpoint()
        mapping=[line for line in Path(f'/proc/{candidate["pid"]}/maps').read_text().splitlines() if str(CONTROL) in line]
        assert mapping and all('r--s' in line for line in mapping)
        report['control_mapping']=mapping;checkpoint()
        run(['python3','-u',str(HERE/'stateful_gate.py'),'--endpoint',f'http://127.0.0.1:{PORT}',
             '--tag','mtp-kv-only-r2','--reference',str(HERE/'stateful-production-control.json')])
        stateful=json.loads((HERE/'stateful-mtp-kv-only-r2.json').read_text())
        assert stateful['regression_passed']
        report['stateful_regression_passed']=True
        report['stateful_intrinsic_consistency']=stateful['passed'];checkpoint()
        for value in [0,1]:
            change(value);row=codex('mtp-kv-only-r2-cold-'+str(value))
            assert row['output_text']==report['production_codex_before']['output_text']
            row['kv_only']=bool(value);report['arms']['cold-'+str(value)]=row;checkpoint()
        reference=None;work=None
        for order,value in enumerate([0,1,1,0,1,0,0,1]):
            change(value);row=codex(f'mtp-kv-only-r2-warm-{order}-{value}',warm=True)
            if reference is None:reference=row['output_text'];work=shape(row)
            assert row['output_text']==reference and shape(row)==work,'A/B text or verification work changed'
            row['kv_only']=bool(value);row['order']=order
            report['arms']['warm-'+str(order)]=row;checkpoint()
        for value in [0,1]:
            change(value);offset=LOG.stat().st_size
            try:
                with ARM.open('x') as f:f.write('1\n')
                row=codex('mtp-kv-only-r2-phase-'+str(value),warm=True)
            finally:ARM.unlink(missing_ok=True)
            assert row['output_text']==reference and shape(row)==work
            trace=LOG.read_bytes()[offset:].decode(errors='replace')
            phase='\n'.join(line for line in trace.splitlines() if 'GRAPH_PHASE' in line or 'META_PHASE' in line)+'\n'
            assert 'GRAPH_PHASE' in phase and 'META_PHASE' in phase
            path=HERE/f'profile-mtp-kv-only-r2-{value}.phase.log';path.write_text(phase)
            from audit_glm_phases import parse
            graphs=parse(path)
            cache_only_graphs=[g for g in graphs if g['nodes'] in [15,16]]
            assert bool(cache_only_graphs)==bool(value),'Cache-only graph mode not confirmed'
            report['profiles'][str(value)]={'benchmark':row,'path':str(path),'kv_only':bool(value),
                'cache_only_graphs':len(cache_only_graphs),'excluded_from_speed_measurements':True};checkpoint()
        report['engagement_verified']=True
        report['completed']=True;checkpoint()
    except BaseException as exc:
        report['error']=repr(exc);checkpoint();raise
    finally:
        ARM.unlink(missing_ok=True)
        if stopped:
            print('Restoring approved production runtime',flush=True)
            child=pid_for(UNIT)
            subprocess.run(['systemctl','--user','stop',UNIT],check=False)
            if child:wait_dead(child)
            if not pid_for(PRODUCTION):
                assert not open_port(PRODUCTION_PORT)
                run(['systemctl','--user','start',PRODUCTION])
            wait_healthy(PRODUCTION_PORT,PRODUCTION)
            restored=snapshot(pid_for(PRODUCTION),PRODUCTION_PORT)
            for key in ['mapped_libraries_sha256','environment','command']:
                assert restored[key]==original[key],'Restore mismatch: '+key
            save(HERE/'mtp-kv-only-r2-window-restored.json',restored)
            report.update(restored_healthy=True,restored_pid=restored['pid'],window_seconds=time.time()-report['window_started_epoch'])
            checkpoint();print('Production restored, PID',restored['pid'],flush=True)
    report['production_codex_after']=codex('mtp-kv-only-r2-production-after',port=PRODUCTION_PORT)
    assert report['production_codex_after']['output_text']==report['production_codex_before']['output_text']
    report['gate']='Completed and restored; no automatic promotion; existing cache consistency issue retained'
    checkpoint()

if __name__=='__main__':main()
