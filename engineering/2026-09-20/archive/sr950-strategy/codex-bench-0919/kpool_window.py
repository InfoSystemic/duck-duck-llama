#!/usr/bin/env python3
"""Flag-controlled kpool A/B on :18131. Restore the exact approved production runtime.
The optional mapped switch is changed only between idle, completed requests.
"""
import argparse,hashlib,json,os,signal,struct,subprocess,time
from pathlib import Path
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port
from thread_window import native,idle

HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
CAND=FLEET/'glm-kpool-wide-0919/build/libggml-cpu.so.0.22.0'
PROD=FLEET/'glm-cpu-fast-0919/libggml-cpu.so.0.22.0'
LAUNCHER=FLEET/'launch-glm-flash-native.sh'
PRODUCTION='glm53-flash-production.service'
UNIT='glm53-flash-kpool-wide-test-0919.service'
PORT=18131
CONTROL=HERE/'kpool-window-control.u32'
REPORT=HERE/'kpool-window-report.json'
LOG=HERE/'kpool-candidate-server.log'
CATALOG=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def run(args):subprocess.run(args,check=True,cwd=HERE)
def mode(value):
    idle(PORT)
    fd=os.open(CONTROL,os.O_WRONLY)
    try:assert os.pwrite(fd,struct.pack('<I',value),0)==4
    finally:os.close(fd)
    assert struct.unpack('<I',CONTROL.read_bytes())[0]==value
    print('Kpool mode:', 'wide' if value else 'deployed scalar',flush=True)
def codex(tag,warm=False,long=False,profile=False):
    idle(PORT)
    before=CONTROL.read_bytes() if CONTROL.exists() else None
    run(['python3','-u',str(HERE/'kpool_appserver_bench.py'),'--catalog',str(CATALOG),
        '--tag',tag,'--endpoint',f'http://127.0.0.1:{PORT}','--greedy','--timeout','2400',
        *(['--warm'] if warm else []),
        *(['--prompt-file',str(HERE/'kpool-long-prompt.txt')] if long else []),
        *(['--profile'] if profile else [])])
    if before is not None:assert CONTROL.read_bytes()==before,'Kernel control changed during a request'
    x=json.loads((HERE/f'appserver-{tag}.json').read_text())
    assert x['marker_pass'] and x['sampling_override']=={'temperature':0,'seed':42}
    assert x['request_instructions_chars']==504 and x['request_reasoning']=={'effort':'low'}
    if warm:assert x['server_metrics']['cached_prompt_tokens']>0
    total=x['server_metrics']['prompt_tokens']+x['server_metrics']['cached_prompt_tokens']
    if long:assert 28000<=total<=30720,('Unexpected long context',total)
    x.pop('prompt',None)
    return x

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production-pid',type=int,required=True)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    manifest=json.loads((HERE/'kpool-window-manifest.json').read_text())
    for path,sha in manifest['files'].items():assert digest(path)==sha,'Changed reviewed input: '+path
    assert digest(CAND)==manifest['candidate_sha256'] and digest(PROD)==manifest['production_sha256']
    gate=json.loads((CAND.parent.parent/'validation-1,2,3,15.json').read_text())
    assert gate['passed'] and gate['candidate_sha256']==digest(CAND)
    extra=json.loads((CAND.parent.parent/'validation-control-copy.json').read_text())
    assert extra['passed'] and extra['candidate_sha256']==digest(CAND)
    assert pid_for(PRODUCTION)==a.production_pid
    original=snapshot(a.production_pid,PORT);idle(PORT)
    assert original['mapped_libraries_sha256'][str(PROD)]==digest(PROD)
    if not a.execute:
        print(json.dumps({'preflight_passed':True,'production_unchanged':True,'candidate_sha256':digest(CAND),'port':PORT},indent=2));return
    assert not REPORT.exists() and not LOG.exists() and not CONTROL.exists(),'Refusing to overwrite an experiment'
    report={'candidate_sha256':digest(CAND),'production_pid_before':a.production_pid,'promoted':False,
        'known_cache_consistency_issue_unresolved':True,'port':PORT,'arms':{},'profiles':{},'mode_changes':[]}
    save(HERE/'kpool-window-original.json',original)
    stopped=False
    def checkpoint():save(REPORT,report)
    def change(value):
        mode(value);report['mode_changes'].append({'epoch':time.time(),'wide':bool(value)});checkpoint()
    def stop_signal(signum,frame):raise KeyboardInterrupt(f'Signal {signum}; restore production')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        report['production_codex_before']=codex('kpool-production-before');checkpoint()
        idle(PORT)
        current=snapshot(a.production_pid,PORT)
        for key in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:
            assert current[key]==original[key]
        assert pid_for(PRODUCTION)==a.production_pid
        fd=os.open(CONTROL,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        os.write(fd,struct.pack('<I',0));os.close(fd)
        stopped=True;report['window_started_epoch']=time.time();checkpoint()
        run(['systemctl','--user','stop',PRODUCTION]);wait_dead(a.production_pid)
        assert not open_port(PORT)
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
        assert available>380*1024*1024,'Insufficient memory headroom'
        run(['systemd-run','--user','--unit='+UNIT,'--collect','--property=MemoryMax=380G','--property=MemorySwapMax=0',
            '--property=TimeoutStopSec=120','--property=RuntimeMaxSec=3600',
            '--property=ExecStopPost=/usr/bin/systemctl --user start '+PRODUCTION,
            '--property=StandardOutput=append:'+str(LOG),'--property=StandardError=append:'+str(LOG),
            '--setenv=PORT='+str(PORT),'--setenv=LIB_PREPEND='+str(CAND.parent),
            '--setenv=GGML_CPU_GLM_POOL_FUSION=1','--setenv=GGML_CPU_CPY_FLAT=1',
            '--setenv=GGML_CPU_GLM_POOL_PROBE=1','--setenv=GGML_CPU_GLM_POOL_WIDE=1',
            '--setenv=GGML_CPU_GLM_POOL_WIDE_CONTROL_FILE='+str(CONTROL),str(LAUNCHER)])
        wait_healthy(PORT,UNIT)
        cand=snapshot(pid_for(UNIT),PORT);save(HERE/'kpool-window-candidate.json',cand)
        assert cand['mapped_libraries_sha256'].get(str(CAND))==digest(CAND)
        assert cand['command']==original['command']
        oldlibs={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
        newlibs={Path(k).name:v for k,v in cand['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
        assert oldlibs==newlibs,'Non-CPU runtime changed'
        env=dict(original['environment'])
        env.update(GGML_CPU_GLM_POOL_PROBE='1',GGML_CPU_GLM_POOL_WIDE='1',GGML_CPU_GLM_POOL_WIDE_CONTROL_FILE=str(CONTROL))
        env['LD_LIBRARY_PATH']=str(CAND.parent)+':'+env['LD_LIBRARY_PATH'].split(':',1)[1]
        assert cand['environment']==env,'Unexpected inference flag change'
        report['candidate_pid']=cand['pid'];checkpoint()
        for wide in [0,1]:
            change(wide)
            row=native(PORT,'kpool-native-'+str(wide));assert row['all_parity']
            report['arms']['native-'+str(wide)]=row;checkpoint()
        log=LOG.read_text(errors='replace');assert 'GLM_POOL_FUSED' in log and 'GLM_POOL_WIDE' in log
        mapping=[line for line in Path(f'/proc/{cand["pid"]}/maps').read_text().splitlines() if str(CONTROL) in line]
        assert mapping and all('r--s' in line for line in mapping)
        report['control_mapping']=mapping;report['engagement_verified']=True;checkpoint()
        run(['python3','-u',str(HERE/'stateful_gate.py'),'--endpoint',f'http://127.0.0.1:{PORT}',
             '--tag','kpool-wide','--reference',str(HERE/'stateful-production-control.json')])
        stateful=json.loads((HERE/'stateful-kpool-wide.json').read_text());assert stateful['regression_passed']
        report['stateful_regression_passed']=True;report['stateful_intrinsic_consistency']=stateful['passed'];checkpoint()
        for long in [False,True]:
            label='long' if long else 'short'
            cold_reference=None
            for wide in [0,1]:
                change(wide)
                row=codex(f'kpool-{label}-cold-{wide}',long=long)
                if cold_reference is None:cold_reference=row['output_text']
                assert row['output_text']==cold_reference,'Cold full-model output changed: '+label
                if not long:assert row['output_text']==report['production_codex_before']['output_text']
                row['wide']=bool(wide);report['arms'][f'{label}-cold-{wide}']=row;checkpoint()
            warm_reference=None;cache_reference=None
            for order,wide in enumerate([0,1,1,0]):
                change(wide)
                row=codex(f'kpool-{label}-warm-{order}-{wide}',warm=True,long=long)
                shape=(row['server_metrics']['prompt_tokens'],row['server_metrics']['cached_prompt_tokens'])
                if warm_reference is None:warm_reference=row['output_text'];cache_reference=shape
                assert row['output_text']==warm_reference,'Cached full-model output changed: '+label
                assert shape==cache_reference,'Cached prefix length changed between A/B arms'
                row['wide']=bool(wide);row['order']=order;row['cold_output_equal']=row['output_text']==cold_reference
                report['arms'][f'{label}-warm-{order}']=row;checkpoint()
        for wide in [0,1]:
            change(wide)
            offset=LOG.stat().st_size
            row=codex(f'kpool-long-profile-{wide}',warm=True,long=True,profile=True)
            trace=LOG.read_bytes()[offset:].decode(errors='replace')
            path=HERE/f'profile-kpool-long-{wide}.optrace.log'
            path.write_text('\n'.join(line for line in trace.splitlines() if 'CPU_OP_PROFILE' in line)+'\n')
            assert 'CPU_OP_PROFILE complete count=8' in trace,'Incomplete operation profile'
            target=HERE/f'profile-kpool-long-{wide}.optrace-summary.json'
            run(['python3',str(HERE/'summarize_optrace.py'),str(path),'--output',str(target)])
            parsed=json.loads(target.read_text());assert len(parsed['graphs'])==8
            report['profiles'][str(wide)]={'tag':row['tag'],'summary':str(target),'excluded_from_speed_measurements':True};checkpoint()
        report['completed']=True;checkpoint()
    except BaseException as exc:
        report['error']=repr(exc);checkpoint();raise
    finally:
        if stopped:
            print('Restoring approved production runtime',flush=True)
            child=pid_for(UNIT)
            subprocess.run(['systemctl','--user','stop',UNIT],check=False)
            if child:wait_dead(child)
            # ExecStopPost starts production even if this controller was killed.
            live=pid_for(PRODUCTION)
            if not live:
                assert not open_port(PORT),'Unexpected listener on production port'
                run(['systemctl','--user','start',PRODUCTION])
            wait_healthy(PORT,PRODUCTION)
            restored=snapshot(pid_for(PRODUCTION),PORT)
            for key in ['mapped_libraries_sha256','environment','command']:
                assert restored[key]==original[key],'Production restore mismatch: '+key
            save(HERE/'kpool-window-restored.json',restored)
            report.update(restored_healthy=True,restored_pid=restored['pid'],window_seconds=time.time()-report['window_started_epoch'])
            checkpoint();print('Production restored, PID',restored['pid'],flush=True)
    report['production_codex_after']=codex('kpool-production-after')
    assert report['production_codex_after']['output_text']==report['production_codex_before']['output_text']
    report['gate']='Completed and restored; no automatic promotion; pre-existing cache consistency issue retained'
    checkpoint()
if __name__=='__main__':main()
