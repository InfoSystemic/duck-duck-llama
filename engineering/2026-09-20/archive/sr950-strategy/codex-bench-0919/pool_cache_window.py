#!/usr/bin/env python3
"""Private pooled-result-cache A/B with a bounded outage and exact restoration."""
import argparse,hashlib,json,os,signal,struct,subprocess,time
from pathlib import Path
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port
from thread_window import idle
HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
ROOT=FLEET/'glm-pool-cache-r2-0920'
CAND=ROOT/'deploy/libggml-cpu.so.0.22.0'
CPU=FLEET/'glm-kpool-wide-0919/deploy/libggml-cpu.so.0.22.0'
LAUNCHER=FLEET/'launch-glm-flash-native.sh'
PRODUCTION='glm53-flash-production.service'
PORT=18141
PRODUCTION_PORT=18131
CATALOG=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')
ARM=Path('/dev/shm/flash-optrace.arm')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')
def run(args):subprocess.run(args,cwd=HERE,check=True)
def shape(row):
    m=row['server_metrics'];return {k:m[k] for k in ['prompt_tokens','cached_prompt_tokens','generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--production-pid',type=int,required=True);ap.add_argument('--context',choices=['short','long'],required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    prefix='pool-cache-r2-'+args.context;unit='glm53-flash-'+prefix+'-test-0920.service'
    control=HERE/(prefix+'-control.u32');probe=HERE/(prefix+'-probe.u32');report_path=HERE/(prefix+'-window-report.json');log=HERE/(prefix+'-server.log');manifest_path=HERE/(prefix+'-window-manifest.json')
    manifest=json.loads(manifest_path.read_text())
    for path,sha in manifest['files'].items():assert digest(path)==sha,path
    assert digest(CAND)==manifest['candidate_sha256']
    gate=json.loads((ROOT/'validation-report.json').read_text());bench=json.loads((ROOT/'numa-report.json').read_text());build=json.loads((ROOT/'build-report.json').read_text())
    assert build['passed'] and build['parent_library_exact'] and build['parent_library_sha256']==digest(CPU)
    assert gate['passed'] and len(gate['stateful'])==12 and all(x['exact'] for x in gate['stateful'])
    assert bench['passed'] and len(bench['runs'])==4 and all(x['summary']['median_speedup']>2 for x in bench['runs'] if 'summary' in x)
    assert gate['candidate_sha256']==bench['candidate_sha256']==digest(CAND)
    if args.context=='long':
        short=json.loads((HERE/'pool-cache-r2-short-window-report.json').read_text());assert short['completed'] and short['restored_healthy'] and short['stateful_regression_passed'] and short['candidate_sha256']==digest(CAND)
    assert pid_for(PRODUCTION)==args.production_pid
    original=snapshot(args.production_pid,PRODUCTION_PORT);idle(PRODUCTION_PORT)
    assert original['mapped_libraries_sha256'][str(CPU)]==digest(CPU)
    assert original['environment']['GGML_CPU_GLM_POOL_WIDE']=='1' and original['environment']['GGML_GLM5N_MTP_KV_ONLY']=='1'
    assert not pid_for(unit) and not open_port(PORT) and not ARM.exists()
    if not args.execute:
        print(json.dumps({'preflight_passed':True,'production_unchanged':True,'candidate_sha256':digest(CAND),'context':args.context,'private_port':PORT,'private_runtime_limit_seconds':1080,'automatic_restore':True},indent=2));return
    assert not any(p.exists() for p in [report_path,log,control,probe]),'Refusing to overwrite an experiment'
    report={'candidate_sha256':digest(CAND),'production_pid_before':args.production_pid,'promoted':False,'completed':False,'known_cache_consistency_issue_unresolved':True,'context':args.context,'arms':{},'profiles':{},'mode_changes':[],'test_port':PORT,'production_port':PRODUCTION_PORT}
    save(HERE/(prefix+'-original.json'),original);stopped=False
    def checkpoint():save(report_path,report)
    def word(path,value):
        idle(PORT);fd=os.open(path,os.O_WRONLY|os.O_NOFOLLOW)
        try:assert os.pwrite(fd,struct.pack('<I',value),0)==4
        finally:os.close(fd)
        assert path.read_bytes()==struct.pack('<I',value)
    def change(value):
        assert probe.read_bytes()==struct.pack('<I',0);word(control,value);report['mode_changes'].append({'epoch':time.time(),'cache':bool(value)});checkpoint();print('Pool result cache mode',value,flush=True)
    def codex(tag,warm=False,long=False,profile=False,port=PORT):
        idle(port);saved=[p.read_bytes() if p.exists() else None for p in [control,probe]]
        run(['python3','-u',str(HERE/'kpool_appserver_bench.py'),'--catalog',str(CATALOG),'--tag',tag,'--endpoint',f'http://127.0.0.1:{port}','--greedy','--timeout','1100',*(['--warm'] if warm else []),*(['--prompt-file',str(HERE/'kpool-long-prompt.txt')] if long else []),*(['--profile'] if profile else [])])
        assert saved==[p.read_bytes() if p.exists() else None for p in [control,probe]],'Mode changed during request'
        row=json.loads((HERE/f'appserver-{tag}.json').read_text());assert row['marker_pass'] and row['sampling_override']=={'temperature':0,'seed':42}
        assert row['request_instructions_chars']==504 and row['request_reasoning']=={'effort':'low'}
        m=row['server_metrics'];u=row['token_usage']['total'];actual={'input':m['prompt_tokens']+m['cached_prompt_tokens'],'cached':m['cached_prompt_tokens'],'output':m['generated_tokens']};expected={'input':u['inputTokens'],'cached':u['cachedInputTokens'],'output':u['outputTokens']};assert actual==expected,('Foreign request accounting',actual,expected)
        if warm:assert m['cached_prompt_tokens']>0
        if long:assert 28000<=actual['input']<=30720
        row['request_accounting_verified']=True;row.pop('prompt',None);return row
    def native(value):
        tag=prefix+'-native-'+str(value);idle(PORT);run(['bash',str(FLEET/'bench3.sh'),str(PORT),'goal0919-'+tag,'256']);rows=[]
        for i in [1,2,3]:
            r=json.loads((FLEET/f'results/bench3-goal0919-{tag}-{i}.json').read_text());reference=json.loads((FLEET/f'results/bench3-goal0919-baseline-{i}.json').read_text());assert r['content']==reference['content'] and r['timings']['predicted_n']==256;rows.append({'prompt':i,'parity':True,'timings':r['timings']})
        return {'tag':tag,'all_parity':True,'rows':rows}
    def stop_signal(signum,frame):raise TimeoutError(f'Signal {signum}; restore production')
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGALRM,stop_signal)
    try:
        report['production_codex_before']=codex(prefix+'-production-before',port=PRODUCTION_PORT);checkpoint();idle(PRODUCTION_PORT)
        current=snapshot(args.production_pid,PRODUCTION_PORT)
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert current[k]==original[k],k
        for path in [control,probe]:
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            try:assert os.write(fd,struct.pack('<I',0))==4
            finally:os.close(fd)
        stopped=True;report['window_started_epoch']=time.time();checkpoint();signal.alarm(1050)
        run(['systemctl','--user','stop',PRODUCTION]);wait_dead(args.production_pid)
        assert not open_port(PRODUCTION_PORT) and not open_port(PORT)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')));assert available>380*1024*1024
        prepend=':'.join(str(x) for x in [CAND.parent,FLEET/'glm-mtp-kv-only-0919/deploy',FLEET/'glm-kv-range-0919/deploy-r2',CPU.parent])
        flags={'GGML_CPU_GLM_POOL_FUSION':'1','GGML_CPU_CPY_FLAT':'1','GGML_CPU_GLM_POOL_WIDE':'1','LLAMA_KV_SEQ_RM_USED_PREFIX':'1','GGML_GLM5N_MTP_KV_ONLY':'1','GGML_CPU_GLM_POOL_CACHE':'1','GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE':str(control),'GGML_CPU_GLM_POOL_CACHE_PROBE':'1','GGML_CPU_GLM_POOL_CACHE_PROBE_CONTROL_FILE':str(probe)}
        run(['systemd-run','--user','--unit='+unit,'--collect','--property=MemoryMax=380G','--property=MemorySwapMax=0','--property=TimeoutStopSec=120','--property=RuntimeMaxSec=1080','--property=ExecStopPost=/usr/bin/systemctl --user start '+PRODUCTION,'--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log),'--setenv=PORT='+str(PORT),'--setenv=LIB_PREPEND='+prepend,*['--setenv='+k+'='+v for k,v in flags.items()],str(LAUNCHER)])
        wait_healthy(PORT,unit,timeout=600);candidate=snapshot(pid_for(unit),PORT);save(HERE/(prefix+'-candidate.json'),candidate)
        assert candidate['mapped_libraries_sha256'][str(CAND)]==digest(CAND)
        oldlibs={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items() if Path(k).name!='libggml-cpu.so.0.22.0'};newlibs={Path(k).name:v for k,v in candidate['mapped_libraries_sha256'].items() if Path(k).name!='libggml-cpu.so.0.22.0'};assert oldlibs==newlibs
        command=original['command'][:];command[command.index('--port')+1]=str(PORT);assert candidate['command']==command
        env=dict(original['environment']);env.update(flags);env['LD_LIBRARY_PATH']=str(CAND.parent)+':'+env['LD_LIBRARY_PATH'];assert candidate['environment']==env,'Unexpected inference environment'
        report['candidate_pid']=candidate['pid'];checkpoint()
        if args.context=='short':
            for value in [0,1]:change(value);report['arms']['native-'+str(value)]=native(value);checkpoint()
            run(['python3','-u',str(HERE/'stateful_gate.py'),'--endpoint',f'http://127.0.0.1:{PORT}','--tag',prefix,'--reference',str(HERE/'stateful-production-control.json')]);state=json.loads((HERE/f'stateful-{prefix}.json').read_text());assert state['regression_passed'];report['stateful_regression_passed']=True;report['stateful_intrinsic_consistency']=state['passed'];checkpoint()
        long=args.context=='long';cold_reference=None
        for value in ([0] if long else [0,1]):
            change(value);row=codex(prefix+'-cold-'+str(value),long=long)
            if cold_reference is None:cold_reference=row['output_text']
            assert row['output_text']==cold_reference
            if not long:assert row['output_text']==report['production_codex_before']['output_text']
            row['cache']=bool(value);report['arms']['cold-'+str(value)]=row;checkpoint()
        report['cold_on_measured']=not long
        reference=None;work=None
        for order,value in enumerate([0,1,1,0,1,0,0,1]):
            change(value);row=codex(f'{prefix}-warm-{order}-{value}',warm=True,long=long)
            if reference is None:reference=row['output_text'];work=shape(row)
            assert row['output_text']==reference and shape(row)==work,'A/B text or verification work changed'
            row.update(cache=bool(value),order=order,cold_output_equal=row['output_text']==cold_reference);report['arms']['warm-'+str(order)]=row;checkpoint()
        for value in [0,1]:
            change(value);offset=log.stat().st_size
            try:
                word(probe,value);row=codex(prefix+'-profile-'+str(value),warm=True,long=long,profile=True)
            finally:
                ARM.unlink(missing_ok=True)
                if pid_for(unit):word(probe,0)
            assert row['output_text']==reference and shape(row)==work
            trace=log.read_bytes()[offset:].decode(errors='replace');assert 'CPU_OP_PROFILE complete count=8' in trace
            if value:assert 'GLM_POOL_CACHE_HIT' in trace,'No live cache hit confirmed'
            trace_path=HERE/f'profile-{prefix}-{value}.optrace.log';trace_path.write_text('\n'.join(x for x in trace.splitlines() if 'CPU_OP_PROFILE' in x or 'GLM_POOL_CACHE_HIT' in x)+'\n');target=HERE/f'profile-{prefix}-{value}.optrace-summary.json'
            run(['python3',str(HERE/'summarize_optrace.py'),str(trace_path),'--output',str(target)]);parsed=json.loads(target.read_text());assert len(parsed['graphs'])==8
            report['profiles'][str(value)]={'benchmark':row,'summary':str(target),'trace':str(trace_path),'cache':bool(value),'excluded_from_speed_measurements':True};checkpoint()
        mapping=[x for x in Path(f'/proc/{candidate["pid"]}/maps').read_text().splitlines() if str(control) in x or str(probe) in x];assert len(mapping)==2 and all('r--s' in x for x in mapping)
        report['control_mappings']=mapping;report['candidate_after']=snapshot(candidate['pid'],PORT);report['engagement_verified']=True;report['completed']=True;checkpoint()
    except BaseException as exc:report['error']=repr(exc);checkpoint();raise
    finally:
        signal.alarm(0);ARM.unlink(missing_ok=True)
        if stopped:
            print('Restoring approved production runtime',flush=True);child=pid_for(unit);subprocess.run(['systemctl','--user','stop',unit],check=False)
            if child:wait_dead(child)
            if not pid_for(PRODUCTION):assert not open_port(PRODUCTION_PORT);run(['systemctl','--user','start',PRODUCTION])
            wait_healthy(PRODUCTION_PORT,PRODUCTION);restored=snapshot(pid_for(PRODUCTION),PRODUCTION_PORT)
            for k in ['mapped_libraries_sha256','environment','command']:assert restored[k]==original[k],'Restore mismatch: '+k
            save(HERE/(prefix+'-restored.json'),restored);report.update(restored_healthy=True,restored_pid=restored['pid'],window_seconds=time.time()-report['window_started_epoch']);checkpoint();print('Production restored, PID',restored['pid'],flush=True)
    report['production_codex_after']=codex(prefix+'-production-after',port=PRODUCTION_PORT);assert report['production_codex_after']['output_text']==report['production_codex_before']['output_text'];report['gate']='Completed and restored; not promoted; existing consistency issue remains';checkpoint()
if __name__=='__main__':main()
