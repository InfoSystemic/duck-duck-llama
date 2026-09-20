#!/usr/bin/env python3
"""Temporarily test 8-16 NUMA workers, then restore the approved production service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port,request

HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
PRODUCTION='glm53-flash-production.service'
UNIT='glm53-flash-threads-test-0919.service'
PORT=18141
CATALOG=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')
LIMIT=HERE/'thread-window-limit.txt'
LAUNCHER=HERE/'launch-glm-threads-0919.sh'
LIBRARY=FLEET/'glm-cpu-fast-0919/libggml-cpu.so.0.22.0'
REPORT=HERE/'thread-window-report.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(args):return subprocess.run(args,cwd=HERE,check=True)
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def idle(port):
    assert request(port,'health')=={'status':'ok'}
    assert not any(s['is_processing'] for s in request(port,'slots'))
    assert not Path('/dev/shm/flash-optrace.arm').exists()
def limit(n):
    idle(PORT)
    temporary=LIMIT.with_suffix('.tmp')
    temporary.write_text(str(n)+'\n');temporary.replace(LIMIT)
    print('NUMA worker control set to',n,flush=True)
def native(port,tag):
    idle(port)
    run(['bash',str(FLEET/'bench3.sh'),str(port),'goal0919-'+tag,'256'])
    rows=[]
    for i in (1,2,3):
        x=json.loads((FLEET/f'results/bench3-goal0919-{tag}-{i}.json').read_text())
        baseline=json.loads((FLEET/f'results/bench3-goal0919-baseline-{i}.json').read_text())
        assert x['timings']['predicted_n']==256
        rows.append({'prompt':i,'parity':x['content']==baseline['content'],'timings':x['timings']})
    return {'tag':tag,'rows':rows,'all_parity':all(r['parity'] for r in rows),
            'aggregate_tps':sum(r['timings']['predicted_n']-1 for r in rows)/sum(r['timings']['predicted_ms']/1000 for r in rows)}
def codex(port,tag,warm):
    idle(port)
    run(['python3','-u',str(HERE/'appserver_bench.py'),'--catalog',str(CATALOG),'--tag',tag,
         '--endpoint',f'http://127.0.0.1:{port}','--greedy',*(['--warm'] if warm else [])])
    x=json.loads((HERE/f'appserver-{tag}.json').read_text())
    assert x['marker_pass'] and x['sampling_override']=={'temperature':0,'seed':42}
    assert x['request_instructions_chars']==504 and x['request_reasoning']=={'effort':'low'}
    if warm:assert x['server_metrics']['cached_prompt_tokens']>0
    return x

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--production-pid',type=int,required=True)
    args=ap.parse_args()
    assert not REPORT.exists(), 'Do not overwrite a previous experiment'
    manifest=json.loads((HERE/'thread-window-manifest.json').read_text())
    assert digest(LAUNCHER)==manifest['private_launcher_sha256']
    assert digest(FLEET/'launch-glm-flash-native.sh')==manifest['original_launcher_sha256']
    assert digest(LIBRARY)==manifest['cpu_sha256']
    probe=json.loads((HERE/'thread-control-probe.json').read_text())
    assert probe['passed'] and probe['cases']==28 and probe['cpu_sha256']==manifest['cpu_sha256']
    assert pid_for(PRODUCTION)==args.production_pid and not open_port(PORT)
    original=snapshot(args.production_pid,18131);idle(18131)
    assert original['environment']['GGML_CPU_NUMA_THREADS']=='15'
    assert original['mapped_libraries_sha256'][str(LIBRARY)]==manifest['cpu_sha256']
    save(HERE/'thread-window-original.json',original)
    report={'original_pid':args.production_pid,'cpu_sha256':manifest['cpu_sha256'],
            'promoted':False,'worker_control_probe':'thread-control-probe.json',
            'known_production_cache_consistency_issue_unresolved':True,'native':[],'codex':[]}
    stopped=False
    def checkpoint():save(REPORT,report)
    def signal_stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}; restoring production')
    signal.signal(signal.SIGTERM,signal_stop)
    try:
        before=native(18131,'threads-production-before')
        assert before['all_parity'];report['production_before']=before;checkpoint()
        original=snapshot(args.production_pid,18131);idle(18131)
        assert pid_for(PRODUCTION)==args.production_pid
        LIMIT.write_text('15\n')
        run(['systemctl','--user','stop',PRODUCTION]);stopped=True
        wait_dead(args.production_pid);assert not open_port(18131)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>380*1024*1024
        run(['systemd-run','--user','--unit='+UNIT,'--collect',
             '--property=MemoryMax=380G','--property=MemorySwapMax=0','--property=TimeoutStopSec=120',
             '--property=RuntimeMaxSec=2700',
             '--property=StandardOutput=append:'+str(HERE/'thread-candidate-server.log'),
             '--property=StandardError=append:'+str(HERE/'thread-candidate-server.log'),
             '--setenv=PORT='+str(PORT),'--setenv=LIB_PREPEND='+str(LIBRARY.parent),
             '--setenv=GGML_CPU_GLM_POOL_FUSION=1','--setenv=GGML_CPU_CPY_FLAT=1',
             '--setenv=GGML_CPU_NUMA_THREADS_FILE='+str(LIMIT),str(LAUNCHER)])
        wait_healthy(PORT,UNIT)
        current=snapshot(pid_for(UNIT),PORT)
        save(HERE/'thread-window-candidate.json',current)
        assert current['mapped_libraries_sha256']==original['mapped_libraries_sha256']
        expected=list(original['command']);expected[expected.index('--port')+1]=str(PORT)
        assert current['command']==expected
        env=dict(original['environment']);env.update(GGML_CPU_NUMA_THREADS='16',GGML_CPU_NUMA_THREADS_FILE=str(LIMIT))
        assert current['environment']==env
        report['candidate_pid']=current['pid'];report['pool_capacity']=16
        # Begin/end controls detect drift and the effect of one extra inactive pool worker.
        for index,n in enumerate([15,16,12,8,14,10,15]):
            limit(n)
            row=native(PORT,f'threads-{index}-{n}');row.update(workers=n,order=index)
            report['native'].append(row);checkpoint()
            if n==15:assert row['all_parity'], '15-worker control output changed'
        limit(15)
        cold=codex(PORT,'threads-15-cold',False)
        report['codex_cold_control']={k:cold[k] for k in ['server_metrics','first_visible_delta_seconds','wall_seconds']}
        reference=None
        for index,n in enumerate([15,16,12,8,14,10,15]):
            limit(n)
            row=codex(PORT,f'threads-{index}-{n}-warm',True)
            if reference is None:reference=row['output_text']
            record={'workers':n,'order':index,'parity':row['output_text']==reference,
                    'server_metrics':row['server_metrics'],'first_visible_delta_seconds':row['first_visible_delta_seconds'],
                    'wall_seconds':row['wall_seconds'],'tag':row['tag']}
            report['codex'].append(record);checkpoint()
            if n==15:assert record['parity'], 'Repeated cached Codex control output changed'
        limit(15)
        report['experiment_completed']=True;checkpoint()
    except BaseException as exc:
        report['error']=repr(exc);checkpoint();raise
    finally:
        if stopped:
            print('Stopping temporary thread test and restoring approved production',flush=True)
            candidate_pid=pid_for(UNIT)
            subprocess.run(['systemctl','--user','stop',UNIT],check=False)
            if candidate_pid:wait_dead(candidate_pid)
            assert not open_port(PORT) and not open_port(18131)
            run(['systemctl','--user','start',PRODUCTION]);wait_healthy(18131,PRODUCTION)
            restored=snapshot(pid_for(PRODUCTION),18131)
            assert restored['mapped_libraries_sha256']==original['mapped_libraries_sha256']
            assert restored['environment']==original['environment'] and restored['command']==original['command']
            save(HERE/'thread-window-restored.json',restored)
            report.update(restored_healthy=True,restored_pid=restored['pid']);checkpoint()
            print('Approved production restored, PID',restored['pid'],flush=True)
        checkpoint()
    after=native(18131,'threads-production-after')
    assert after['all_parity'];report['production_after']=after
    report['production_control_ratio']=after['aggregate_tps']/report['production_before']['aggregate_tps']
    report['gate']='Completed and restored; no thread setting promoted'
    checkpoint()
    print(json.dumps({'production_before':report['production_before']['aggregate_tps'],
                      'production_after':after['aggregate_tps'],
                      'native':[(r['workers'],round(r['aggregate_tps'],3),r['all_parity']) for r in report['native']],
                      'codex':[(r['workers'],round(r['server_metrics']['decode_tokens_per_second'],3),r['parity']) for r in report['codex']]},indent=2),flush=True)

if __name__=='__main__':main()
