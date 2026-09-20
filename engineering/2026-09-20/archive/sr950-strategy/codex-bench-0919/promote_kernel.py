#!/usr/bin/env python3
"""Apply the reviewed CPU-only service drop-in; restore original deployment on failure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port

HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
UNIT='glm53-flash-production.service'

def run(args): subprocess.run(args,cwd=HERE,check=True)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    args=ap.parse_args()
    manifest_path=HERE/'glm-kernel-promotion-manifest.json'
    m=json.loads(manifest_path.read_text())
    proof=json.loads((HERE/m['proof']).read_text())
    assert proof['restored_healthy'] and proof['short_prompt_parity']==[True]*3
    assert proof['engagement'] and proof['coding_contract']=='PASS' and proof['stateful_regression_gate']
    assert proof.get('error') in (None,"AssertionError('Requested MTP depth did not engage')")
    library=Path(m['library']);dropin=Path(m['dropin']);proposed=Path(m['proposed_dropin'])
    assert digest(library)==m['sha256'] and digest(proposed)==m['dropin_sha256']
    assert not dropin.exists(), 'Do not overwrite an existing service change'
    assert pid_for(UNIT)==args.pid
    original=snapshot(args.pid,18131)
    assert original['health']=={'status':'ok'} and not any(s['is_processing'] for s in original['slots'])
    assert not open_port(18141), 'Private candidate still running'
    assert not Path('/dev/shm/flash-optrace.arm').exists()
    (HERE/'promotion-before.json').write_text(json.dumps(original,indent=2)+'\n')
    report={'original_pid':args.pid,'library':str(library),'library_sha256':m['sha256'],
            'promoted':False,'known_production_cache_consistency_issue_preserved':True}
    stopped=False;installed=False;success=False
    try:
        run(['systemctl','--user','stop',UNIT]);stopped=True
        wait_dead(args.pid)
        assert not open_port(18131)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>380*1024*1024, 'Insufficient headroom for safe load'
        dropin.parent.mkdir(exist_ok=True)
        with dropin.open('xb') as f: f.write(proposed.read_bytes())
        installed=True
        run(['systemctl','--user','daemon-reload'])
        run(['systemctl','--user','start',UNIT])
        wait_healthy(18131,UNIT)
        current=snapshot(pid_for(UNIT),18131)
        assert current['command']==original['command']
        assert current['mapped_libraries_sha256'].get(str(library))==m['sha256']
        old={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
        new={Path(k).name:v for k,v in current['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
        assert old==new, 'Unexpected non-CPU library change'
        expected_env=dict(original['environment'])
        expected_env['LD_LIBRARY_PATH']=str(library.parent)+':'+expected_env['LD_LIBRARY_PATH']
        expected_env.update(GGML_CPU_GLM_POOL_FUSION='1',GGML_CPU_CPY_FLAT='1')
        assert current['environment']==expected_env, 'Unexpected inference flag change'
        (HERE/'promotion-running.json').write_text(json.dumps(current,indent=2)+'\n')
        run(['bash',str(FLEET/'bench3.sh'),'18131','goal0919-promoted','256'])
        rows=[]
        for i in (1,2,3):
            a=json.loads((FLEET/f'results/bench3-goal0919-baseline-{i}.json').read_text())
            b=json.loads((FLEET/f'results/bench3-goal0919-promoted-{i}.json').read_text())
            rows.append({'prompt':i,'parity':a['content']==b['content'],'timings':b['timings']})
        report['short_prompts']=rows
        assert all(x['parity'] for x in rows), 'Production output differs after deployment'
        catalog=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')
        for tag,extra in [('promoted-alias-cold',[]),('promoted-alias-warm',['--warm'])]:
            run(['python3','-u',str(HERE/'appserver_bench.py'),'--catalog',str(catalog),'--tag',tag,*extra])
        warm=json.loads((HERE/'appserver-promoted-alias-warm.json').read_text())
        assert warm['server_metrics']['cached_prompt_tokens']>0
        final=snapshot(current['pid'],18131)
        assert final['health']=={'status':'ok'} and not any(s['is_processing'] for s in final['slots'])
        assert final['mapped_libraries_sha256']==current['mapped_libraries_sha256']
        (HERE/'promotion-final.json').write_text(json.dumps(final,indent=2)+'\n')
        report.update(promoted=True,production_pid=final['pid'],healthy=True)
        m.update(state='APPLIED and verified',production_pid=final['pid'],report='promotion-report.json')
        manifest_path.write_text(json.dumps(m,indent=2)+'\n')
        success=True
        print('Faster CPU kernel deployed and verified, PID',final['pid'],flush=True)
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        (HERE/'promotion-report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not success and (stopped or installed):
            print('Promotion gate failed; restoring original service configuration',flush=True)
            pid=pid_for(UNIT)
            subprocess.run(['systemctl','--user','stop',UNIT],check=False)
            if pid: wait_dead(pid)
            assert not open_port(18131)
            if installed:
                assert digest(dropin)==m['dropin_sha256'], 'Concurrent drop-in edit; do not remove it'
                dropin.unlink()
            run(['systemctl','--user','daemon-reload'])
            run(['systemctl','--user','start',UNIT])
            wait_healthy(18131,UNIT)
            restored=snapshot(pid_for(UNIT),18131)
            assert restored['mapped_libraries_sha256']==original['mapped_libraries_sha256']
            assert restored['environment']==original['environment']
            report.update(rolled_back=True,restored_pid=restored['pid'],restored_healthy=True)
            (HERE/'promotion-report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
