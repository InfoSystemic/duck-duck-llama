#!/usr/bin/env python3
"""Serial repeated production control, cache latency, and actual Codex file-edit check."""
import argparse
import json
from pathlib import Path
import subprocess
from capture_runtime import snapshot

HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
CATALOG=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')

def run(args):
    subprocess.run(args,cwd=HERE,check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    args=ap.parse_args()
    report={'pid':args.pid}
    before=snapshot(args.pid,18131)
    assert before['health']=={'status':'ok'}
    assert not any(s['is_processing'] for s in before['slots'])
    (HERE/'production-control-runtime.json').write_text(json.dumps(before,indent=2)+'\n')
    try:
        run(['bash',str(FLEET/'bench3.sh'),'18131','goal0919-control','256'])
        rows=[]
        for i in (1,2,3):
            a=json.loads((FLEET/f'results/bench3-goal0919-baseline-{i}.json').read_text())
            b=json.loads((FLEET/f'results/bench3-goal0919-control-{i}.json').read_text())
            rows.append({'prompt':i,'parity':a['content']==b['content'],'timings':b['timings']})
        report['short_control']=rows
        assert all(r['parity'] for r in rows)
        for tag,extra in [('control-alias-cold',[]),('control-alias-warm',['--warm'])]:
            run(['python3','-u',str(HERE/'appserver_bench.py'),'--catalog',str(CATALOG),'--tag',tag,*extra])
        warm=json.loads((HERE/'appserver-control-alias-warm.json').read_text())
        report['warm_cache_used']=warm['server_metrics']['cached_prompt_tokens']>0
        assert report['warm_cache_used'], 'Warm run did not reuse any prompt'
        assert not (HERE/'tool-smoke-0919/lru_cache.py').exists(), 'Do not overwrite an existing coding smoke result'
        run(['python3','-u',str(HERE/'appserver_tool_smoke.py'),'--catalog',str(CATALOG),'--tag','control-tool-smoke'])
        run(['python3','-I','-B',str(HERE/'test_lru_contract.py'),str(HERE/'tool-smoke-0919/lru_cache.py')])
        report['coding_contract']='PASS'
        run(['python3','-u',str(HERE/'stateful_gate.py'),'--tag','production-control'])
        report['stateful_gate']='PASS'
        after=snapshot(args.pid,18131)
        assert after['mapped_libraries_sha256']==before['mapped_libraries_sha256']
        assert after['environment']==before['environment']
        report['passed']=True
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        (HERE/'production-control.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PRODUCTION CONTROL PASSED',flush=True)

if __name__=='__main__': main()
