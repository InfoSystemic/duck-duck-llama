#!/usr/bin/env python3
"""Complete saved MTP trace validation after a strict-parser failure; never alter the raw window report."""
import argparse, datetime, hashlib, json
from pathlib import Path
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
from audit_glm_phases import parse, VERIFY_NODES, DRAFT_NODES
from analyze_mtp_kv_r2 import profile
from mtp_kv_window_r2 import codex, shape

HERE=Path(__file__).resolve().parent
RAW=HERE/'mtp-kv-only-r2-window-report.json'
FINAL=HERE/'mtp-kv-only-r2-validation-report.json'
UNIT='glm53-flash-production.service'
TEST_UNIT='glm53-flash-mtp-kv-only-r2-test-0919.service'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def own_usage(row):
    m=row['server_metrics'];u=row['token_usage']['total']
    assert m['prompt_tokens']+m['cached_prompt_tokens']==u['inputTokens']
    assert m['cached_prompt_tokens']==u['cachedInputTokens']
    assert m['generated_tokens']==u['outputTokens']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    a=ap.parse_args()
    assert not FINAL.exists(),'Preserve completed validation'
    r=json.loads(RAW.read_text())
    assert r.get('restored_healthy') and r['restored_pid']==a.pid
    assert not r.get('completed') and 'nodes' in r.get('error','') and "'nodes': 15" in r['error']
    assert r['error'].startswith('AssertionError(')
    assert pid_for(UNIT)==a.pid and pid_for(TEST_UNIT)==0
    for arm in ['/dev/shm/glm-mtp-kv-r2-phase.arm','/dev/shm/glm-graph-phase.arm','/dev/shm/flash-optrace.arm']:
        assert not Path(arm).exists(),arm
    expected=json.loads((HERE/'mtp-kv-only-r2-window-original.json').read_text())
    current=snapshot(a.pid,18131);idle(18131)
    for key in ['command','environment','mapped_libraries_sha256']:assert current[key]==expected[key],key
    save(HERE/'mtp-kv-only-r2-restoration-audit.json',current)
    assert r['stateful_regression_passed'] and all(r['arms'][f'native-{v}']['all_parity'] for v in [0,1])
    warm=[r['arms'][f'warm-{i}'] for i in range(8)]
    reference=r['production_codex_before']['output_text']
    for row in warm:
        assert row['output_text']==reference and shape(row)==shape(warm[0])
        own_usage(row)
    for value in [0,1]:
        cold=r['arms'][f'cold-{value}'];assert cold['output_text']==reference;own_usage(cold)
    own_usage(r['production_codex_before'])
    provenance={str(RAW):digest(RAW)}
    for value in [0,1]:
        path=HERE/f'profile-mtp-kv-only-r2-{value}.phase.log'
        benchmark_path=HERE/f'appserver-mtp-kv-only-r2-phase-{value}.json'
        row=json.loads(benchmark_path.read_text());own_usage(row)
        assert row['output_text']==reference and shape(row)==shape(warm[0])
        graphs=parse(path,allowed_nodes=(VERIFY_NODES,DRAFT_NODES,15,16) if value else (VERIFY_NODES,DRAFT_NODES))
        catchups=[g for g in graphs if g['nodes'] in [15,16]]
        assert bool(catchups)==bool(value)
        item=dict(benchmark=row,path=str(path),kv_only=bool(value),cache_only_graphs=len(catchups),excluded_from_speed_measurements=True)
        analysis=profile(item)
        assert analysis['reported_verifications']==analysis['target_verifications']==105
        r['profiles'][str(value)]=item
        save(HERE/f'mtp-kv-only-r2-phase-analysis-{value}.json',analysis)
        provenance[str(path)]=digest(path);provenance[str(benchmark_path)]=digest(benchmark_path)
    r['production_codex_after']=codex('mtp-kv-only-r2-production-after',port=18131)
    assert r['production_codex_after']['output_text']==reference
    own_usage(r['production_codex_after'])
    final=snapshot(a.pid,18131);idle(18131)
    for key in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert final[key]==current[key],key
    save(HERE/'mtp-kv-only-r2-final-audit.json',final)
    r['controller_error']=r.pop('error')
    r.update(completed=True,engagement_verified=True,promoted=False,
        live_controller_completed=False,
        gate='All recorded gates validated offline after parser repair; exact production restoration and final control verified. No automatic promotion.',
        offline_completion=dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reason='The original strict phase parser allowed only 7152/89 nodes. The new 15-node cache-only graph was saved but rejected. Both profiles are now parsed with explicit mode-specific allowed shapes; no inference A/B was repeated.',
            source_files_sha256=provenance,raw_report_preserved=True))
    assert digest(RAW)==provenance[str(RAW)]
    save(FINAL,r)
    print('Saved completed validation:',FINAL,flush=True)

if __name__=='__main__':main()
