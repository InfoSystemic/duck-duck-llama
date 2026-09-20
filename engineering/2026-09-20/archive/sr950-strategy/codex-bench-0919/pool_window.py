#!/usr/bin/env python3
"""Serial GLM pool candidate window, with identity gates and production restore.

The candidate uses an unadvertised port. No simultaneous model loads are allowed.
Run only after standalone numerical parity has passed and live requests are idle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
from capture_runtime import snapshot
from appserver_bench import metrics
from summarize_metrics import summarize

HERE = Path(__file__).resolve().parent
FLEET = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
CANDIDATE = FLEET / 'glm-pool-0919/build/libggml-cpu.so.0.22.0'
UNIT = 'glm53-flash-pool-test-0919.service'
PRODUCTION = 'glm53-flash-production.service'
PORT = 18141


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def request(port, route):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}', timeout=5) as r:
        return json.load(r)


def open_port(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


def pid_for(unit):
    result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True,capture_output=True)
    return int(result.stdout.strip() or '0')


def wait_dead(pid, timeout=180):
    end = time.monotonic()+timeout
    while Path(f'/proc/{pid}').exists():
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:
            return
        if stat[0] == 'Z': return
        if time.monotonic() > end: raise TimeoutError(f'PID {pid} has not exited; no second model will be loaded')
        time.sleep(1)


def wait_healthy(port, unit, timeout=900):
    start=time.monotonic(); report=start
    while time.monotonic()-start < timeout:
        try:
            if request(port,'health').get('status')=='ok': return
        except Exception: pass
        if not pid_for(unit): raise RuntimeError(f'{unit} has no live process')
        if time.monotonic()-report > 30:
            print(f'{unit}: loading {time.monotonic()-start:.0f}s',flush=True);report=time.monotonic()
        time.sleep(2)
    raise TimeoutError(f'{unit} readiness timeout; process remains owned by its unit')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production-pid',type=int,required=True)
    ap.add_argument('--candidate-sha256',required=True)
    ap.add_argument('--variant', choices=['pool','copy','combo'], default='pool')
    ap.add_argument('--short-client', action='store_true', help='Only measure the corrected-catalog Codex request')
    ap.add_argument('--mtp-sweep', action='store_true', help='Compare MTP1 and repeated MTP2 through the existing bounded draft-limit hook')
    ap.add_argument('--stateful-reference', type=Path, help='Run cache/concurrency gates and compare to this production reference')
    ap.add_argument('--tool-smoke', action='store_true', help='Verify an actual Codex file edit and its functional contract')
    ap.add_argument('--profile', action='store_true', help='After speed checks, capture one bounded operation trace on the private candidate')
    args=ap.parse_args()
    if args.stateful_reference:
        stateful_ref=json.loads(args.stateful_reference.read_text())
        assert set(stateful_ref['cases'])=={'initial','append_cached','append_fresh','diverge_cached','diverge_fresh',
                                          'single_0','single_1','concurrent_0','concurrent_1'}, 'Incomplete production reference'
        assert stateful_ref['checks']['two_slots_active'], 'Reference did not exercise both slots'
    global CANDIDATE,UNIT
    CANDIDATE=FLEET / ('glm-' + {'pool':'pool','copy':'copy','combo':'pool-copy'}[args.variant] + '-0919/build/libggml-cpu.so.0.22.0')
    UNIT='glm53-flash-' + args.variant + '-test-0919.service'
    flags={}
    if args.variant in ('pool','combo'): flags.update(GGML_CPU_GLM_POOL_FUSION='1', GGML_CPU_GLM_POOL_PROBE='1', GGML_CPU_CPY_ADDRESS_PROBE='1')
    if args.variant in ('copy','combo'): flags.update(GGML_CPU_CPY_FLAT='1', GGML_CPU_CPY_FLAT_PROBE='1')
    stem=args.variant
    draft_file=HERE/f'{stem}-mtp-draft-limit.txt'
    if args.mtp_sweep:
        common=Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin/libllama-common.so.0.3.0')
        assert b'LLAMA_MTP_DRAFT_N_FILE' in common.read_bytes(), 'Runtime has no draft limit hook'
        draft_file.write_text('2\n')
        flags['LLAMA_MTP_DRAFT_N_FILE']=str(draft_file)
    with CANDIDATE.open('rb') as f: candidate_sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert candidate_sha==args.candidate_sha256, 'Candidate changed after review'
    original=snapshot(args.production_pid,18131)
    assert original['health']=={'status':'ok'}
    assert not any(s['is_processing'] for s in original['slots']), 'Active production request'
    assert not open_port(PORT), 'Candidate port already occupied'
    assert not Path('/dev/shm/flash-optrace.arm').exists(), 'Disable op profiling for speed measurement'
    (HERE/f'{stem}-window-original.json').write_text(json.dumps(original,indent=2)+'\n')
    report={'candidate_sha256':candidate_sha,'original_pid':args.production_pid,'candidate_promoted':False}
    stopped=False
    try:
        # Exact PID, verified executable and command by snapshot; no broad pkill.
        assert Path(f'/proc/{args.production_pid}/stat').read_text().rsplit(')',1)[1].split()[19]==original['proc_start_ticks']
        if pid_for(PRODUCTION)==args.production_pid:
            run(['systemctl','--user','stop',PRODUCTION])
        else:
            os.kill(args.production_pid,signal.SIGTERM)
        stopped=True
        wait_dead(args.production_pid)
        assert not open_port(18131), 'Production port still occupied'
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>380*1024*1024, 'Insufficient free headroom; refusing candidate load'
        run(['systemd-run','--user','--unit='+UNIT,'--collect',
             '--property=MemoryMax=380G','--property=MemorySwapMax=0',
             '--property=TimeoutStopSec=120', '--property=RuntimeMaxSec=1800',
             '--property=StandardOutput=append:'+str(HERE/f'{stem}-candidate-server.log'),
             '--property=StandardError=append:'+str(HERE/f'{stem}-candidate-server.log'),
             '--setenv=PORT='+str(PORT), '--setenv=LIB_PREPEND='+str(CANDIDATE.parent),
             *['--setenv='+k+'='+v for k,v in flags.items()], str(FLEET/'launch-glm-flash-native.sh')])
        wait_healthy(PORT,UNIT)
        current=snapshot(pid_for(UNIT),PORT)
        (HERE/f'{stem}-window-candidate.json').write_text(json.dumps(current,indent=2)+'\n')
        assert current['mapped_libraries_sha256'].get(str(CANDIDATE))==candidate_sha
        expected_cmd=list(original['command']);expected_cmd[expected_cmd.index('--port')+1]=str(PORT)
        assert current['command']==expected_cmd, 'Command-line confound'
        allowed={'LD_LIBRARY_PATH'} | set(flags)
        delta={k:[original['environment'].get(k),current['environment'].get(k)] for k in original['environment'].keys()|current['environment'].keys() if original['environment'].get(k)!=current['environment'].get(k)}
        report['environment_delta']=delta
        assert set(delta)<=allowed, 'Unexpected inference configuration change'
        oldlibs={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items()}
        newlibs={Path(k).name:v for k,v in current['mapped_libraries_sha256'].items()}
        assert {k:v for k,v in oldlibs.items() if 'libggml-cpu' not in k}=={k:v for k,v in newlibs.items() if 'libggml-cpu' not in k}, 'Non-CPU library changed'
        run(['bash',str(FLEET/'bench3.sh'),str(PORT),'goal0919-'+stem, '256'])
        matches=[]
        for i in (1,2,3):
            before=json.loads((FLEET/f'results/bench3-goal0919-baseline-{i}.json').read_text())
            after=json.loads((FLEET/f'results/bench3-goal0919-{stem}-{i}.json').read_text())
            matches.append(before['content']==after['content'])
        report['short_prompt_parity']=matches
        assert all(matches), 'Full-model generated text changed; reject candidate'
        log=(HERE/f'{stem}-candidate-server.log').read_text(errors='replace')
        if args.variant in ('pool','combo'): assert 'GLM_POOL_FUSED' in log, 'No full-model pool engagement'
        if args.variant in ('copy','combo'): assert 'CPY_FLAT' in log, 'No full-model copy engagement'
        report['engagement']=True
        probes=[('glm-5.3-flash.catalog.proposed.json',stem+'-alias-cold')]
        if not args.short_client: probes.insert(0,('glm-5.3-flash.catalog.before.json',stem+'-old-cold'))
        for catalog,tag in probes:
            run(['python3',str(HERE/'appserver_bench.py'),'--catalog',str(HERE/catalog),'--tag',tag,'--endpoint',f'http://127.0.0.1:{PORT}'],cwd=HERE)
        if args.tool_smoke:
            target_name='lru_cache_'+stem
            run(['python3','-u',str(HERE/'appserver_tool_smoke.py'),
                 '--catalog',str(HERE/'glm-5.3-flash.catalog.proposed.json'),
                 '--tag',stem+'-tool-smoke','--endpoint',f'http://127.0.0.1:{PORT}',
                 '--target-name',target_name,'--warm','--approval-file',str(HERE/(stem+'-tool-approval.json'))],cwd=HERE)
            run(['python3','-I','-B',str(HERE/'test_lru_contract.py'),
                 str(HERE/'tool-smoke-0919'/(target_name+'.py'))],cwd=HERE)
            report['coding_contract']='PASS'
        if args.stateful_reference:
            run(['python3','-u',str(HERE/'stateful_gate.py'),'--endpoint',f'http://127.0.0.1:{PORT}',
                 '--tag',stem,'--reference',str(args.stateful_reference)],cwd=HERE)
            stateful=json.loads((HERE/('stateful-'+stem+'.json')).read_text())
            report['stateful_regression_gate']=stateful['regression_passed']
            report['stateful_intrinsic_consistency']=stateful['passed']
            report['production_intrinsic_consistency']=stateful['baseline_consistency_passed']
        if args.mtp_sweep:
            sweeps=[]
            for depth in (1,2):
                assert not any(s['is_processing'] for s in request(PORT,'slots')), 'MTP adjustment requires idle slots'
                draft_file.write_text(str(depth)+'\n')
                tag='goal0919-'+stem+'-mtp'+str(depth)
                metrics_before=metrics(f'http://127.0.0.1:{PORT}')
                run(['bash',str(FLEET/'bench3.sh'),str(PORT),tag,'256'])
                rows=[]
                for i in (1,2,3):
                    before=json.loads((FLEET/f'results/bench3-goal0919-baseline-{i}.json').read_text())
                    after=json.loads((FLEET/f'results/bench3-{tag}-{i}.json').read_text())
                    rows.append({'prompt':i,'parity':before['content']==after['content'],'timings':after['timings']})
                log=(HERE/f'{stem}-candidate-server.log').read_text(errors='replace')
                counter_delta=summarize(metrics_before,metrics(f'http://127.0.0.1:{PORT}'))
                actual_depth=counter_delta['draft_tokens_per_verification']
                assert actual_depth is not None and abs(actual_depth-depth)<0.03, 'Requested MTP depth not supported by runtime counters'
                sweeps.append({'depth':depth,'rows':rows,'runtime_counters':counter_delta,
                               'diagnostic_marker_visible':f'MTP_DRAFT_LIMIT active={depth} configured=2' in log})
                report['mtp_sweep']=sweeps
                if depth==2:
                    assert all(row['parity'] for row in rows), 'Repeated MTP2 output changed'
                elif not all(row['parity'] for row in rows):
                    print('MTP1 rejected: fixed-prompt output changed; reverting to MTP2 control',flush=True)
            def score(arm):
                rows=arm['rows']
                return sum(r['timings']['predicted_n']-1 for r in rows)/sum(r['timings']['predicted_ms']/1000 for r in rows)
            report['mtp_depth_scores']={str(a['depth']):score(a) for a in sweeps}
            one,two=sweeps
            if all(r['parity'] for r in one['rows']) and score(one)>score(two)*1.03:
                draft_file.write_text('1\n')
                run(['python3','-u',str(HERE/'appserver_bench.py'),
                     '--catalog',str(HERE/'glm-5.3-flash.catalog.proposed.json'),
                     '--tag',stem+'-mtp1-alias-cold','--endpoint',f'http://127.0.0.1:{PORT}'],cwd=HERE)
                draft_file.write_text('2\n')
                report['mtp1_codex_measured']=True
        if args.profile:
            run(['python3','-u',str(HERE/'profile_once.py'),'--pid',str(current['pid']),'--tag',stem],cwd=HERE)
            report['operation_profile']='captured separately; excluded from speed results'
        report['gate']='Kernel checks completed; known baseline consistency issue retained; not promoted' if report.get('stateful_intrinsic_consistency') is False else 'PASS; not promoted automatically'
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        (HERE/f'{stem}-window-report.json').write_text(json.dumps(report,indent=2)+'\n')
        if stopped:
            print('Stopping candidate and restoring supervised production',flush=True)
            candidate_pid=pid_for(UNIT)
            subprocess.run(['systemctl','--user','stop',UNIT],check=False)
            if candidate_pid: wait_dead(candidate_pid)
            assert not open_port(PORT), 'Candidate port still live; refusing overlapping restore'
            assert not open_port(18131), 'Unexpected production listener appeared'
            run(['systemctl','--user','start',PRODUCTION])
            wait_healthy(18131,PRODUCTION)
            restored=snapshot(pid_for(PRODUCTION),18131)
            (HERE/f'{stem}-window-restored.json').write_text(json.dumps(restored,indent=2)+'\n')
            assert restored['mapped_libraries_sha256']==original['mapped_libraries_sha256'], 'Restore libraries differ'
            assert restored['environment']==original['environment'], 'Restore inference flags differ'
            report['restored_production_pid']=restored['pid']
            report['restored_healthy']=True
            print('Supervised production restored, PID',restored['pid'],flush=True)
        (HERE/f'{stem}-window-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
