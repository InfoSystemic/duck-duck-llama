#!/usr/bin/env python3
"""Apply the reviewed wider kpool kernel, or verify the proposal without applying it."""
import argparse,hashlib,json,signal,subprocess
from pathlib import Path
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port
from thread_window import native,codex,idle
HERE=Path(__file__).resolve().parent
UNIT='glm53-flash-production.service'
TEST_UNIT='glm53-flash-kpool-wide-test-0919.service'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd):subprocess.run(cmd,check=True,cwd=HERE)
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pid',required=True,type=int);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    manifest=json.loads((HERE/'kpool-promotion-manifest.json').read_text())
    proof=json.loads((HERE/'kpool-window-report.json').read_text())
    summary=json.loads((HERE/'kpool-window-summary.json').read_text())
    assert proof.get('completed') and proof.get('restored_healthy') and not proof.get('error')
    assert proof.get('gate') and 'production_codex_after' in proof, 'Final restored-production control is incomplete'
    assert summary.get('completed') and summary.get('restored_healthy')
    assert proof['engagement_verified'] and proof['stateful_regression_passed']
    assert len(proof['profiles'])==2 and summary['long_profile_pool_shapes_match']
    assert all(proof['arms'][f'native-{mode}']['all_parity'] for mode in [0,1])
    for label in ['short','long']:
        row=next(x for x in summary['rows'] if x['context']==label and x['cache']=='warm ABBA')
        assert row['same_generated_text'] and row['same_work_counters']
        assert row['speedup']>1.02, 'No adequate reproducible benefit at '+label
    library=Path(manifest['library']);proposed=Path(manifest['proposed_dropin']);dropin=Path(manifest['dropin'])
    assert digest(library)==manifest['sha256']==proof['candidate_sha256']
    assert digest(proposed)==manifest['dropin_sha256'] and not dropin.exists()
    assert pid_for(UNIT)==a.pid and pid_for(TEST_UNIT)==0
    original=snapshot(a.pid,18131);idle(18131)
    baseline=json.loads((HERE/'kpool-window-restored.json').read_text())
    for key in ['command','environment','mapped_libraries_sha256']:assert original[key]==baseline[key]
    if not a.execute:
        print(json.dumps({'proposal_verified':True,'production_unchanged':True,'library_sha256':manifest['sha256'],'dropin':proposed.read_text()},indent=2));return
    report_path=HERE/'kpool-promotion-report.json'
    assert not report_path.exists()
    report={'original_pid':a.pid,'library_sha256':manifest['sha256'],'promoted':False,'known_cache_consistency_issue_unresolved':True}
    save(HERE/'kpool-promotion-before.json',original)
    stopped=False;installed=False;success=False
    def stop_signal(signum,frame):raise KeyboardInterrupt(f'Signal {signum}; restore prior configuration')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        stopped=True;run(['systemctl','--user','stop',UNIT]);wait_dead(a.pid)
        assert not open_port(18131)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>380*1024*1024
        with dropin.open('xb') as f:f.write(proposed.read_bytes())
        installed=True
        run(['systemctl','--user','daemon-reload']);run(['systemctl','--user','start',UNIT]);wait_healthy(18131,UNIT)
        current=snapshot(pid_for(UNIT),18131)
        assert current['command']==original['command'] and current['mapped_libraries_sha256'].get(str(library))==manifest['sha256']
        old={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
        new={Path(k).name:v for k,v in current['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
        assert old==new
        env=dict(original['environment']);env['LD_LIBRARY_PATH']=str(library.parent)+':'+env['LD_LIBRARY_PATH'].split(':',1)[1];env['GGML_CPU_GLM_POOL_WIDE']='1'
        assert current['environment']==env
        save(HERE/'kpool-promotion-running.json',current)
        row=native(18131,'kpool-promoted');assert row['all_parity'];report['native']=row
        for warm in [False,True]:
            tag='kpool-promoted-'+('warm' if warm else 'cold')
            row=codex(18131,tag,warm)
            ref=proof['arms']['short-warm-1' if warm else 'short-cold-1']
            assert row['output_text']==ref['output_text']
            report[tag]={k:row[k] for k in ['server_metrics','wall_seconds','marker_pass']}
        final=snapshot(current['pid'],18131);idle(18131)
        for key in ['command','environment','mapped_libraries_sha256']:assert final[key]==current[key]
        save(HERE/'kpool-promotion-final.json',final)
        report.update(promoted=True,healthy=True,production_pid=current['pid']);success=True
        print('Wider kpool kernel deployed and verified, PID',current['pid'],flush=True)
    except BaseException as exc:report['error']=repr(exc);raise
    finally:
        save(report_path,report)
        if not success and stopped:
            print('Rolling back wider-kernel deployment',flush=True)
            pid=pid_for(UNIT);subprocess.run(['systemctl','--user','stop',UNIT],check=False)
            if pid:wait_dead(pid)
            assert not open_port(18131)
            if installed:
                assert digest(dropin)==manifest['dropin_sha256'],'Concurrent drop-in change'
                dropin.unlink()
            run(['systemctl','--user','daemon-reload']);run(['systemctl','--user','start',UNIT]);wait_healthy(18131,UNIT)
            restored=snapshot(pid_for(UNIT),18131)
            for key in ['command','environment','mapped_libraries_sha256']:assert restored[key]==original[key]
            report.update(rolled_back=True,restored_healthy=True,restored_pid=restored['pid']);save(report_path,report)
if __name__=='__main__':main()
