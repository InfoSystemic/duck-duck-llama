#!/usr/bin/env python3
"""Measure selected Q4 under observed host load; qualify counters separately."""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from benchmark_qwen_q6 import host_cpu
from dram_bandwidth import parse_records, summarize_samples
from flash_hugepages250_trial_0910 import output_record
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import BASE, PORT, SELECTED, Manager

OUT = BASE / 'results/glm-flash-q4-measured-0910'


def background(pid):
    before, started = host_cpu(), time.monotonic()
    time.sleep(5)
    after = host_cpu()
    elapsed = time.monotonic()-started
    rows = []
    for other,(ticks,birth,name) in after.items():
        old = before.get(other)
        if other == pid or old is None or old[1] != birth:
            continue
        cores = (ticks-old[0])/os.sysconf('SC_CLK_TCK')/elapsed
        if cores > 0:
            rows.append(dict(pid=other,name=name,cores=cores))
    return dict(time=time.time(),background_cores=sum(r['cores'] for r in rows),largest=sorted(rows,key=lambda r:-r['cores'])[:5])


def validate_counters(directory,row):
    saved = json.loads((directory/'samples.json').read_text())
    metadata = saved['metadata']
    assert metadata == row['counter_metadata'] and metadata['valid'] and metadata['exit_code'] == 0
    assert metadata['required_counters_per_interval'] == 48
    assert metadata['cpus'] == [0,16,32,48] and metadata['pmus'] == ['uncore_imc_'+str(i) for i in range(6)]
    assert metadata['socket_ids'] == {'0':0,'16':1,'32':2,'48':3}
    records = []
    for line in (directory/'perf.csv').read_text().splitlines():
        fields = next(csv.reader([line]))
        if len(fields) >= 7 and fields[1].startswith('CPU'):
            records.append((metadata['anchor_monotonic']+float(fields[0]),line))
    parsed,capture = parse_records(records,metadata['cpus'],metadata['pmus'],{int(k):v for k,v in metadata['socket_ids'].items()})
    assert capture['valid'] and len(parsed) == len(saved['samples'])
    for actual,wanted in zip(parsed,saved['samples']):
        assert actual['valid'] and wanted['valid'] and actual['sockets'] == wanted['sockets']
        assert all(math.isclose(actual[k],wanted[k],rel_tol=1e-12,abs_tol=1e-7) for k in ('start','end','duration','total_gb_s'))
    decode = summarize_samples(saved['samples'],row['first_content_monotonic']+.5,row['last_content_monotonic']-.5)
    assert decode == row['decode'] and decode['sampled_seconds'] >= 4
    before,after = row['baseline_before']['total_gb_s'],row['baseline_after']['total_gb_s']
    adjusted = max(0,decode['total_gb_s']-max(before,after))
    assert adjusted == row['background_subtracted_gb_s']
    return dict(counters_per_interval=48,intervals=len(parsed),decode_seconds=decode['sampled_seconds'],
        baseline_before_gb_s=before,baseline_after_gb_s=after,adjusted_gb_s=adjusted,
        adjacent_idle_qualifies=max(before,after)<=19 and abs(before-after)<=9.5)


def read_measurement(path,current):
    data = json.loads(path.read_text())
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    assert data['server_command'] == current['command'] and data['runtime_env'] == current['runtime_env']
    assert len(data['checks']) == 2 and all(c['pass_check'] and not c['abort'] for c in data['checks'])
    assert all(sha256(p)==h for p,h in data['input_sha256'].items())
    rows = {}
    for row in data['measurements']:
        assert not row['abort'] and not row['inference_churn'] and not row['other_inference']
        assert all(row[k]['valid'] for k in ('baseline_before','baseline_after','decode'))
        assert row['draft_n']==2 and row['timings']['cache_n']==0 and row['timings']['draft_n']>0
        assert 128 <= row['timings']['predicted_n'] <= 512
        directory = path.parent/f"{row['kind']}-draft2"
        counters = validate_counters(directory,row)
        chunks = directory/'chunks.json'
        rows[row['kind']] = dict(tok_s=row['timings']['predicted_per_second'],
            adjusted_gb_s=row['background_subtracted_gb_s'],counters=counters,
            generated_tokens=row['timings']['predicted_n'],draft_tokens=row['timings']['draft_n'],
            accepted_draft_tokens=row['timings']['draft_n_accepted'],completed_answer=row['completed_answer'],
            output_sha256=output_record(json.loads(chunks.read_text())),chunks_sha256=sha256(chunks),
            qualified_over_250=counters['adjacent_idle_qualifies'] and row['background_subtracted_gb_s']>=250)
    assert set(rows)=={'prose','code'}
    return rows


def main(lock_fd):
    assert not OUT.exists()
    audit = BASE/'results/glm-flash-q4-activation-audit-0910c.json'
    activated = json.loads(audit.read_text())
    assert activated['passed']
    manager = Manager()
    current = manager.validate_current()
    assert current['pid']==activated['selected_pid'] and current['drafts']==2 and current['quant']=='UD-Q4_K_XL'
    assert set(inference_snapshot())=={str(current['pid'])}
    OUT.mkdir()
    result = dict(started=time.time(),passed=False,controller_pid=os.getpid(),runs=[],
        current=current,selected_config_sha256=sha256(SELECTED),activation_audit_sha256=sha256(audit),
        source_sha256=sha256(__file__),benchmark_policy='Measure with current host applications present; record CPU load and qualify adjacent IMC idle separately. Do not claim an uncontended maximum or isolated Q4/Q8 speedup.')
    cancelled = False
    def cancel(*_):
        nonlocal cancelled
        cancelled = True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,cancel)
    def save():
        atomic_json(OUT/'result.json',result)
    try:
        for index in range(2):
            assert not cancelled
            current = manager.validate_current()
            guard = ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot)
            guard.wait_idle(OUT/'waiting-for-idle.json',quiet_seconds=15)
            guard.assert_idle()
            observed = background(current['pid'])
            label=f'glm-flash-q4-current-load-0910-{index:02d}'
            command=[sys.executable,'-u',str(BASE/'measure-model-bandwidth.py'),label,'--port',str(PORT),
                '--pid',str(current['pid']),'--alias','glm-flash-q4','--drafts','2','--tokens','512',
                '--request-timeout-seconds','300','--allowed-idle-pids','','--skip-idle-gate',
                '--bandwidth-target-gb-s','250','--bandwidth-capacity-gb-s','380',
                '--chat-template-kwargs','{"reasoning_effort":"max"}','--check-reasoning-budget-tokens','0']
            run=dict(index=index,started=time.time(),background_before=observed,command=command)
            result['runs'].append(run)
            save()
            print(json.dumps(dict(measuring=label,background_cores=observed['background_cores'])),flush=True)
            child=subprocess.Popen(command,pass_fds=(lock_fd,))
            try:
                while child.poll() is None:
                    if cancelled:
                        raise InterruptedError('Owned measurement cancelled; selected model remains running')
                    time.sleep(.5)
                assert child.returncode==0
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait(timeout=15)
            path=BASE/'results'/label/'result.json'
            rows=read_measurement(path,current)
            if index:
                assert all(rows[k][f]==result['runs'][0]['rows'][k][f] for k in rows
                    for f in ('output_sha256','generated_tokens','draft_tokens','accepted_draft_tokens'))
            run.update(finished=time.time(),rows=rows,measurement=str(path),measurement_sha256=sha256(path),background_after=background(current['pid']))
            save()
            print(json.dumps(dict(run=index,rows=rows)),flush=True)
        assert sha256(SELECTED)==result['selected_config_sha256']
        manager.validate_current()
        result.update(passed=True,summaries={kind:dict(
            mean_tok_s=statistics.mean(run['rows'][kind]['tok_s'] for run in result['runs']),
            tok_s_range=[min(run['rows'][kind]['tok_s'] for run in result['runs']),max(run['rows'][kind]['tok_s'] for run in result['runs'])],
            adjusted_gb_s_range=[min(run['rows'][kind]['adjusted_gb_s'] for run in result['runs']),max(run['rows'][kind]['adjusted_gb_s'] for run in result['runs'])],
            idle_qualifies_all=all(run['rows'][kind]['counters']['adjacent_idle_qualifies'] for run in result['runs'])) for kind in ('prose','code')},
            target_reached=all(row['qualified_over_250'] for run in result['runs'] for row in run['rows'].values()),
            all_model_goal_complete=False)
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        result['finished']=time.time()
        save()


if __name__=='__main__':
    os.umask(0o077)
    assert os.sched_getaffinity(0)=={127}
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(lock.fileno())
