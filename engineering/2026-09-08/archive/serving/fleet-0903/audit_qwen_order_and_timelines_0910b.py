#!/usr/bin/env python3
"""Reparse counters, retained responses, and socket timelines after restoration."""
from collections import Counter
import fcntl
import json
import os
import re
from pathlib import Path
import time

from analyze_qwen_timeline_0910b import analyze
from benchmark_flash_q4_selected_0910 import validate_counters
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager
from trace_qwen_shared_dispatch_ops_0909 import output_record

BASE=Path(__file__).resolve().parent
DESTINATION=BASE/'results/qwen-order-and-timeline-audit-0910b.json'
CASES=['qwen-whole-server-timeline-0910','qwen-whole-server-timeline-0910b',
    'qwen-repeatability-0910c','qwen-order-control-0910d','qwen-mtp-fresh-seed-0910e','qwen-mtp-fresh-seed-0910f']
COUNT_KEYS=('predicted_n','cache_n','draft_n','draft_n_accepted')


def read(path):
    assert '.private.' not in str(path)
    return json.loads(path.read_text())


def compact_timeline(entry):
    path=Path(entry['trace_file']);assert sha256(path)==entry['trace_sha256']
    parsed=analyze(path.read_text(),entry['timeline_graph_count'])
    waves=parsed['complete_four_socket_waves'];roles=[]
    by_index={g['index']:g for g in parsed['graphs']}
    for role in ('target','draft'):
        selected=[w for w in waves if w['role']==role]
        ops,families=Counter(),Counter()
        matrix_groups={}
        barrier=0
        for wave in selected:
            ops.update(wave['last_finishing_graph_ops_us'])
            graph=max((by_index[i] for i in wave['graph_indices']),key=lambda g:g['end_us'])
            for node in graph['nodes']:
                barrier+=node['end_us']-node['work_end_us']
                if node['op'] in ('MUL_MAT','MUL_MAT_ID'):
                    family=re.sub(r'blk\.\d+\.','blk.#.',node['src0_name'])
                    key=(node['op'],node['src0_type'],family,tuple(node['src0_ne']),tuple(node['dst_ne']))
                    group=matrix_groups.setdefault(key,dict(op=node['op'],weight_type=node['src0_type'],weight_family=family,
                        source_shape=node['src0_ne'],destination_shape=node['dst_ne'],calls=0,stage_us=0,outer_barrier_us=0))
                    group['calls']+=1
                    group['stage_us']+=node['end_us']-node['start_us']
                    group['outer_barrier_us']+=node['end_us']-node['work_end_us']
            families.update(wave['last_finishing_graph_families_us'])
        roles.append(dict(role=role,complete_waves=len(selected),
            summed_wave_envelopes_us=sum(w['elapsed_envelope_us'] for w in selected),
            summed_last_finishing_graph_us=sum(w['last_finishing_graph_us'] for w in selected),
            last_finishing_graph_ops_us=dict(ops.most_common()),
            last_finishing_graph_outer_barrier_us=barrier,
            last_finishing_graph_top_families_us=dict(families.most_common(15)),
            last_finishing_graph_matrix_groups=sorted(matrix_groups.values(),key=lambda g:-g['stage_us'])))
    return dict(kind=entry['kind'],raw_trace=str(path),raw_trace_sha256=sha256(path),
        raw_bytes=path.stat().st_size,raw_trace_retained_on_server=True,
        output_and_counts_match_reference=entry['output_matches_reference'] and entry['counts_match_reference'],
        summaries=parsed['summaries'],roles=roles,complete_four_socket_waves=waves,
        all_graph_interval_union_us=parsed['all_graph_interval_union_us'],limitations=parsed['limitations'])


def audit_case(label):
    directory=BASE/'results'/label;path=directory/'result.json';result=read(path);plan_path=directory/'plan.json';plan=read(plan_path)
    assert result['finished'] and result['stage']=='finished' and result['restored']
    assert result['plan_sha256']==sha256(plan_path)
    assert result['owned_qwen_exit']==0 and result['dispatch_cleanup_valid']
    assert result['full_environment_restored'] and result['restored_command_and_affinity'] and result['restored_libraries_match']
    assert all(sha256(p)==h for p,h in plan['source_sha256'].items())
    if 'measurement' not in result:
        assert not result['passed'] and result.get('error') and not result['checks'] and not result['traces']
        return dict(label=label,result_sha256=sha256(path),plan_sha256=sha256(plan_path),
            collection_passed=False,collection_error=result['error'],measurements=[],repeats=[],repeatability_passed=None,
            controlled_code_to_prose_hashes={},cache_erasure_proven_to_restore_reference=False,timelines=[],
            restoration_reported_exact=True,restored_flash_pid=result['restored_flash_pid'],peer_pid=plan['peer']['pid'],
            no_requests_ran=True)
    if 'common' in plan:
        assert result['mapped_common_libraries']==[str(Path(plan['common']).resolve())]
    measured_path=Path(result['measurement']);assert sha256(measured_path)==result['measurement_sha256']
    data=read(measured_path)
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    assert all(c['pass_check'] and not c['abort'] for c in data['checks'])
    assert data['server_command']==plan['command'] and data['runtime_env']==plan['runtime_env']
    measurements=[]
    for row in data['measurements']:
        assert not row['abort'] and not row['inference_churn'] and not row['other_inference']
        sample_dir=measured_path.parent/(row['kind']+'-draft4')
        counters=validate_counters(sample_dir,row)
        _,timings,digest=output_record(read(sample_dir/'chunks.json'))
        expected=result['reference'][row['kind']]
        assert digest==expected['output_sha256'] and timings==expected['timings']
        measurements.append(dict(kind=row['kind'],tok_s=timings['predicted_per_second'],
            output_sha256=digest,counts={k:timings.get(k,0) for k in COUNT_KEYS},counters=counters,
            current_host_load=True,uncontended_comparison=False))
    repeats=[]
    for entry in result.get('repeat_requests',[]):
        chunks_path=directory/f"repeat-{entry['index']:02d}-{entry['kind']}-chunks.json"
        assert sha256(chunks_path)==entry['chunks_sha256']
        _,timings,digest=output_record(read(chunks_path))
        assert digest==entry['output_sha256'] and timings==entry['timings']
        expected=result['reference'][entry['kind']]
        same_counts=all(timings.get(k,0)==expected['timings'].get(k,0) for k in COUNT_KEYS)
        assert entry['output_matches_reference']==(digest==expected['output_sha256'])
        assert entry['counts_match_reference']==same_counts and not entry['abort'] and timings['cache_n']==0
        repeats.append(dict(index=entry['index'],kind=entry['kind'],preceding_kind=entry.get('preceding_kind'),
            erase_before=entry['erase_before'],profile=entry.get('profile',False),output_sha256=digest,
            output_matches_reference=entry['output_matches_reference'],counts_match_reference=same_counts,
            counts={k:timings.get(k,0) for k in COUNT_KEYS}))
    control=[r for r in repeats if r['kind']=='prose' and r['preceding_kind']=='code' and not r['profile']]
    control_groups={str(erase):[r['output_sha256'] for r in control if r['erase_before']==erase] for erase in (False,True)}
    traces=[compact_timeline(e) for e in result.get('traces',[]) if 'trace_file' in e]
    return dict(label=label,result_sha256=sha256(path),plan_sha256=sha256(plan_path),
        collection_passed=result['passed'],collection_error=result.get('error'),measurements=measurements,repeats=repeats,
        repeatability_passed=result.get('repeatability_passed'),controlled_code_to_prose_hashes=control_groups,
        cache_erasure_proven_to_restore_reference=any(r['erase_before'] for r in control) and all(r['output_matches_reference'] for r in control if r['erase_before'])
            and any(not r['output_matches_reference'] for r in control if not r['erase_before']),
        timelines=traces,restoration_reported_exact=True,restored_flash_pid=result['restored_flash_pid'],
        peer_pid=plan['peer']['pid'])


def main():
    assert os.sched_getaffinity(0)=={127} and not DESTINATION.exists()
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manager=Manager();current=manager.validate_current()
        guard=ModelMeasurementGuard(current['pid'],{current['pid']:18131},inference_snapshot);guard.assert_idle()
        cases=[]
        for label in CASES:
            guard.assert_idle();cases.append(audit_case(label));print(json.dumps(dict(audited=label)),flush=True)
        for before,after in zip(cases,cases[1:]):assert before['restored_flash_pid']==after['peer_pid']
        assert cases[-1]['restored_flash_pid']==current['pid']
        guard.assert_idle();manager.validate_current()
        result=dict(finished=time.time(),passed=True,source_sha256=sha256(__file__),cases=cases,
            selected_flash_preserved=True,selected_flash_pid=current['pid'],target_quant='UD-Q6_K_XL',
            limitations=['Bandwidth is DRAM read plus write, conservatively subtracting the larger adjacent idle baseline.',
                'Experiments record existing host load; their speed differences are not controlled optimization gains.',
                'Historical exact environment restoration is reported by the guarded controller; private context files are not read or exported.',
                'Profile throughput is not a speed result. Four simultaneous socket intervals are counted by their union.',
                'Fresh-sequence repeatability does not establish target-only numerical parity or a general quality benchmark.'])
        atomic_json(DESTINATION,result)
        print(json.dumps(dict(passed=True,cases=len(cases),matched_timeline_captures=sum(t['output_and_counts_match_reference'] for c in cases for t in c['timelines']),selected_flash_pid=current['pid'])))


if __name__=='__main__':
    os.umask(0o077);main()
