#!/usr/bin/env python3
"""Classify ordered Q8 projection execution using verified instruction offsets."""
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
from glm_flash_q8_trial import BASE,Manager,PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot,sha256

ROOT=BASE/'results/glm-flash-q8-r8-ordered-k-profile-0908'

def main():
    manager=Manager();current=manager.validate_current()
    guard=ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot)
    guard.assert_idle()
    profile_path=ROOT/'result.json';profile=json.loads(profile_path.read_text())
    assert profile['completed'] and profile['current']==current
    entry,=profile['profiles']
    assert entry['perf_exit']==entry['report_exit']==0
    assert not entry['abort'] and not entry['other_inference'] and not entry['inference_churn']
    cpu=Path(current['cpu_library'])
    assert sha256(cpu)==current['cpu_sha256'] and current['rms_guard'] and current['ordered_k']
    symbol_text=subprocess.run(['nm','-S','--defined-only',str(cpu)],capture_output=True,text=True,check=True).stdout
    helper_ranges={}
    for category,stem in [('rms_guard','flash_rms_try_guarded'),('q8_sum16','flash_q8_sum_candidate')]:
        matches=[line.split() for line in symbol_text.splitlines()
                 if stem in line and len(line.split())==4 and line.split()[2] in ('t','T')]
        symbol,=matches
        start,size=int(symbol[0],16),int(symbol[1],16)
        helper_ranges[category]=(start,start+size)
    segments=subprocess.run(['readelf','--wide','--segments',str(cpu)],capture_output=True,text=True,check=True).stdout
    loads=[line.split() for line in segments.splitlines() if line.strip().startswith('LOAD ')]
    assert loads and all(int(row[1],16)==int(row[2],16) for row in loads)

    previous_path=BASE/'results/glm-flash-q8-pool-profile-0908/analysis/host-library.json'
    previous=json.loads(previous_path.read_text())
    gomp=Path(previous['libgomp_path'])
    assert sha256(gomp)==previous['libgomp_sha256']=='135f3c8f006d2fe5e68e51281c7974cb991a03de3bfb3593d68d174dfcf854d1'
    mapping_lines=Path(f"/proc/{current['pid']}/maps").read_text().splitlines()
    mappings={}
    for path in (cpu,gomp):
        relevant=[line for line in mapping_lines if line.split()[-1]==str(path)]
        assert relevant
        mappings[str(path)]=[(int(line.split()[0].split('-')[0],16),int(line.split()[0].split('-')[1],16),int(line.split()[2],16)) for line in relevant]
    ordered_path=BASE/'results/glm-flash-q8-r8-ordered-k-0908/projection-disassembly.json'
    ordered_all=json.loads(ordered_path.read_text());ordered=ordered_all['candidate']
    assert ordered['library']==str(cpu) and ordered['sha256']==current['cpu_sha256']
    assert not ordered_all['parent']['vector_operations']
    for row in ordered_all.values():
        assert sha256(row['library'])==row['sha256'] and sha256(row['disassembly'])==row['disassembly_sha256']
    unique_offsets={row['offset'] for row in ordered['vector_operations']}
    samples=ROOT/'prose-draft0/samples-by-thread.txt'
    assert sha256(samples)==entry['samples_sha256']
    def offset(path,ip):
        found=[ip-lo+off for lo,hi,off in mappings[path] if lo<=ip<hi]
        assert len(found)==1
        return found[0]
    pattern=re.compile(r'^(\d+)/(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.+)\s+\((.+)\)$')
    totals=Counter();counts=Counter();symbols=Counter();fast_offsets=Counter();rms_offsets=Counter();projection_offsets=Counter();ordered_offsets=Counter()
    for line in samples.read_text().splitlines():
        if not line.strip():continue
        match=pattern.match(line.strip());assert match,line
        pid,tid,cpu_id,timestamp,period,ip,symbol,dso=match.groups()
        assert int(pid)==current['pid']
        period,ip=int(period),int(ip,16)
        category='other'
        if dso==str(gomp):
            address=offset(dso,ip)
            category='gomp_spin' if any(lo<=address<hi for lo,hi in ((0x256b7,0x256d7),(0x2587b,0x258b3))) else 'gomp_other'
        elif dso==str(cpu):
            if symbol==ordered['symbol']:
                address=offset(dso,ip);assert ordered['start']<=address<ordered['end']
                category='q8_r8_projection_dispatch';projection_offsets[hex(address)]+=1
                if address in unique_offsets:
                    category='q8_r8_ordered_vector';ordered_offsets[hex(address)]+=1
            elif 'flash_rms_try_guarded' in symbol:
                address=offset(dso,ip)
                lo,hi=helper_ranges['rms_guard'];assert lo<=address<hi
                category='rms_guard';rms_offsets[hex(address)]+=1
            elif 'flash_q8_sum_candidate' in symbol:
                address=offset(dso,ip)
                lo,hi=helper_ranges['q8_sum16'];assert lo<=address<hi
                category='q8_sum16';fast_offsets[hex(address)]+=1
            elif symbol=='ggml_compute_forward_rms_norm':category='rms_norm'
            elif symbol=='ggml_gemv_q8_0_x16_q8_0':category='q8_x16'
            elif symbol=='ggml_gemv_q8_0_8x8_q8_0':category='q8_x8'
            elif symbol=='ggml_vec_dot_f32':category='f32_dot'
        totals[category]+=period;counts[category]+=1;symbols[symbol]+=period
    record=ROOT/'prose-draft0/record.log'
    expected,=re.findall(r'\((\d+) samples\)',record.read_text())
    assert counts.total()==int(expected) and counts.total()>0
    assert counts['q8_sum16']>0 and counts['rms_guard']>0 and counts['q8_r8_ordered_vector']>0
    command=['sudo','-n','perf','script','-G','--show-lost-events','-F','event','-i',str(ROOT/'prose-draft0/perf.data')]
    decoded=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert decoded.returncode==0,decoded.stderr
    lost=[line for line in decoded.stdout.splitlines() if 'LOST' in line.upper()]
    assert not lost,lost[:5]
    analysis=ROOT/'analysis';analysis.mkdir(exist_ok=False)
    (analysis/'loss-check.stderr').write_text(decoded.stderr)
    baseline_path=BASE/'results/glm-flash-rms-guard-profile-0908/analysis/result.json'
    baseline=json.loads(baseline_path.read_text())
    result=dict(time=time.time(),passed=True,pid=current['pid'],cpu_sha256=current['cpu_sha256'],
        sample_count=counts.total(),sample_counts=dict(counts),period_percent={k:100*v/totals.total() for k,v in totals.items()},
        total_sampled_period=totals.total(),fast_sum_offsets=dict(fast_offsets),rms_offsets=dict(rms_offsets),helper_ranges=helper_ranges,
        mapped_libraries={k:[list(row) for row in rows] for k,rows in mappings.items()},
        gomp_sha256=sha256(gomp),lost_records=0,loss_check_command=command,
        earlier_rms_period_percent=baseline['period_percent'],projection_offsets=dict(projection_offsets),ordered_vector_offsets=dict(ordered_offsets),
        top_symbols=[dict(symbol=k,percent=100*v/totals.total()) for k,v in symbols.most_common(15)],
        source_sha256={str(p):sha256(p) for p in (Path(__file__),profile_path,samples,record,previous_path,baseline_path,cpu,gomp,ordered_path,Path(ordered['disassembly']),Path(ordered_all['parent']['disassembly']))},
        scope='Verified helper execution and sampled user-cycle distribution. Sample fractions are not removable wall-time fractions; profiled speed is not the unprofiled benchmark.')
    manager.validate_current();guard.assert_idle()
    (analysis/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','sample_count','sample_counts','period_percent','lost_records')}))

if __name__=='__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
