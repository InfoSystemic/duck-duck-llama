#!/usr/bin/env python3
"""Audit repeated Flash IMC measurements and warmed-up cycle profiles."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from benchmark_flash_q4_selected_0910 import read_measurement
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import BASE, PORT, Manager, mapped_libraries

OUT = BASE / 'results/flash-q4-quiet-audit-0910.json'


def read(path):
    assert '.private.' not in str(path)
    return json.loads(Path(path).read_text())


def hashes(items):
    for path, digest in items.items():
        assert '.private.' not in path and sha256(path) == digest, path


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager(); current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        benchmark_path = BASE / 'results/flash-q4-quiet-mtp2-0910/result.json'
        profile_dir = BASE / 'results/flash-q4-quiet-mtp2-profile-0910'
        benchmark, profile = read(benchmark_path), read(profile_dir / 'result.json')
        assert benchmark['passed'] and benchmark['finished'] and benchmark['selected_flash_preserved']
        assert benchmark['current'] == profile['current'] == current
        assert profile['completed'] and profile['finished'] and not profile.get('error')
        hashes(benchmark['source_sha256']); hashes(benchmark['mapped_libraries'])
        hashes(profile['source_sha256']); hashes(profile['mapped_libraries'])
        assert mapped_libraries(current['pid']) == set(benchmark['mapped_libraries']) == set(profile['mapped_libraries'])
        rows = []
        assert len(benchmark['runs']) == 2
        earlier = read(BASE / 'results/glm-flash-q4-measured-0910/result.json')
        for run in benchmark['runs']:
            path = Path(run['measurement'])
            assert sha256(path) == run['measurement_sha256']
            parsed = read_measurement(path, current)
            assert parsed == run['rows'] and run['outputs_and_counts_match']
            gate = run['background_gate']
            assert gate['max_cores'] == 6 and gate['samples'][-1]['quiet_seconds'] >= 20
            for kind, row in parsed.items():
                assert all(row[key] == earlier['runs'][0]['rows'][kind][key]
                    for key in ['output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'])
                rows.append(dict(repetition=run['index'], kind=kind, **row))
        assert benchmark['all_outputs_and_counts_match'] and benchmark['all_adjacent_idle_qualifies']
        assert all(row['counters']['adjacent_idle_qualifies'] for row in rows)
        assert not benchmark['target_240_reached'] and not benchmark['target_250_reached']

        assert len(profile['profiles']) == 2 and all(c['passed'] and not c['abort'] for c in profile['checks'])
        assert profile['baseline_sha256'] == sha256(profile['config']['after'])
        pattern = re.compile(r'^(\d+)/(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.+)\s+\((.+)\)$')
        mappings = Path(f"/proc/{current['pid']}/maps").read_text().splitlines()
        gomp_paths = {r.split()[-1] for r in mappings if '/libgomp.so.' in r}
        gomp_path, = gomp_paths
        old_library = read(BASE / 'results/glm-flash-q8-pool-profile-0908/analysis/host-library.json')
        assert gomp_path == old_library['libgomp_path']
        assert sha256(gomp_path) == old_library['libgomp_sha256'] == '135f3c8f006d2fe5e68e51281c7974cb991a03de3bfb3593d68d174dfcf854d1'
        gomp_maps = [(int(r.split()[0].split('-')[0],16), int(r.split()[0].split('-')[1],16), int(r.split()[2],16))
            for r in mappings if r.split()[-1] == gomp_path]
        summaries = []
        socket_map = {cpu:int(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text()) for cpu in range(128)}
        for entry in profile['profiles']:
            assert entry['perf_exit'] == entry['report_exit'] == 0
            assert not entry['abort'] and not entry['other_inference'] and not entry['inference_churn']
            assert entry['output_matches_baseline'] and entry['counts_match_baseline']
            directory = profile_dir / (entry['kind'] + '-draft2')
            assert sha256(directory / 'report.txt') == entry['report_sha256']
            assert sha256(directory / 'samples-by-thread.txt') == entry['samples_sha256']
            periods, counts, symbols, threads, sockets = Counter(), Counter(), Counter(), Counter(), Counter()
            for line in (directory / 'samples-by-thread.txt').read_text().splitlines():
                if not line.strip(): continue
                match = pattern.fullmatch(line.strip()); assert match
                pid, tid, cpu, timestamp, period, ip, symbol, dso = match.groups()
                pid, tid, cpu, period, ip = int(pid), int(tid), int(cpu), int(period), int(ip,16)
                assert pid == current['pid'] and str(tid) in entry['threads_before']
                assert entry['first_content_monotonic'] <= float(timestamp) <= entry['last_content_monotonic']
                category = 'other'
                if dso == gomp_path:
                    offset, = [ip - lo + offset for lo, hi, offset in gomp_maps if lo <= ip < hi]
                    category = 'openmp_spin' if any(lo <= offset < hi for lo, hi in [(0x256b7,0x256d7),(0x2587b,0x258b3)]) else 'openmp_other'
                elif 'ggml_gemv_q4_K_x16' in symbol: category = 'q4_x16_gemv'
                elif 'ggml_gemv_q5_K_x16' in symbol: category = 'q5_x16_gemv'
                elif 'ggml_gemv_q8_0_x16' in symbol: category = 'q8_x16_gemv'
                elif 'ggml_gemv_q8_0_8x8' in symbol: category = 'q8_x8_gemv'
                elif 'flash_q8_sum_candidate' in symbol: category = 'q8_sum_batch'
                elif dso == current['cpu_library']: category = 'other_cpu_backend'
                periods[category] += period; counts[category] += 1; symbols[symbol] += period; threads[tid] += period
                socket = socket_map[cpu]
                sockets[socket] += period
            expected, = re.findall(r'\((\d+) samples\)', (directory / 'record.log').read_text())
            assert counts.total() == int(expected) and counts.total() > 0
            decoded = subprocess.run(['sudo', '-n', 'perf', 'script', '-G', '--show-lost-events', '-F', 'event',
                '-i', str(directory / 'perf.data')], capture_output=True, text=True, timeout=60)
            assert decoded.returncode == 0 and not any('LOST' in line.upper() for line in decoded.stdout.splitlines())
            total = periods.total()
            summaries.append(dict(kind=entry['kind'], sample_count=counts.total(), total_sampled_period=total,
                category_period_percent={k:100*v/total for k,v in periods.items()},
                socket_period_percent={k:100*v/total for k,v in sockets.items()},
                top_symbols=[dict(symbol=k, period_percent=100*v/total) for k,v in symbols.most_common(12)],
                sampled_threads=len(threads), lost_samples=0, outputs_and_counts_match=True,
                retained_sha256={name:(subprocess.check_output(['sudo','-n','sha256sum',str(directory/name)],text=True).split()[0] if name == 'perf.data' else sha256(directory/name)) for name in ['perf.data','record.log','report.txt','samples-by-thread.txt','chunks.json']}))
        manager.validate_current(); guard.assert_idle()
        result = dict(passed=True, finished=time.time(), source_sha256=sha256(__file__), rows=rows,
            profile_summaries=summaries, selected_flash_pid=current['pid'], selected_flash_preserved=True,
            original_q4_outputs_and_counts_preserved=True, target_240_reached=False, target_250_reached=False,
            runtime_promoted=False, input_sha256={str(p):sha256(p) for p in [benchmark_path, profile_dir/'result.json',
                BASE/'results/glm-flash-q8-pool-profile-0908/analysis/host-library.json', Path(gomp_path)]},
            limitations=['The two unprofiled pairs share one selected process; they are not independent reloads.',
                'All four responses reach a 512-token reasoning cap without a completed answer.',
                'Adjacent idle subtraction estimates model-attributable system-wide traffic.',
                'Cycle sampling is separate from the throughput measurements; cycle fractions are not removable wall-time fractions.',
                'Request draft overrides are disabled in this server; both launch and returned counts establish MTP2.'])
        atomic_json(OUT, result)
        print(json.dumps(dict(passed=True, rows=[{k:r[k] for k in ['repetition','kind','tok_s','adjusted_gb_s']} for r in rows],
            profiles=[{k:r[k] for k in ['kind','sample_count','category_period_percent']} for r in summaries],
            selected_flash_pid=current['pid'])))


if __name__ == '__main__':
    os.umask(0o077)
    main()
