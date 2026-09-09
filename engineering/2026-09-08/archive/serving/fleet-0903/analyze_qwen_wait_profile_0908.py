#!/usr/bin/env python3
"""Attribute completed model-cycle samples using recorded mappings and thread data."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

BASE = Path(__file__).resolve().parent
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('label')
options = parser.parse_args()
assert re.fullmatch(r'qwen-private-[A-Za-z0-9_-]+-profile', options.label)
ROOT = BASE / 'results' / options.label
OUT = ROOT / 'analysis'
SAMPLE = re.compile(r'\s*(\d+)/(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.*?)\s+\(([^()]*)\)$')
MAPPING = re.compile(r'PERF_RECORD_MMAP2 \d+/\d+: \[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0) .*?\]: (....) (.*)$')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentages(counts, total):
    return {key:100 * value / total for key,value in counts.most_common()}


def main():
    source = ROOT / 'result.json'
    result = json.loads(source.read_text())
    assert result['completed'] and not result.get('error')
    assert all(sha(Path(path)) == digest for path,digest in result['source_sha256'].items())
    cpu = Path(result['current']['runtime_env']['LD_LIBRARY_PATH']) / 'libggml-cpu.so.0'
    cpu = cpu.resolve()
    assert sha(cpu) == '21519acabf56bd088113a7495feba357497af35a346aebe408f78f174d80f4fd'
    disassembly = subprocess.run(['objdump','-d','--disassemble=ggml_barrier',str(cpu)],capture_output=True,text=True,check=True).stdout
    assert '25a08:' in disassembly and 'pause' in disassembly
    (OUT / 'ggml-barrier-disassembly.txt').write_text(disassembly)
    gomp_notes = (OUT / 'libgomp-notes.txt').read_text()
    buildid = re.search(r'Build ID: (\w+)', gomp_notes).group(1)
    gomp_asm = (OUT / 'libgomp-wait-disassembly.txt').read_text()
    assert '256c0:' in gomp_asm and 'pause' in gomp_asm and '256ec:' in gomp_asm and 'syscall' in gomp_asm
    summaries = []
    evidence = [source, cpu, OUT / 'libgomp.so.1.0.0', OUT / 'libgomp-notes.txt',
                OUT / 'libgomp-wait-disassembly.txt', OUT / 'ggml-barrier-disassembly.txt', Path(__file__).resolve()]
    for profile in result['profiles']:
        kind = profile['kind']
        assert not profile['abort'] and profile['perf_exit'] == 0
        directory = ROOT / (kind + '-draft4')
        report = directory / 'report.txt'
        assert 'Total Lost Samples: 0' in report.read_text()
        buildids = OUT / (kind + '-buildids.txt')
        assert any(line.split()[0] == buildid and line.endswith('/libgomp.so.1.0.0') for line in buildids.read_text().splitlines())
        maps_path = OUT / (kind + '-mmaps.txt')
        mappings = []
        for line in maps_path.read_text().splitlines():
            found = MAPPING.search(line)
            assert found, line
            start, size, offset, permissions, path = found.groups()
            if permissions == 'r-xp':
                mappings.append((int(start,16),int(start,16)+int(size,16),int(offset,16),path))
        before, after = profile['threads_before'], profile['threads_after']
        pairs = defaultdict(list)
        for tid, row in before.items():
            if tid in after and row['start'] == after[tid]['start'] and len(row['affinity']) == 1:
                ticks = after[tid]['ticks'] - row['ticks']
                if ticks > 0:
                    pairs[row['affinity'][0]].append((ticks, tid))
        assert len(pairs) == 60 and all(len(rows) == 2 for rows in pairs.values())
        cohorts = {}
        for worker_cpu, rows in pairs.items():
            for name, (_,tid) in zip(('higher-activity','lower-activity'), sorted(rows,reverse=True)):
                cohorts[tid] = name
        periods, counts, by_thread = Counter(), Counter(), defaultdict(Counter)
        symbols = defaultdict(Counter)
        samples = directory / 'samples-by-thread.txt'
        n_samples = 0
        for line in samples.read_text().splitlines():
            if not line.strip():
                continue
            found = SAMPLE.fullmatch(line)
            assert found, line
            pid, tid, worker_cpu, timestamp, period, address, symbol, dso = found.groups()
            assert int(pid) == result['current']['pid']
            assert profile['sample_window_monotonic'][0] <= float(timestamp) <= profile['sample_window_monotonic'][1]
            address, period = int(address,16), int(period)
            mapping = [row for row in mappings if row[0] <= address < row[1] and row[3] == dso]
            category = 'other'
            if dso.endswith(('/libggml-cpu.so.0.22.0','/libgomp.so.1.0.0')):
                assert len(mapping) == 1, (address,dso,mapping)
                start, _, offset, _ = mapping[0]
                file_address = address - start + offset
                if dso.endswith('/libggml-cpu.so.0.22.0') and symbol == 'ggml_barrier':
                    category = 'dissemination-wait' if 0x25a08 <= file_address < 0x25a2f else 'barrier-other'
                elif dso.endswith('/libgomp.so.1.0.0'):
                    category = 'gomp-wait-verified' if 0x256c0 <= file_address < 0x256d1 else 'gomp-other'
                elif symbol == 'ggml_gemv_q6_K_x16_q8_K': category = 'q6-x16'
                elif symbol == 'ggml_gemv_q8_0_x16_q8_0': category = 'q8-x16'
                elif 'ggml_gemv_q8_0_8x8_q8_0' in symbol or 'ggml_gemm_q8_0_8x8_q8_0' in symbol: category = 'q8-8x8'
            if 'ggml_backend_meta_fused_reduce_op' in symbol:
                category = 'cross-socket-reduction'
            cohort = cohorts.get(tid, 'other-threads')
            periods[cohort] += period
            counts[category] += period
            by_thread[tid][category] += period
            symbols[cohort][symbol] += period
            n_samples += 1
        total = sum(periods.values())
        cohort_rows = []
        for cohort in periods:
            combined = Counter()
            tids = [tid for tid in by_thread if cohorts.get(tid,'other-threads') == cohort]
            for tid in tids:
                combined.update(by_thread[tid])
            cohort_rows.append(dict(cohort=cohort, sampled_threads=len(tids), fraction_of_all_periods=periods[cohort]/total,
                                   within_cohort_percent=percentages(combined,periods[cohort]),
                                   top_symbols=[dict(symbol=symbol,percent=100*value/periods[cohort]) for symbol,value in symbols[cohort].most_common(10)]))
        summaries.append(dict(workload=kind, samples=n_samples, total_periods=total,
                              category_percent=percentages(counts,total),cohorts=cohort_rows))
        evidence += [report, buildids, maps_path, samples]
    output = OUT / 'cycle-attribution.json'
    assert not output.exists()
    analysis = dict(time=time.time(),passed=True,workloads=summaries,
                    input_sha256={str(path):sha(path) for path in evidence},
                    scope='Period-weighted sampled CPU cycles. These include concurrent worker waits and are not removable wall-time fractions. Cohorts are classified by observed thread CPU ticks, not assumed target/draft roles.')
    output.write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps({key:value for key,value in analysis.items() if key!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
