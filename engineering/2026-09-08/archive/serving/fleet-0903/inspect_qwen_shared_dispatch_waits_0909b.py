#!/usr/bin/env python3
"""Resolve sampled OpenMP instruction addresses against captured mappings."""
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time

from analyze_qwen_shared_dispatch_profiles_0909 import SAMPLE, analyze
from compare_qwen_private_shared_dispatch_0909 import read_run
from qwen_split_trial import inference_snapshot

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-shared-dispatch-wait-addresses-0909b'
MAPPING = re.compile(r'PERF_RECORD_MMAP2 (\d+)/(\d+): \[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0) .*?\]: (....) (.*)$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perf_sha(path):
    output = subprocess.check_output(['sudo', '-n', 'sha256sum', '--', str(path)], text=True).rstrip('\n')
    digest, name = output.split('  ', 1)
    assert name == str(path) and re.fullmatch(r'[0-9a-f]{64}', digest)
    return digest


def main():
    assert not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert set(inference_snapshot()) == {'1219506'}
        OUT.mkdir()
        evidence = {str(Path(__file__).resolve()): sha(__file__)}
        rows = []
        libraries = {}
        for arm in ['off', 'on']:
            run = read_run('qwen-private-shared-dispatch-' + arm + '-0909c')
            _, inputs = analyze(run)
            evidence.update(inputs)
            root = BASE / 'results' / (run['label'] + '-profile')
            profile = json.loads((root / 'result.json').read_text())
            for item in profile['profiles']:
                kind = item['kind']
                data = root / (kind + '-draft4') / 'perf.data'
                evidence[str(data)] = perf_sha(data)
                mappings = set()
                errors = OUT / (arm + '-' + kind + '-perf-errors.txt')
                saved_maps = OUT / (arm + '-' + kind + '-mappings.txt')
                with errors.open('w') as stderr, saved_maps.open('w') as maps:
                    process = subprocess.Popen(['sudo', '-n', 'perf', 'script', '-D', '-i', str(data)],
                        stdout=subprocess.PIPE, stderr=stderr, text=True)
                    for line in process.stdout:
                        match = MAPPING.search(line.strip())
                        if not match:
                            continue
                        pid, tid, start, size, offset, permissions, path = match.groups()
                        if int(pid) == profile['current']['pid'] and permissions == 'r-xp' and path.endswith('/libgomp.so.1.0.0'):
                            mappings.add((int(start, 16), int(start, 16) + int(size, 16), int(offset, 16), path))
                            maps.write(line)
                    assert process.wait(timeout=60) == 0
                paths = {row[3] for row in mappings}
                assert len(paths) == 1
                library = Path(next(iter(paths)))
                if str(library) not in libraries:
                    raw = library.read_bytes()
                    assert raw[:6] == b'\x7fELF\x02\x01'
                    phoff = struct.unpack_from('<Q', raw, 32)[0]
                    phsize, phnum = struct.unpack_from('<HH', raw, 54)
                    segments = []
                    for i in range(phnum):
                        ptype, flags, offset, vaddr, _, filesz, memsz, align = struct.unpack_from('<IIQQQQQQ', raw, phoff + i * phsize)
                        if ptype == 1 and flags & 1:
                            segments.append((offset, offset + filesz, vaddr))
                    notes = subprocess.check_output(['readelf', '-n', str(library)], text=True)
                    buildid = re.search(r'Build ID: (\w+)', notes).group(1)
                    libraries[str(library)] = dict(sha256=sha(library), build_id=buildid, executable_segments=segments)
                    (OUT / 'libgomp-notes.txt').write_text(notes)
                    evidence[str(library)] = sha(library)
                record = libraries[str(library)]
                assert sha(library) == record['sha256']
                buildids = subprocess.check_output(['sudo', '-n', 'perf', 'buildid-list', '-i', str(data)], text=True)
                assert any(line.split()[0] == record['build_id'] and line.endswith(str(library)) for line in buildids.splitlines())
                (OUT / (arm + '-' + kind + '-buildids.txt')).write_text(buildids)
                counts = Counter()
                all_periods = 0
                for line in (root / (kind + '-draft4') / 'samples-by-thread.txt').read_text().splitlines():
                    if not line.strip():
                        continue
                    match = SAMPLE.fullmatch(line)
                    assert match
                    pid, tid, cpu, timestamp, period, address, symbol, dso = match.groups()
                    all_periods += int(period)
                    if dso != str(library):
                        continue
                    address = int(address, 16)
                    mapping = [row for row in mappings if row[0] <= address < row[1] and row[3] == dso]
                    assert len(mapping) == 1
                    start, _, offset, _ = mapping[0]
                    file_offset = address - start + offset
                    segment = [r for r in record['executable_segments'] if r[0] <= file_offset < r[1]]
                    assert len(segment) == 1
                    offset, _, vaddr = segment[0]
                    counts[file_offset - offset + vaddr] += int(period)
                assert counts and perf_sha(data) == evidence[str(data)]
                top = [dict(address=hex(address), sampled_periods=period, percent_of_all_cycles=100 * period / all_periods)
                       for address, period in counts.most_common(30)]
                rows.append(dict(arm=arm, workload=kind, library=str(library), all_sampled_periods=all_periods,
                                 openmp_periods=sum(counts.values()), top_addresses=top,
                                 all_openmp_addresses={hex(address): count for address, count in counts.items()}))
                print(json.dumps(dict(arm=arm, workload=kind, top_addresses=top[:6])), flush=True)
        for library, record in libraries.items():
            assembly = subprocess.check_output(['objdump', '-d', '--start-address=0x25670', '--stop-address=0x25930', library], text=True)
            (OUT / 'libgomp-wait-region.txt').write_text(assembly)
        for path in OUT.iterdir():
            if path.is_file():
                evidence[str(path)] = sha(path)
        result = dict(time=time.time(), passed=True, libraries=libraries, rows=rows, input_sha256=evidence,
            scope='File offsets are translated through the sampled executable mapping and ELF load segment. Captured and on-disk build IDs match. OpenMP addresses require inspection before classifying waits; sampled concurrent cycles are not removable wall time.')
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
