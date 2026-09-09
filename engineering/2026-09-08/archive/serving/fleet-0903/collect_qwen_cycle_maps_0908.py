#!/usr/bin/env python3
"""Collect mapping and build-ID evidence from a completed private Qwen profile."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-private-[A-Za-z0-9_-]+-profile',args.label)
    root = BASE / 'results' / args.label
    result = json.loads((root / 'result.json').read_text())
    assert result['completed'] and not result.get('error')
    assert all(sha(Path(path)) == digest for path,digest in result['source_sha256'].items())
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    library = Path('/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0')
    saved = out / library.name
    shutil.copyfile(library,saved)
    records = []
    for entry in result['profiles']:
        kind = entry['kind']
        assert entry['perf_exit'] == 0 and not entry['abort']
        data = root / (kind + '-draft4') / 'perf.data'
        command = ['sudo','-n','perf','script','--show-mmap-events','-G','-F','pid,tid,time,event','-i',str(data)]
        with (out / (kind + '-mmap-errors.txt')).open('w') as error:
            process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=error,text=True)
            mappings = []
            lines = 0
            for line in process.stdout:
                lines += 1
                if 'PERF_RECORD_MMAP' in line:
                    mappings.append(line.rstrip())
            assert process.wait() == 0
        (out / (kind + '-mmaps.txt')).write_text('\n'.join(mappings) + '\n')
        builds = subprocess.run(['sudo','-n','perf','buildid-list','-i',str(data)],capture_output=True,text=True,check=True)
        (out / (kind + '-buildids.txt')).write_text(builds.stdout)
        records.append(dict(workload=kind, mappings=len(mappings), output_lines=lines,
                            gomp_build_ids=[line for line in builds.stdout.splitlines() if line.endswith(str(library))]))
    notes = subprocess.run(['readelf','-n',str(saved)],capture_output=True,text=True,check=True)
    (out / 'libgomp-notes.txt').write_text(notes.stdout)
    buildid = re.search(r'Build ID: (\w+)',notes.stdout).group(1)
    assert all(len(row['gomp_build_ids']) == 1 and row['gomp_build_ids'][0].split()[0] == buildid for row in records)
    assembly = subprocess.run(['objdump','-d','--start-address=0x25630','--stop-address=0x25730',str(saved)],capture_output=True,text=True,check=True)
    (out / 'libgomp-wait-disassembly.txt').write_text(assembly.stdout)
    summary = dict(time=time.time(),passed=True,records=records,source=str(library),sha256=sha(saved),
                   build_id=buildid,profile_sha256=sha(root / 'result.json'),collector_sha256=sha(Path(__file__).resolve()))
    (out / 'libgomp-origin.json').write_text(json.dumps(summary,indent=2) + '\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
