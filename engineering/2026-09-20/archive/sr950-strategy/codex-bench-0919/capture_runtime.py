#!/usr/bin/env python3
"""Capture the actual loaded GLM runtime, not a launcher's presumed settings.

Only inference-related environment variables are recorded; arbitrary inherited
credentials and unrelated process arguments are not collected.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def snapshot(pid, port):
    proc = Path('/proc') / str(pid)
    cmdline = proc.joinpath('cmdline').read_bytes().split(b'\0')
    cmdline = [p.decode() for p in cmdline if p]
    if not cmdline or 'llama-server' not in cmdline[0]:
        raise ValueError('Refusing to snapshot a process other than llama-server')
    env = {}
    for entry in proc.joinpath('environ').read_bytes().split(b'\0'):
        if b'=' not in entry:
            continue
        key, value = entry.decode().split('=', 1)
        if key.startswith(('GGML_', 'GLM_', 'NUMA_', 'META_', 'ARGSORT_', 'Q4E_', 'OMP_', 'KMP_')) or key in ('LD_LIBRARY_PATH', 'LLAMA_MTP_DRAFT_N_FILE', 'LLAMA_GRAPH_PHASE_ARM_FILE', 'LLAMA_KV_SEQ_RM_USED_PREFIX', 'LLAMA_KV_SEQ_RM_CONTROL_FILE', 'LLAMA_KV_SEQ_RM_PROBE'):
            env[key] = value
    paths = set()
    for line in proc.joinpath('maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/'):
            path = parts[5]
            if any(name in Path(path).name for name in ('libggml', 'libllama', 'llama-server')):
                paths.add(path)
    libraries = {}
    for path in sorted(paths):
        if path.endswith(' (deleted)'):
            raise ValueError(f'Runtime library was replaced while mapped: {path}')
        with open(path, 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        libraries[path] = digest
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/slots', timeout=10) as response:
        slots = json.load(response)
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=10) as response:
        health = json.load(response)
    return {
        'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pid': pid, 'proc_start_ticks': proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
        'command': cmdline, 'environment': dict(sorted(env.items())),
        'mapped_libraries_sha256': libraries,
        'cgroup': proc.joinpath('cgroup').read_text().strip(),
        'cpu_affinity': sorted(os.sched_getaffinity(pid)),
        'load_average': os.getloadavg(),
        'memory': Path('/proc/meminfo').read_text(),
        'health': health,
        'slots': [{k: s.get(k) for k in ('id', 'is_processing', 'n_ctx', 'speculative', 'n_prompt_tokens', 'n_prompt_tokens_processed', 'n_prompt_tokens_cache')} for s in slots],
        'production_unit': subprocess.check_output([
            'systemctl', '--user', 'show', 'glm53-flash-production.service',
            '-p', 'MainPID', '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState'], text=True).strip(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid', type=int)
    parser.add_argument('--port', type=int, default=18131)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.pid, args.port), indent=2))
