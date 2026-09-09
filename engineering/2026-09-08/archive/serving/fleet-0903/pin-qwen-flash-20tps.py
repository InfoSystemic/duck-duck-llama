#!/usr/bin/env python3
"""Preserve the tested runtime and write a reproducible standalone launcher."""
import hashlib
import json
from pathlib import Path
import shutil

base = Path(__file__).resolve().parent
root = base.parents[1]
evidence = base / 'results/q4e-goal-iq-batch3-threads-1024/result.json'
result = json.loads(evidence.read_text())
assert result.get('server_exit') == 0 and not result.get('error')
assert all(check['pass'] for check in result['cache_checks'])
winner = next(case for case in result['thread_sweep'] if case['threads'] == result['selected_numa_threads'])
assert all(check['pass'] for check in winner['checks'][:3])
bench = [check for check in winner['checks'] if check['expected'] is None]
assert len(bench) == 2 and all(check['response']['timings']['predicted_per_second'] >= 20 for check in bench)
assert not any(check['throughput_contended'] for check in bench)
reference = json.loads((base / 'results/q4e-goal-nibble-sort-thread-sweep-1024/result.json').read_text())
assert all(check['response']['choices'][0]['message'] == reference['checks'][i]['response']['choices'][0]['message']
           for i, check in enumerate(winner['checks']))

engine = root / 'engines/llama.cpp-q4e-goal-0904'
source = engine / 'build-goal/bin'
pinned = engine / 'validated-iq-batch3-bin'
if not pinned.exists():
    shutil.copytree(source, pinned, symlinks=True)
hashes = {}
for path in source.iterdir():
    if path.is_file():
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256((pinned / path.name).read_bytes()).hexdigest() == original
        hashes[path.name] = original

env = {key: value for key, value in result['runtime_env'].items()
       if not key.startswith(('GGML_CPU_OP_PROFILE', 'LLAMA_GRAPH_PHASE'))
       and key != 'GGML_CPU_NUMA_THREADS_FILE'}
env['GGML_CPU_NUMA_THREADS'] = str(winner['threads'])
env['LD_LIBRARY_PATH'] = str(pinned)
command = list(result['command'])
command[0] = str(pinned / 'llama-server')
for option in ['--threads', '--threads-batch']:
    command[command.index(option) + 1] = str(winner['threads'])
command[command.index('--verbosity') + 1] = '2'
manifest = {'command': command, 'runtime_env': env, 'binary_sha256': hashes,
            'evidence': str(evidence), 'threads_per_socket': winner['threads'],
            'measured_rates': [check['response']['timings'] for check in bench],
            'note': 'Measured on the SR950 with one Flash test model loaded; host load and output affect rates.'}
(base / 'qwen-flash-20tps.json').write_text(json.dumps(manifest, indent=2) + '\n')
launcher = base / 'launch-qwen-flash-20tps.py'
launcher.write_text('''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
from exclusive_model_launch import acquire_exclusive_model_launch, ModelLaunchConflict

config = json.loads(Path(__file__).with_name("qwen-flash-20tps.json").read_text())
try:
    launch_lease = acquire_exclusive_model_launch(Path(__file__).resolve().parent)
except ModelLaunchConflict as error:
    raise SystemExit(f"Qwen launch refused: {error}")
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("GGML_", "LLAMA_GRAPH_PHASE", "LLAMA_MTP_DRAFT_N_FILE", "OMP_", "GOMP_"))}
env.update(config["runtime_env"])
os.execvpe(config["command"][0], config["command"] + sys.argv[1:], env)
''')
launcher.chmod(0o755)
print('Pinned', len(hashes), 'runtime files; launcher:', launcher)
