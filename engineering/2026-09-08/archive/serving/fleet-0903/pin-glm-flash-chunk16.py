#!/usr/bin/env python3
"""Write a launcher for the verified GLM baseline; its measured rate is below 20."""
import ast
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
evidence = base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json'
result = json.loads(evidence.read_text())
assert result.get('server_exit') == 0 and not result.get('error')
assert all(c['pass'] for c in result['checks'][:3])
assert len(result['cache_checks']) == 4 and all(c['pass'] for c in result['cache_checks'])
identity = json.loads(evidence.with_name('response-identity-check.json').read_text())
assert identity['all_identical']
reference = json.loads((base / 'results/glm5n-goal-iq-batch3-best-threads-1024/result.json').read_text())
assert all(c['response']['choices'][0]['message'] == reference['checks'][i]['response']['choices'][0]['message']
           for i, c in enumerate(result['checks']))
snapshot = json.loads((base / 'results/glm-chunk16-binary-snapshot.json').read_text())
assert result['config']['label'] in snapshot['evidence']
pinned = Path(snapshot['directory'])
for name, digest in snapshot['sha256'].items():
    assert hashlib.sha256((pinned / name).read_bytes()).hexdigest() == digest
draft = json.loads((base / 'results/glm-mtp-q8-persistent.json').read_text())
assert draft['verified'] and Path(draft['destination']).stat().st_size == draft['bytes']
env = {k: v for k, v in result['runtime_env'].items()
       if not k.startswith(('GGML_CPU_OP_PROFILE', 'LLAMA_GRAPH_PHASE')) and not k.endswith('_FILE')}
env['LD_LIBRARY_PATH'] = str(pinned)
command = list(result['command'])
command[0] = str(pinned / 'llama-server')
command[command.index('--spec-draft-model') + 1] = draft['destination']
command[command.index('--verbosity') + 1] = '2'
bench = [c for c in result['checks'] if c['expected'] is None]
assert not any(c['throughput_contended'] for c in bench)
standard = [c for c in bench if c['reasoning_budget_tokens'] is None]
manifest = dict(command=command, runtime_env=env, binary_sha256=snapshot['sha256'],
                evidence=str(evidence), draft_artifact=draft,
                measured_samples=[dict(reasoning_budget_tokens=c['reasoning_budget_tokens'],
                                       timings=c['response']['timings']) for c in bench],
                measured_standard_target_met=all(c['response']['timings']['predicted_per_second'] >= 20 for c in standard),
                note='GLM baseline is validated but below the 20 tok/s goal. Run only one Flash model at a time; host load and output affect rates.')
(base / 'glm-flash-validated.json').write_text(json.dumps(manifest, indent=2) + '\n')
source = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

config = json.loads(Path(__file__).with_name("glm-flash-validated.json").read_text())
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("GGML_", "LLAMA_GRAPH_PHASE", "LLAMA_MTP_DRAFT_N_FILE", "OMP_", "GOMP_"))}
env.update(config["runtime_env"])
os.execvpe(config["command"][0], config["command"] + sys.argv[1:], env)
'''
ast.parse(source)
launcher = base / 'launch-glm-flash-validated.py'
launcher.write_text(source)
launcher.chmod(0o755)
print('Verified baseline launcher:', launcher)
print('Measured standard target met:', manifest['measured_standard_target_met'])
