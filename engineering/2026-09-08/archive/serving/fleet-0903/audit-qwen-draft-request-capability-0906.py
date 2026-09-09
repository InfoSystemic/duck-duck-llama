#!/usr/bin/env python3
"""Audit the rejected zero-draft capture and the fork's request capability."""
import importlib.util
import json
import math
from pathlib import Path
import time

from dram_bandwidth import summarize_samples
from model_measurement_guard import read_service
from qwen_split_trial import (identity_matches, inference_snapshot, process_environment,
                              process_info, runtime_environment, sha256)


def main():
    base = Path(__file__).resolve().parent
    root = base / 'results/qwen-existing-draft01-bandwidth-0906'
    destination = root / 'failure-audit.json'
    assert not destination.exists()
    d = json.loads((root / 'result.json').read_text())
    assert d['finished'] and d['error'] == "AssertionError('Requested raw decode was not raw')"
    assert d['idle_gate']['quiet_seconds'] >= 60
    assert len(d['checks']) == 2 and all(c['pass_check'] and not c['abort'] for c in d['checks'])
    assert len(d['measurements']) == 1
    m = d['measurements'][0]
    assert m['kind'] == 'prose' and m['draft_n'] == 0 and not m['abort']
    for path, digest in d['input_sha256'].items():
        assert sha256(path) == sha256(root / Path(path).name) == digest
    helper = base / 'audit-full-replay-bandwidth-0906.py'
    spec = importlib.util.spec_from_file_location('capture_audit', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = module.capture(root / 'prose-draft0', m['counter_metadata'])
    decode = summarize_samples(samples, m['first_content_monotonic'] + .5, m['last_content_monotonic'] - .5)
    assert decode['valid'] and decode['sampled_seconds'] >= 4
    chunks = json.loads((root / 'prose-draft0/chunks.json').read_text())
    timing = [c['timings'] for c in chunks if c.get('timings')][-1]
    assert timing['predicted_n'] == 512 and timing['draft_n'] == 398 and timing['draft_n_accepted'] == 312
    assert math.isclose(timing['predicted_per_second'], 511000 / timing['predicted_ms'])
    engine = base.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
    schema_path = engine / 'tools/server/server-schema.cpp'
    context_path = engine / 'tools/server/server-context.cpp'
    schema, context = schema_path.read_text(), context_path.read_text()
    start = schema.index('// TODO: to keep things simple, we disable speculative parameter adjustments for now')
    disabled = schema[start:schema.index('#endif', start) + len('#endif')]
    assert '#if 0' in disabled and all(f'field_num("speculative.{name}"' in disabled for name in ('n_max', 'n_min', 'p_min'))
    start = context.index('    int get_n_draft_max() const {')
    limit = context[start:context.index('\n    }', start) + len('\n    }')]
    assert 'task->params.speculative' not in limit
    plan = json.loads((base / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
    full = json.loads((base / 'results/glm53-full-replay-bandwidth-0906/result.json').read_text())
    audit = dict(started=time.time(), result_sha256=sha256(root / 'result.json'),
        source_sha256={str(p): sha256(p) for p in (Path(__file__), helper, schema_path, context_path)},
        requested_draft_limit=0, observed_timings=timing, complete_counter_intervals=len(samples),
        observed_gross_decode=decode, requested_condition_honored=False, valid_zero_draft_measurement=False,
        one_draft_condition_attempted=False, schema_excerpt=disabled, draft_limit_excerpt=limit,
        real_model_requests=0, real_model_signals=0, protected={})
    for pid, expected in plan['protected'].items():
        info = process_info(int(pid))
        assert identity_matches(info, expected)
        command = full['server_command'] if pid == '4005448' else plan['original_command']
        environment = full['runtime_env'] if pid == '4005448' else plan['original_environment']
        assert info['command'] == command and runtime_environment(process_environment(int(pid))) == environment
        assert info['affinity'] == ([15, 31, 47, 63] if pid == '4005448' else list(range(128)))
        audit['protected'][pid] = dict(start=info['start'], exe=info['exe'], affinity=info['affinity'],
                                        port=expected['port'], **read_service(expected['port']))
    pinned = Path(plan['original_command'][0]).parent
    for name, digest in plan['baseline_binary_sha256'].items():
        assert sha256(pinned / name) == digest
    audit['verified_qwen_pinned_entries'] = len(plan['baseline_binary_sha256'])
    audit['current_inference_pids'] = sorted(inference_snapshot())
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(arg.decode(errors='replace')).name in ('measure-model-bandwidth.py', 'run-qwen-expert-task-rows.py')
               for arg in argv if arg):
            active.append(int(proc.name))
    assert not active, active
    audit['active_measurement_runners'] = active
    audit['note'] = ('The request sweep stopped correctly when the requested zero-draft condition was not honored. '
                     'The loaded fork fixes MTP settings at launch and compiles out these per-request fields. '
                     'The saved capture is not a zero-draft benchmark; no adjusted bandwidth is reconstructed '
                     'because its background bounds were not saved before the validity assertion. '
                     'Existing MTP2 baselines and the separate split-only trial remain unchanged.')
    audit['finished'] = time.time()
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(audit=str(destination), requested_condition_honored=False,
                         observed_timings=timing, pinned_entries=audit['verified_qwen_pinned_entries'],
                         active_measurement_runners=active, protected=audit['protected'])))


if __name__ == '__main__':
    main()
