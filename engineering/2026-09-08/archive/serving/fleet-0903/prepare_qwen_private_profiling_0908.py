#!/usr/bin/env python3
"""Derive a private, inherited-lock profiling controller from audited helpers."""
import ast
import hashlib
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:160]
    return source.replace(old, new, 1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    original = BASE / 'profile_qwen_resident_0908.py'
    profile = original.read_text()
    profile = replace_once(profile, 'Profile resident Qwen Q6 after warmup without adopting or reconfiguring it.',
                           'Profile an owned private Qwen trial after its unprofiled measurement.')
    profile = profile.replace("default='2'", "default='4'").replace("default='4005448'", "default='1219506'")
    profile = replace_once(profile, "parser.add_argument('--resident-controller',type=Path,required=True)",
        "parser.add_argument('--trial-plan',type=Path,required=True)\nparser.add_argument('--trial-result',type=Path,required=True)\nparser.add_argument('--lifecycle-lock-fd',type=int,required=True)")
    profile = replace_once(profile,
        "assert options.port == 18095 and options.alias == 'qwen-goal' and options.allowed_idle_pids == ''\nassert int(options.drafts) == 4",
        "assert options.port == 18155 and options.alias == 'qwen-q6-private'\nassert int(options.drafts) in (3, 4)")
    profile = replace_once(profile,
        "lifecycle_lock = (base / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a')\nfcntl.flock(lifecycle_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "lock_path = base / 'results/qwen-q6-trial-0907/lifecycle.lock'\nlock_stat, inherited_stat = lock_path.stat(), os.fstat(options.lifecycle_lock_fd)\nassert (lock_stat.st_dev, lock_stat.st_ino) == (inherited_stat.st_dev, inherited_stat.st_ino)\nfcntl.flock(options.lifecycle_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)")
    profile = replace_once(profile, "'resident_controller':str(options.resident_controller)",
        "'trial_plan':str(options.trial_plan),'trial_result':str(options.trial_result)")
    begin = profile.index('from qwen_high_quant_trial import unit_state\n')
    end = profile.index('\nsave()\ntry:\n', begin)
    profile = profile[:begin] + '''from qwen_private_profile_trial_0908 import verify_plan, memory_gate
trial = json.loads(options.trial_result.read_text())
trial_plan = json.loads(options.trial_plan.read_text())
assert trial['controller_pid'] == os.getppid() and trial['model_pid'] == options.pid
assert trial['stage'] == 'profiling private Qwen' and trial['passed'] and not trial.get('error')
assert trial['plan_sha256'] == hashlib.sha256(options.trial_plan.read_bytes()).hexdigest()
assert trial['measurement'] == str(options.after)
assert trial['measurement_sha256'] == hashlib.sha256(options.after.read_bytes()).hexdigest()
verify_plan(trial_plan)
memory_gate(loading=True)
assert options.allowed_idle_pids == str(trial_plan['peer_pid'])
assert int(options.drafts) == trial_plan['drafts']
assert trial_plan['profile']
current = dict(pid=options.pid, info=trial['current'], command=trial_plan['command'],
               runtime_env=trial_plan['runtime_env'], workers=15, drafts=trial_plan['drafts'])
result['current'] = current
plan = dict(protected={str(options.pid): dict(start_ticks=current['info']['start'], exe=current['info']['exe']),
            str(trial_plan['peer_pid']): dict(start_ticks=trial_plan['peer']['start'], exe=trial_plan['peer']['exe'])},
            original_command=current['command'], original_environment=current['runtime_env'])
plan_path = out / 'plan.json'
plan_path.write_text(json.dumps(plan, indent=2) + '\\n')
trial_snapshot = out / 'trial-controller-at-start.json'
trial_snapshot.write_bytes(options.trial_result.read_bytes())
result['protected_before'] = {}
for pid, expected in plan['protected'].items():
    info = process_info(int(pid))
    assert identity_matches(info, expected)
    result['protected_before'][pid] = dict(info=info, environment=runtime_environment(process_environment(int(pid))))
assert result['protected_before'][str(options.pid)]['info']['command'] == plan['original_command']
assert result['protected_before'][str(options.pid)]['environment'] == plan['original_environment']
source_paths = [Path(__file__).resolve(), base / 'model_measurement_guard.py', base / 'guarded_inference_request.py',
                base / 'qwen_split_trial.py', base / 'inference_contention_guard.py',
                base / 'qwen_private_profile_trial_0908.py', options.trial_plan, trial_snapshot, plan_path]
result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
for p in source_paths:
    destination = out / p.name
    if destination != p:
        if destination.exists():
            destination = out / ('source-' + p.parent.name + '-' + p.name)
        assert not destination.exists()
        destination.write_bytes(p.read_bytes())
guard = ModelMeasurementGuard(options.pid, {options.pid:18155, trial_plan['peer_pid']:18095}, snapshot)
result['note'] = 'Six-second cycle profiles of private Qwen Q6 after 32 streamed events, after an unprofiled measurement. Effective speculative settings are verified from launch; request overrides are ignored. Prompt reuse is disabled. Profiled rates are excluded from throughput and bandwidth claims.'
''' + profile[end:]
    profile = profile.replace("'--','sleep','8'", "'--','sleep','6'")
    profile = replace_once(profile, "last-first>=10", "last-first>=8")
    profile = replace_once(profile,
        "chunks = stream_completion(options.port, body, guard.abort_reason, check['abort'], interval=0.5)",
        "chunks = stream_completion(options.port, body, guard.abort_reason, check['abort'], interval=0.5, max_elapsed=180)")
    profile = replace_once(profile,
        "stream_completion(options.port,payload,check_abort,entry['abort'],on_chunk)",
        "stream_completion(options.port,payload,check_abort,entry['abort'],on_chunk,max_elapsed=180)")

    original_trial = BASE / 'qwen_private_trial_0908.py'
    controller = original_trial.read_text()
    controller = replace_once(controller, 'def execute(out):', 'def execute(out, lifecycle_lock_fd=None):')
    controller = replace_once(controller, "    records, source_paths = model_records(preset)",
        "    assert not args.profile or args.drafts in (3, 4)\n    records, source_paths = model_records(preset)\n    source_paths.append(BASE / 'profile_qwen_private_0908.py')")
    controller = replace_once(controller, "port=PORT, command=command, runtime_env=env, drafts=args.drafts, barrier=args.barrier,",
        "port=PORT, command=command, runtime_env=env, drafts=args.drafts, barrier=args.barrier, profile=args.profile,")
    controller = replace_once(controller,
        "        guard = ModelMeasurementGuard(model.pid, {model.pid:PORT, plan['peer_pid']:18095}, inference_snapshot)",
        "        class ReservedMemoryGuard(ModelMeasurementGuard):\n            def inspect(self, *args, **kwargs):\n                memory_gate(loading=True)\n                return super().inspect(*args, **kwargs)\n        guard = ReservedMemoryGuard(model.pid, {model.pid:PORT, plan['peer_pid']:18095}, inference_snapshot)\n        save('waiting for quiet decode')")
    marker = "        print(json.dumps(dict(passed=True, rows=rows, target_reached=result['target_reached'])), flush=True)"
    controller = replace_once(controller, marker, '''        if plan['profile']:
            assert lifecycle_lock_fd is not None
            save('profiling private Qwen')
            profile_label = plan['label'] + '-profile'
            command = [sys.executable, '-u', str(BASE / 'profile_qwen_private_0908.py'), profile_label,
                       '--pid', str(model.pid), '--port', str(PORT), '--alias', 'qwen-q6-private',
                       '--drafts', str(plan['drafts']), '--allowed-idle-pids', str(plan['peer_pid']),
                       '--trial-plan', str(out / 'plan.json'), '--trial-result', str(out / 'result.json'),
                       '--after', str(path), '--lifecycle-lock-fd', str(lifecycle_lock_fd), '--call-graph', 'dwarf,8192']
            measurement = subprocess.Popen(command, cwd=BASE, start_new_session=True, pass_fds=(lifecycle_lock_fd,))
            while measurement.poll() is None:
                peer_guard.assert_idle()
                memory_gate(loading=True)
                assert model.poll() is None, 'Private model exited during profiling'
                time.sleep(.5)
            assert measurement.returncode == 0, 'Diagnostic profile did not complete'
            profile_path = BASE / 'results' / profile_label / 'result.json'
            profiled = json.loads(profile_path.read_text())
            assert profiled['completed'] and not profiled.get('error')
            assert len(profiled['profiles']) == 2 and all(not row['abort'] for row in profiled['profiles'])
            result.update(profile=str(profile_path), profile_sha256=sha256(profile_path))
            verify_plan(plan)
''' + marker)
    controller = replace_once(controller, "        result['error'] = repr(error)",
        "        result['passed'] = False\n        result['error'] = repr(error)")
    controller = replace_once(controller, "    args = parser.parse_args()",
        "    parser.add_argument('--profile', action='store_true')\n    args = parser.parse_args()")
    controller = replace_once(controller, "else execute(out)", "else execute(out, lock.fileno())")

    outputs = {'profile_qwen_private_0908.py':profile, 'qwen_private_profile_trial_0908.py':controller}
    for name, source in outputs.items():
        ast.parse(source)
        target = BASE / name
        assert not target.exists(), target
    for name, source in outputs.items():
        (BASE / name).write_text(source)
    result = dict(prepared=time.time(), executed=False,
        original_sha256={str(path):digest(path) for path in (original, original_trial, Path(__file__).resolve())},
        outputs={str(BASE / name):digest(BASE / name) for name in outputs},
        scope='Prepared optional private profile after an unprofiled model benchmark, with inherited lifecycle lock, verified parent identity, peer and RAM monitoring. No capture or new speed result.')
    output = BASE / 'results/qwen-30tps-tools-0908/private-profile-preparation.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    prepare()
