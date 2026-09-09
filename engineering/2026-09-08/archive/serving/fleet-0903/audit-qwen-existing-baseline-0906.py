#!/usr/bin/env python3
"""Reconcile the saved baseline and read current service state; never send work."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from dram_bandwidth import parse_records, summarize_samples
from model_measurement_guard import read_service


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    base = Path(__file__).resolve().parent
    directory = base / 'results/qwen-even-split-existing-baseline-0906'
    destination = directory / 'validation-audit.json'
    assert not destination.exists(), 'Keep earlier audit evidence intact'
    result_path = directory / 'result.json'
    result = json.loads(result_path.read_text())
    staging = base / 'results/qwen-even-split-model-trial-0906-staging'
    plan = json.loads((staging / 'plan.json').read_text())
    assert result.get('finished') and not result.get('error')
    assert result['input_integrity_verified']
    assert result['idle_gate']['quiet_seconds'] >= 60
    assert len(result['checks']) == 2 and all(
        c['pass_check'] and not c['abort'] for c in result['checks'])
    assert len(result['measurements']) == 2
    audit = dict(time=time.time(), result_sha256=digest(result_path),
                 validated_sources={}, measurements=[], protected={},
                 real_model_requests=0, note='Read-only validation; IMC attribution remains approximate.')
    for name, expected in result['input_sha256'].items():
        assert digest(name) == digest(directory / Path(name).name) == expected, name
        audit['validated_sources'][name] = expected
    tests = json.loads((staging / 'measurement-guard-check.json').read_text())
    assert len(tests['cases']) == 9 and all(x['passed'] for x in tests['cases'])
    assert tests['real_model_requests'] == 0
    for name, expected in tests['source_sha256'].items():
        assert digest(name) == expected, name
    audit['guard_checks'] = dict(cases=9, all_passed=True,
                                evidence_sha256=digest(staging / 'measurement-guard-check.json'))
    for measurement in result['measurements']:
        assert measurement['draft_n'] == 2 and not measurement['abort']
        assert not measurement['inference_churn']
        assert all(x['pid'] == 4005448 and x['cpu_percent'] < 1
                   for x in measurement['other_inference'])
        case = directory / (measurement['kind'] + '-draft2')
        capture = json.loads((case / 'samples.json').read_text())
        metadata = capture['metadata']
        assert metadata == measurement['counter_metadata']
        assert metadata['valid'] and metadata['exit_code'] == 0 and not metadata['errors']
        assert metadata['required_counters_per_interval'] == 48
        assert all(s['valid'] and len(s['sockets']) == 4 for s in capture['samples'])
        records, running = [], []
        for line in (case / 'perf.csv').read_text().splitlines(True):
            row = next(csv.reader([line]))
            if len(row) >= 7 and row[1].startswith('CPU'):
                elapsed = float(row[0])
                running.append(float(row[6]))
                records.append((metadata['anchor_monotonic'] + elapsed, line))
        sockets = {int(k): v for k, v in metadata['socket_ids'].items()}
        reconstructed, reconstructed_metadata = parse_records(
            records, metadata['cpus'], metadata['pmus'], sockets)
        assert reconstructed_metadata['valid'] and min(running) == 100, 'Multiplexed counters'
        assert len(reconstructed) == len(capture['samples'])
        for actual, saved in zip(reconstructed, capture['samples']):
            assert actual['sockets'] == saved['sockets']
            assert actual['duration'] == saved['duration']
            assert math.isclose(actual['start'], saved['start'], abs_tol=1e-6, rel_tol=0)
            assert math.isclose(actual['end'], saved['end'], abs_tol=1e-6, rel_tol=0)
        decode = summarize_samples(capture['samples'], measurement['first_content_monotonic'] + 0.5,
                                   measurement['last_content_monotonic'] - 0.5)
        assert decode == measurement['decode'] and decode['sampled_seconds'] >= 4
        for key in ('baseline_before', 'baseline_after'):
            window = measurement[key]
            assert window['valid'] and window['samples'] >= 8
            assert window['sampled_seconds'] >= 4
        background = max(measurement[key]['total_gb_s'] for key in ('baseline_before', 'baseline_after'))
        adjusted = max(0, decode['total_gb_s'] - background)
        assert adjusted == measurement['background_subtracted_gb_s']
        assert adjusted / 380 == measurement['background_subtracted_utilization']
        chunks = json.loads((case / 'chunks.json').read_text())
        timings = [chunk['timings'] for chunk in chunks if chunk.get('timings')][-1]
        assert timings == measurement['timings']
        assert timings['predicted_n'] > 0 and 0 < timings['draft_n_accepted'] <= timings['draft_n']
        rate = timings['predicted_per_second']
        assert math.isclose(rate, 1000 * (timings['predicted_n'] - 1) / timings['predicted_ms'])
        audit['measurements'].append(dict(
            kind=measurement['kind'], tokens=timings['predicted_n'], tok_s=rate,
            draft_accepted=timings['draft_n_accepted'], draft_offered=timings['draft_n'],
            adjusted_gb_s=adjusted, utilization_percent=100 * adjusted / 380,
            approximate_gb_per_output_token=adjusted / rate,
            decode_samples=decode['samples'], decode_sampled_seconds=decode['sampled_seconds'],
            complete_counter_intervals=len(reconstructed), minimum_counter_running_percent=min(running),
            finish_reasons=measurement['finish_reasons'],
            other_inference=measurement['other_inference'],
            other_host_cpu=measurement['other_host_cpu'],
            artifact_sha256={path.name: digest(path) for path in case.iterdir() if path.is_file()}))
    for pid, expected in plan['protected'].items():
        proc = Path('/proc') / pid
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        assert fields[19] == expected['start_ticks'] and fields[0] != 'Z', pid
        exe = os.readlink(proc / 'exe')
        assert exe == expected['exe'], pid
        command = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        ports = [int(command[i + 1]) for i, arg in enumerate(command[:-1]) if arg == '--port']
        assert ports[-1] == expected['port'], pid
        state = read_service(ports[-1])
        audit['protected'][pid] = dict(start_ticks=fields[19], exe=exe, port=ports[-1],
                                       cwd=os.readlink(proc / 'cwd'),
                                       affinity=sorted(os.sched_getaffinity(int(pid))), **state)
        if pid == '2308651':
            assert command == plan['original_command']
            environment = {}
            for item in (proc / 'environ').read_bytes().split(b'\0'):
                if b'=' not in item:
                    continue
                key, value = item.split(b'=', 1)
                if key.startswith((b'GGML_', b'LLAMA_GRAPH_PHASE', b'LLAMA_MTP_', b'OMP_', b'GOMP_')) or key == b'LD_LIBRARY_PATH':
                    environment[key.decode()] = value.decode()
            assert environment == plan['original_environment']
            audit['protected'][pid]['command_and_runtime_env_unchanged'] = True
    binary_dir = Path(plan['original_command'][0]).parent
    for name, expected in plan['baseline_binary_sha256'].items():
        assert digest(binary_dir / name) == expected, name
    assert digest(plan['candidate_library']) == plan['candidate_library_sha256']
    audit['verified_pinned_files'] = len(plan['baseline_binary_sha256'])
    audit['candidate_library_sha256'] = plan['candidate_library_sha256']
    previous = json.loads((base / 'results/qwen-expert-moe-split-0906-final-state.json').read_text())
    mapped_path = previous['qwen_mapped_library']['map_file']
    mapped = subprocess.run(['sudo', '-n', 'python3', '-c',
                             'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())',
                             mapped_path], check=True, capture_output=True, text=True).stdout.strip()
    assert mapped == plan['baseline_binary_sha256']['libllama.so.0.3.0']
    audit['mapped_qwen_library_sha256'] = mapped
    audit['active_benchmark_processes'] = []
    benchmark_names = {'measure-model-bandwidth.py', 'run-interleaved-full-graph.py',
                       'run-flash-bandwidth-baselines.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if any(Path(arg).name in benchmark_names for arg in args[:3]):
                audit['active_benchmark_processes'].append(int(proc.name))
        except (OSError, UnicodeError):
            pass
    assert not audit['active_benchmark_processes']
    with socket.socket() as sock:
        sock.settimeout(1)
        trial_listening = sock.connect_ex(('127.0.0.1', 18155)) == 0
    audit['trial_port_listening'] = trial_listening
    assert not trial_listening
    audit['finished'] = time.time()
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(audit=str(destination), measurements=audit['measurements'],
                          protected=audit['protected'], active_benchmarks=[],
                          candidate_deployed=False), indent=2))


if __name__ == '__main__':
    main()
