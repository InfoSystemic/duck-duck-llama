#!/usr/bin/env python3
"""Package the corrected gather CPU with the previously validated Qwen runtime."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-get-rows-runtime-0909'
FIXED_SHA = '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'


def package(peer_pid, start_ticks):
    assert process_info(peer_pid)['start'] == start_ticks
    guard = ModelMeasurementGuard(peer_pid, {peer_pid: 18095}, inference_snapshot)
    guard.assert_idle()
    paths = [BASE / ('results/' + name + '/result.json') for name in
             ('qwen-shared-dispatch-runtime-0909', 'qwen-get-rows-columns-build-0909',
              'qwen-get-rows-columns-validation-0909')]
    parent, build, validation = [json.loads(path.read_text()) for path in paths]
    assert all(row['passed'] and row['finished'] and not row.get('error') for row in (parent, build, validation))
    assert build['baseline_object_identical'] and build['baseline_library_identical']
    assert build['parent_sha256'] == parent['cpu_sha256'] == sha256(parent['cpu'])
    assert validation['fixed_cpu_sha256'] == build['library_sha256'] == sha256(build['library']) == FIXED_SHA
    assert validation['parent_cpu_sha256'] == parent['cpu_sha256']
    assert len(validation['direct_checks']) == len(validation['graph_checks']) == 4
    assert all(row['unexpected_failures'] == 0 and row['inputs_preserved'] for row in validation['direct_checks'])
    assert all(row['known_quantized_failures'] == 0 for row in validation['direct_checks'] if row['name'].startswith('fixed'))
    assert all(row['cases'] == 46 and row['bit_exact'] for row in validation['graph_checks'])
    assert validation['peer_preserved']
    for record in (parent, build, validation):
        for key in ('sources', 'input_sha256', 'private_source_sha256'):
            assert all(sha256(path) == digest for path, digest in record.get(key, {}).items()), key
    for key in ('server', 'base', 'llama'):
        assert sha256(parent[key]) == parent[key + '_sha256']
    source_bin = Path(parent['runtime_directory'])
    loader_source = source_bin.parent / 'loader-check.cpp'
    assert sha256(loader_source) == parent['fixture_sha256']
    cpu, base_library, llama = [Path(path) for path in (build['library'], parent['base'], parent['llama'])]
    inputs = [Path(__file__).resolve(), *paths, source_bin / 'loader-check', loader_source,
              BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py',
              Path(parent['server']), cpu, base_library, llama]
    inputs += [path.resolve() for path in source_bin.glob('lib*.so*')]
    sources = {str(path): sha256(path) for path in inputs}
    OUT.mkdir(exist_ok=False)
    binary_dir = OUT / 'bin'
    binary_dir.mkdir()
    for name in ('llama-server', 'loader-check'):
        shutil.copy2(source_bin / name, binary_dir / name)
        assert sha256(binary_dir / name) == sha256(source_bin / name)
    for path in sorted(source_bin.glob('lib*.so*')):
        target = cpu if path.name.startswith('libggml-cpu.so') else path.resolve()
        (binary_dir / path.name).symlink_to(target.resolve())
    runtime = dict(parent['runtime_env'], LD_LIBRARY_PATH=str(binary_dir))
    environment = {key: value for key, value in os.environ.items() if key not in runtime_environment(os.environ)}
    environment.update(runtime)
    result = dict(started=time.time(), passed=False, model_loaded=False, sources=sources,
                  server=str(binary_dir / 'llama-server'), server_sha256=parent['server_sha256'],
                  cpu=str(cpu), cpu_sha256=sha256(cpu), base=str(base_library), base_sha256=sha256(base_library),
                  llama=str(llama), llama_sha256=sha256(llama), runtime_directory=str(binary_dir), runtime_env=runtime,
                  loader_sha256=sha256(binary_dir / 'loader-check'),
                  symlinks={str(path): str(path.resolve()) for path in binary_dir.iterdir() if path.is_symlink()},
                  protected_pid=peer_pid, protected_start=start_ticks,
                  scope='Only the corrected gather CPU changes; reuse the unchanged server and loader executable. No compilation or model weights.')

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        guard.assert_idle()
        probe = subprocess.run(['taskset', '-c', '0-127', str(binary_dir / 'loader-check'), 'private', str(source_bin)],
                               cwd=BASE, env=environment, capture_output=True, text=True, timeout=30)
        (OUT / 'private.log').write_text(probe.stdout + probe.stderr)
        assert probe.returncode == 0
        maps = {}
        for name, expected in [('CPU', cpu), ('BASE', base_library), ('LLAMA', llama)]:
            actual = sorted({line.split()[-1] for line in probe.stdout.splitlines() if line.startswith(name + '_MAP ')})
            symbols = [line.removeprefix(name + '_SYMBOL ') for line in probe.stdout.splitlines() if line.startswith(name + '_SYMBOL ')]
            assert actual == [str(expected.resolve())] and len(symbols) == 1 and Path(symbols[0]).resolve() == expected.resolve()
            maps[name] = actual
        devices = [line.removeprefix('DEVICE ') for line in probe.stdout.splitlines() if line.startswith('DEVICE ')]
        assert all('CPU-NUMA' + str(i) in devices for i in range(4))
        guard.assert_idle()
        listed = subprocess.run(['taskset', '-c', '0-127', result['server'], '--list-devices'],
                                cwd=BASE, env=environment, capture_output=True, text=True, timeout=30)
        (OUT / 'server-devices.log').write_text(listed.stdout + listed.stderr)
        assert listed.returncode == 0 and all('CPU-NUMA' + str(i) in listed.stdout + listed.stderr for i in range(4))
        assert all(sha256(path) == digest for path, digest in sources.items())
        assert sha256(result['server']) == result['server_sha256']
        guard.assert_idle()
        result.update(passed=True, maps=maps, devices=devices, peer_preserved=process_info(peer_pid)['start'] == start_ticks)
        print(json.dumps(dict(runtime_passed=True, cpu_sha256=result['cpu_sha256'], devices=devices)), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peer-pid', type=int, default=1219506)
    parser.add_argument('--start-ticks', default='103969952')
    args = parser.parse_args()
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        package(args.peer_pid, args.start_ticks)
