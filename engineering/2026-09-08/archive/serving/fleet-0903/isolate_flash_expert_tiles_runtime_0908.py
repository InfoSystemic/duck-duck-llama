#!/usr/bin/env python3
"""Probe the private Q8 expert tile runtime without loading model weights."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager, PORT
from isolate_flash_q8_clamp_runtime_0908 import SOURCE
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, runtime_environment, sha256

OUT = BASE / 'results/glm-flash-q8-expert-tiles-runtime-0908'
BIN = OUT / 'bin'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'


def main():
    os.umask(0o077)
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        parent, parent_env = manager.candidate(True, 0, 15, q8_experts=True, q8_clamp=True, pooling=True)
        for key in ('command', 'runtime_env', 'cpu_sha256'):
            assert parent[key] == current[key]
        validation_path = BASE/'results/glm-flash-q8-expert-tiles-0908/result.json'
        validation = json.loads(validation_path.read_text())
        manifest_path = validation_path.parent/'private-cpu/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        rotating_path = BASE/'results/glm-flash-q8-expert-tiles-rotating-0908/result.json'
        rotating = json.loads(rotating_path.read_text())
        assert validation['passed'] and rotating['passed']
        assert validation['baseline_link_identical'] and validation['unpatched_text_identical']
        assert len(validation['comparisons']) == 90 and len(rotating['comparisons']) == 432
        assert all(x['bit_exact'] for x in validation['comparisons'] + rotating['comparisons'])
        assert validation['parent_sha256'] == manifest['parent_sha256'] == current['cpu_sha256']
        cpu = Path(validation['library'])
        assert sha256(cpu) == validation['library_sha256'] == manifest['library_sha256']
        preset_path = BASE/'glm-flash-validated.json'
        preset = json.loads(preset_path.read_text())
        sources = {str(PINNED/name): digest for name, digest in preset['binary_sha256'].items()}
        sources.update(manifest['input_sha256'])
        sources.update(manifest['private_source_sha256'])
        sources.update(rotating['input_sha256'])
        for path in (validation_path, manifest_path, rotating_path, preset_path, Path(__file__),
                     BASE/'glm_flash_q8_trial.py', BASE/'isolate_flash_q8_clamp_runtime_0908.py'):
            sources[str(path)] = sha256(path)
        assert all(sha256(path) == digest for path, digest in sources.items())
        OUT.mkdir(exist_ok=False)
        BIN.mkdir()
        server = BIN/'llama-server'
        shutil.copy2(PINNED/'llama-server', server)
        assert sha256(server) == preset['binary_sha256']['llama-server']
        for path in sorted(PINNED.glob('lib*.so*')):
            if not path.name.startswith('libggml-cpu'):
                (BIN/path.name).symlink_to(path.resolve())
        for name in ('libggml-cpu.so', 'libggml-cpu.so.0', 'libggml-cpu.so.0.22.0'):
            (BIN/name).symlink_to(cpu.resolve())
        fixture = OUT/'loader-check.cpp'
        fixture.write_text(SOURCE)
        binary = BIN/'loader-check'
        command = ['c++', '-O2', '-std=c++17', '-I'+str(ENGINE/'ggml/include'), str(fixture),
                   '-L'+str(BIN), '-Wl,-rpath,'+str(BIN), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
        result = dict(started=time.time(), passed=False, sources=sources, server=str(server),
                      server_sha256=sha256(server), cpu=str(cpu), cpu_sha256=sha256(cpu),
                      runtime_directory=str(BIN), compile_command=command, fixture_sha256=sha256(fixture),
                      symlinks={str(p): str(p.resolve()) for p in BIN.iterdir() if p.is_symlink()}, runs=[])
        (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())

        def save():
            (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')

        save()
        try:
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (OUT/'compile.log').write_text(compiled.stdout+compiled.stderr)
            assert compiled.returncode == 0
            for tile_rows in (32, 48, 64):
                guard.assert_idle()
                environment = dict(parent_env)
                environment.update(LD_LIBRARY_PATH=str(cpu.parent)+':'+str(PINNED),
                                   GGML_CPU_Q8_MOE_TILE_ROWS=str(tile_rows))
                probe = subprocess.run(['taskset', '-c', '0-127', str(binary), 'private', str(PINNED)],
                                       env=environment, cwd=BASE, capture_output=True, text=True, timeout=30)
                (OUT/f'private-{tile_rows}.log').write_text(probe.stdout+probe.stderr)
                assert probe.returncode == 0
                maps = sorted({line.split()[-1] for line in probe.stdout.splitlines() if line.startswith('CPU_MAP ')})
                symbols = [line.removeprefix('CPU_SYMBOL ') for line in probe.stdout.splitlines() if line.startswith('CPU_SYMBOL ')]
                devices = [line.removeprefix('DEVICE ') for line in probe.stdout.splitlines() if line.startswith('DEVICE ')]
                assert maps == [str(cpu.resolve())]
                assert len(symbols) == 1 and Path(symbols[0]).resolve() == cpu.resolve()
                assert all(f'CPU-NUMA{i}' in devices for i in range(4))
                result['runs'].append(dict(mode='private', tile_rows=tile_rows, cpu_maps=maps,
                                           cpu_symbol=symbols[0], devices=devices,
                                           runtime_env=runtime_environment(environment)))
                save()
                print(json.dumps(dict(tile_rows=tile_rows, cpu_maps=maps, devices=devices)), flush=True)
            assert all(sha256(path) == digest for path, digest in sources.items())
            assert sha256(cpu) == result['cpu_sha256'] and sha256(server) == result['server_sha256']
            manager.validate_current()
            guard.assert_idle()
            result['passed'] = True
            print(json.dumps(dict(passed=True, cpu_sha256=result['cpu_sha256'], server_sha256=result['server_sha256'])), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
