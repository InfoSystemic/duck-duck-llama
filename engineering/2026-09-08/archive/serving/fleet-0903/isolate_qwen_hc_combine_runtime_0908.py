#!/usr/bin/env python3
"""Package and inspect the Qwen HC-fusion runtime without loading weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from isolate_flash_q8_clamp_runtime_0908 import SOURCE
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
OUT = BASE / 'results/qwen-hc-combine-runtime-0908'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard-pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = process_info(args.guard_pid)
        assert info['start'] == args.start_ticks
        guard = ModelMeasurementGuard(args.guard_pid, {args.guard_pid:18095}, inference_snapshot)
        guard.assert_idle()
        validation_path = BASE / 'results/qwen-hc-combine-proof-0908/result.json'
        build_path = BASE / 'results/qwen-hc-combine-build-0908b/result.json'
        preset_path = BASE / 'qwen-flash-20tps.json'
        validation, build, preset = [json.loads(path.read_text()) for path in (validation_path, build_path, preset_path)]
        assert validation['passed'] and len(validation['checks']) == 2
        assert sum(row['cases'] for row in validation['checks']) == 384
        assert all(row['bit_exact'] and row['scalar_exact'] for row in validation['checks'])
        assert build['build_completed'] and build['baseline_link_identical']
        assert all(sha256(p) == h for p,h in validation['input_sha256'].items())
        assert all(sha256(p) == h for p,h in build['input_sha256'].items())
        cpu = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
        assert sha256(cpu) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
        llama = Path(build['library'])
        assert sha256(llama) == build['library_sha256']
        sources = {str(path):sha256(path) for path in [Path(__file__).resolve(), validation_path, build_path,
                   preset_path, BASE / 'isolate_flash_q8_clamp_runtime_0908.py', cpu, llama]}
        for path in sorted(PINNED.glob('lib*.so*')):
            assert sha256(path) == preset['binary_sha256'][str(path)]
            sources[str(path.resolve())] = sha256(path)
        OUT.mkdir(exist_ok=False)
        binary_dir = OUT / 'bin'
        binary_dir.mkdir()
        server = binary_dir / 'llama-server'
        shutil.copy2(PINNED / 'llama-server', server)
        assert sha256(server) == preset['binary_sha256'][str(PINNED / 'llama-server')]
        for path in sorted(PINNED.glob('lib*.so*')):
            target = cpu if path.name.startswith('libggml-cpu.so') else llama if path.name.startswith('libllama.so') else path
            (binary_dir / path.name).symlink_to(target.resolve())
        source = SOURCE.replace('#include "ggml.h"', '#include "ggml.h"\n#include "llama.h"')
        source = source.replace('    Dl_info info{};', '''    llama_backend_init();
    Dl_info llama_info{};
    if (!dladdr(reinterpret_cast<void *>(llama_backend_init), &llama_info)) return 4;
    std::printf("LLAMA_SYMBOL %s\\n", llama_info.dli_fname);
    Dl_info info{};''')
        source = source.replace('        if (line.find("/libggml-cpu.so.")', '''        if (line.find("/libllama.so.") != std::string::npos) {
            std::printf("LLAMA_MAP %s\\n", line.c_str());
        }
        if (line.find("/libggml-cpu.so.")''')
        fixture = OUT / 'loader-check.cpp'
        fixture.write_text(source)
        binary = binary_dir / 'loader-check'
        command = ['c++', '-O2', '-std=c++17', '-I' + str(ENGINE / 'ggml/include'), '-I' + str(ENGINE / 'include'),
                   str(fixture), '-L' + str(binary_dir), '-Wl,-rpath,' + str(binary_dir),
                   '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
        environment = {k:v for k,v in os.environ.items() if k not in runtime_environment(os.environ)}
        environment.update(preset['runtime_env'])
        environment.update(LD_LIBRARY_PATH=str(binary_dir), GGML_QWEN_HC_COMBINE_FUSED='1')
        result = dict(started=time.time(), passed=False, sources=sources, server=str(server), server_sha256=sha256(server),
                      cpu=str(cpu), cpu_sha256=sha256(cpu), llama=str(llama), llama_sha256=sha256(llama),
                      runtime_directory=str(binary_dir), runtime_env=runtime_environment(environment),
                      fixture_sha256=sha256(fixture), compile_command=command,
                      symlinks={str(path):str(path.resolve()) for path in binary_dir.iterdir() if path.is_symlink()},
                      protected_pid=args.guard_pid, protected_start=args.start_ticks, model_loaded=False)
        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        save()
        try:
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=90)
            (OUT / 'compile.log').write_text(compiled.stdout + compiled.stderr)
            assert compiled.returncode == 0
            guard.assert_idle()
            probe = subprocess.run(['taskset', '-c', '0-127', str(binary), 'private', str(PINNED)],
                                   env=environment, cwd=BASE, capture_output=True, text=True, timeout=30)
            (OUT / 'private.log').write_text(probe.stdout + probe.stderr)
            assert probe.returncode == 0
            maps = {}
            for name, expected in [('CPU', cpu), ('LLAMA', llama)]:
                actual = sorted({line.split()[-1] for line in probe.stdout.splitlines() if line.startswith(name + '_MAP ')})
                symbols = [line.removeprefix(name + '_SYMBOL ') for line in probe.stdout.splitlines() if line.startswith(name + '_SYMBOL ')]
                assert actual == [str(expected.resolve())] and len(symbols) == 1 and Path(symbols[0]).resolve() == expected.resolve()
                maps[name] = actual
            devices = [line.removeprefix('DEVICE ') for line in probe.stdout.splitlines() if line.startswith('DEVICE ')]
            assert all('CPU-NUMA' + str(i) in devices for i in range(4))
            guard.assert_idle()
            listed = subprocess.run(['taskset', '-c', '0-127', str(server), '--list-devices'],
                                    env=environment, cwd=BASE, capture_output=True, text=True, timeout=30)
            (OUT / 'server-devices.log').write_text(listed.stdout + listed.stderr)
            assert listed.returncode == 0 and all('CPU-NUMA' + str(i) in listed.stdout + listed.stderr for i in range(4))
            assert all(sha256(path) == digest for path, digest in sources.items())
            assert sha256(server) == result['server_sha256']
            guard.assert_idle()
            result.update(passed=True, maps=maps, devices=devices, server_devices_exit=listed.returncode)
            print(json.dumps(dict(passed=True, cpu_sha256=result['cpu_sha256'], server_sha256=result['server_sha256'], devices=devices)), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
