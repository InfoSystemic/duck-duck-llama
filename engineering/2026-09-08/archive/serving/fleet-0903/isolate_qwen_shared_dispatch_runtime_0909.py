#!/usr/bin/env python3
"""Package the validated shared-dispatch CPU/base runtime without loading weights."""
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
OUT = BASE / 'results/qwen-shared-dispatch-runtime-0909'


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
        validation_path = BASE / 'results/qwen-shared-dispatch-numa-0909/result.json'
        lifecycle_path = BASE / 'results/qwen-shared-dispatch-lifecycle-0909/result.json'
        build_path = BASE / 'results/qwen-shared-dispatch-build-0909b/result.json'
        preset_path = BASE / 'qwen-flash-20tps.json'
        validation, lifecycle, build, preset = [json.loads(path.read_text()) for path in (validation_path, lifecycle_path, build_path, preset_path)]
        assert validation['passed'] and validation['bit_exact'] and validation['arms'] == 24
        assert lifecycle['passed'] and lifecycle['bit_exact'] and len(lifecycle['checks']) == 3
        assert build['passed'] and build['build_completed']
        assert all(build['baseline_objects'].values()) and all(build['baseline_libraries'].values())
        assert all(sha256(p) == h for p,h in build['private_source_sha256'].items())
        assert sha256(validation_path.parent / 'numa-check.cpp') == validation['fixture_sha256']
        for record in (validation, lifecycle, build):
            assert all(sha256(p) == h for p,h in record['input_sha256'].items())
        cpu = Path(build['libraries']['cpu']['path'])
        base_library = Path(build['libraries']['base']['path'])
        assert sha256(cpu) == build['libraries']['cpu']['sha256']
        assert sha256(base_library) == build['libraries']['base']['sha256']
        assert build['parent_cpu_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
        assert build['parent_base_sha256'] == 'a8d6ff25ffab12b993c98f11c6d5b4146fc1dd5bd7bebb8c43303842cd959d36'
        llama_build_path = BASE / 'results/qwen-hc-norm-flat-build-0908/result.json'
        llama_build = json.loads(llama_build_path.read_text())
        assert llama_build['build_completed'] and llama_build['baseline_link_identical']
        assert all(sha256(p) == h for p,h in llama_build['input_sha256'].items())
        llama = Path(llama_build['library'])
        assert sha256(llama) == llama_build['library_sha256'] == 'd0b2321eae443255dd84a5ad98cda8d3d9bf5540a31891bdc5c03be91061d647'
        sources = {str(path):sha256(path) for path in [Path(__file__).resolve(), validation_path, lifecycle_path, build_path,
                   preset_path, llama_build_path, BASE / 'isolate_flash_q8_clamp_runtime_0908.py', cpu, base_library, llama]}
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
            target = cpu if path.name.startswith('libggml-cpu.so') else base_library if path.name.startswith('libggml-base.so') else llama if path.name.startswith('libllama.so') else path
            (binary_dir / path.name).symlink_to(target.resolve())
        source = SOURCE.replace('#include "ggml.h"', '#include "ggml.h"\n#include "llama.h"')
        source = source.replace('    Dl_info info{};', '''    llama_backend_init();
    Dl_info llama_info{};
    if (!dladdr(reinterpret_cast<void *>(llama_backend_init), &llama_info)) return 4;
    std::printf("LLAMA_SYMBOL %s\\n", llama_info.dli_fname);
    Dl_info base_info{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_graph_compute), &base_info)) return 5;
    std::printf("BASE_SYMBOL %s\\n", base_info.dli_fname);
    Dl_info info{};''')
        source = source.replace('        if (line.find("/libggml-cpu.so.")', '''        if (line.find("/libggml-base.so.") != std::string::npos) {
            std::printf("BASE_MAP %s\\n", line.c_str());
        }
        if (line.find("/libllama.so.") != std::string::npos) {
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
        environment.update(LD_LIBRARY_PATH=str(binary_dir), GGML_QWEN_HC_COMBINE_FUSED='1', GGML_QWEN_HC_MIX_FUSED='1', GGML_QWEN_HC_NORM_FLAT='1', GGML_CPU_X16_QUANTIZE_BLOCKS='0', GGML_CPU_NUMA_HUGEPAGES='0', GGML_CPU_NUMA_SHARED_DISPATCH='1')
        result = dict(started=time.time(), passed=False, sources=sources, server=str(server), server_sha256=sha256(server),
                      cpu=str(cpu), cpu_sha256=sha256(cpu), base=str(base_library), base_sha256=sha256(base_library),
                      llama=str(llama), llama_sha256=sha256(llama),
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
            for name, expected in [('CPU', cpu), ('BASE', base_library), ('LLAMA', llama)]:
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
