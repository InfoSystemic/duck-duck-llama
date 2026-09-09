#!/usr/bin/env python3
"""Verify backend discovery in a private runtime without loading model weights."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE / 'results/glm-flash-q8-clamp-runtime-0908'
BIN = OUT / 'bin'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'

SOURCE = r'''
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    if (!std::strcmp(argv[1], "pinned")) ggml_backend_load_all_from_path(argv[2]);
    else if (!std::strcmp(argv[1], "private")) ggml_backend_load_all();
    else return 2;
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &info)) return 3;
    std::printf("CPU_SYMBOL %s\n", info.dli_fname);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        std::printf("DEVICE %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i)));
    }
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("/libggml-cpu.so.") != std::string::npos) {
            std::printf("CPU_MAP %s\n", line.c_str());
        }
    }
    return 0;
}
'''


def main():
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        assert manager.state['current'] is None
        current = manager.qwen.state['current']
        assert current and current['drafts'] == 4 and current['q8_wide_batch']
        assert manager.qwen.state['full_stopped'] and not current.get('op_profile_count')
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18095}, inference_snapshot)
        guard.assert_idle()
        validation = json.loads((BASE / 'results/glm-flash-q8-clamp-0908/result.json').read_text())
        assert validation['passed']
        cpu = Path(validation['library'])
        assert sha256(cpu) == validation['library_sha256']
        preset = json.loads((BASE / 'glm-flash-validated.json').read_text())
        sources = {str(PINNED / name): digest for name, digest in preset['binary_sha256'].items()}
        assert all(sha256(path) == digest for path, digest in sources.items())
        OUT.mkdir()
        BIN.mkdir()
        server = BIN / 'llama-server'
        shutil.copy2(PINNED / 'llama-server', server)
        assert sha256(server) == preset['binary_sha256']['llama-server']
        for path in sorted(PINNED.glob('lib*.so*')):
            if path.name.startswith('libggml-cpu'):
                continue
            (BIN / path.name).symlink_to(path.resolve())
        for name in ('libggml-cpu.so', 'libggml-cpu.so.0', 'libggml-cpu.so.0.22.0'):
            (BIN / name).symlink_to(cpu.resolve())
        fixture = OUT / 'loader-check.cpp'
        fixture.write_text(SOURCE)
        binary = BIN / 'loader-check'
        command = ['c++', '-O2', '-std=c++17', '-I' + str(ENGINE / 'ggml/include'),
                   str(fixture), '-L' + str(BIN), '-Wl,-rpath,' + str(BIN),
                   '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
        result = dict(started=time.time(), passed=False, sources=sources,
                      source_sha256=sha256(__file__), fixture_sha256=sha256(fixture),
                      server=str(server), server_sha256=sha256(server), cpu=str(cpu),
                      cpu_sha256=sha256(cpu), runtime_directory=str(BIN), compile_command=command,
                      symlinks={str(p): str(p.resolve()) for p in BIN.iterdir() if p.is_symlink()}, runs=[])
        (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        save = lambda: (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        save()
        try:
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (OUT / 'compile.log').write_text(compiled.stdout + compiled.stderr)
            assert compiled.returncode == 0
            configuration, environment = manager.candidate(True, 0, 15, q8_experts=True, q8_clamp=True)
            result['runtime_env'] = configuration['runtime_env']
            for mode in ('pinned', 'private'):
                guard.assert_idle()
                completed = subprocess.run(['taskset', '-c', '0-127', str(binary), mode, str(PINNED)],
                                           env=environment, cwd=BASE, capture_output=True, text=True, timeout=30)
                (OUT / f'{mode}.log').write_text(completed.stdout + completed.stderr)
                assert completed.returncode == 0
                maps = sorted({line.split()[-1] for line in completed.stdout.splitlines() if line.startswith('CPU_MAP ')})
                symbols = [line.removeprefix('CPU_SYMBOL ') for line in completed.stdout.splitlines() if line.startswith('CPU_SYMBOL ')]
                devices = [line.removeprefix('DEVICE ') for line in completed.stdout.splitlines() if line.startswith('DEVICE ')]
                assert len(symbols) == 1 and Path(symbols[0]).resolve() == cpu.resolve()
                assert all(f'CPU-NUMA{i}' in devices for i in range(4)), devices
                result['runs'].append(dict(mode=mode, cpu_maps=maps, cpu_symbol=symbols[0], devices=devices))
                save()
                print(json.dumps(result['runs'][-1]), flush=True)
            assert set(result['runs'][0]['cpu_maps']) == {str(cpu.resolve()), str((PINNED / cpu.name).resolve())}
            assert result['runs'][1]['cpu_maps'] == [str(cpu.resolve())]
            assert all(sha256(path) == digest for path, digest in sources.items())
            assert sha256(cpu) == result['cpu_sha256'] and sha256(server) == result['server_sha256']
            guard.assert_idle()
            result['passed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
