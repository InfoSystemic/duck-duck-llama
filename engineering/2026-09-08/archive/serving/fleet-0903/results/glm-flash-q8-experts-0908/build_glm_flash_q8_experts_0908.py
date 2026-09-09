#!/usr/bin/env python3
"""Compare a private x16 Q8 expert selector on Flash's 288-expert geometry."""
import array
import difflib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from build_qwen_q6_expert_capacity import transform
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'
OUT = BASE / 'results/glm-flash-q8-experts-0908'
PRIVATE = OUT / 'private-cpu'


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    OUT.mkdir()
    PRIVATE.mkdir()
    state = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
    pid = state['current']['pid']
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.assert_idle()
    parent_path = BASE / 'results/glm-flash-q8-batch-0908/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    parent_library = Path(parent['library'])
    assert sha256(parent_library) == parent['library_sha256'] == '3601757e1c77e92dea67e5cc28ab242a69de5f106c5fac34b210ab7e4f7cb3f8'
    for path, digest in parent['private_source_sha256'].items():
        assert sha256(path) == digest, path
    compile_command = list(parent['compile_commands'][0])
    source_path = Path(compile_command[-1])
    source = source_path.read_text()
    changed = transform(source)
    marker = '    const bool is_attn  = nd == 2 && strstr(cur->name, ".attn_") != nullptr;'
    changed = replace_once(changed, marker, '''    static const bool q8_experts = [] {
        const char * value = getenv("GGML_CPU_X16_Q8_EXPERTS");
        return value && atoi(value) == 1;
    }();
    if (q8_experts && nd == 3 && cur->type == GGML_TYPE_Q8_0 && cur->ne[2] <= 512 &&
            strstr(cur->name, "_exps.weight") != nullptr && cur->ne[0] % QK8_0 == 0) {
        return &t_q8_0;
    }
''' + marker)
    private_source = PRIVATE / 'repack.cpp'
    private_source.write_text(changed)
    (PRIVATE / 'q8-experts.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile=str(source_path), tofile=str(private_source))))
    old_object = compile_command[compile_command.index('-o') + 1]
    new_object = str(PRIVATE / 'repack.cpp.o')
    compile_command[compile_command.index('-o') + 1] = new_object
    compile_command[-1] = str(private_source)
    parent_link = list(parent['link_command'])
    link_dir = ENGINE / 'build-goal/ggml/src'
    inputs = {str(parent_path), str(parent_library), str(source_path), str(BASE / 'iq2-repack-check.cpp'),
              str(BASE / 'build_qwen_q6_expert_capacity.py'), str(Path(__file__))}
    inputs.update(x for x in parent_link if x.endswith(('.o', '.so.0.22.0', '.a')))
    inputs.update(parent['private_source_sha256'])
    original_hashes = {p: sha256(p) for p in inputs}
    result = dict(started=time.time(), passed=False, parent_sha256=sha256(parent_library),
                  experts=288, capacity=512, steps=[], runs=[], comparisons=[], input_sha256=original_hashes,
                  scope='Component correctness and timing on Flash Q8 expert shapes; no whole-model bandwidth claim.')
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(command, label, environment=None):
        guard.assert_idle()
        path = OUT / (label + '.log')
        with path.open('w') as log:
            proc = subprocess.Popen(command, cwd=link_dir, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 600
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.5)
                assert proc.returncode == 0, (label, proc.returncode, str(path))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
        result['steps'].append(dict(label=label, command=command, log=str(path)))
        save()
        print(json.dumps(dict(completed=label)), flush=True)
        return path.read_text()

    try:
        baseline_link = list(parent_link)
        baseline_link[baseline_link.index('-o') + 1] = str(PRIVATE / 'baseline.so')
        run(baseline_link, 'baseline-link')
        assert sha256(PRIVATE / 'baseline.so') == sha256(parent_library)
        result['baseline_link_identical'] = True
        run(compile_command, 'compile')
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        link = [new_object if x == old_object else x for x in parent_link]
        link[link.index('-o') + 1] = str(library)
        run(link, 'link')
        (PRIVATE / 'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(library=str(library), library_sha256=sha256(library),
                        parent_manifest=str(parent_path), parent_sha256=sha256(parent_library),
                        input_sha256=original_hashes, compile_command=compile_command, link_command=link,
                        private_source_sha256={str(private_source): sha256(private_source)},
                        experts=True, capacity=512, scope='Same-precision opt-in Q8 expert repacking; no requantization.')
        (PRIVATE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result.update(library=str(library), library_sha256=sha256(library))

        fixture = (BASE / 'iq2-repack-check.cpp').read_text()
        replacements = {
            '#include <fstream>': None,
            'const int experts = moe ? 12 : 1;': 'const int experts = moe ? 288 : 1;',
            'q8 ? std::vector<std::pair<int, int>>{{320, 512}, {10240, 320}}':
                'q8 ? std::vector<std::pair<int, int>>{{4096, 512}, {512, 4096}}',
            ': std::vector<int>{1, 4, 9};': ': std::vector<int>{1, 3, 4};',
            '        if (work_sharing && moe == dense_work_sharing) continue;':
                '        if (!moe || (shape.first == 512 && fused)) continue;',
            '(j + 3 * t) % experts': '(j + 31 * t + 250) % experts',
            'packed_ms=%.3f': 'packed_ms=%.6f',
            'native_ms=%.3f': 'native_ms=%.6f',
        }
        assert '#include <fstream>' not in fixture
        fixture = '#include <fstream>\n' + fixture
        for old, new in replacements.items():
            if new is not None:
                fixture = replace_once(fixture, old, new)
        marker = '        ++cases; failures += !okay;'
        fixture = replace_once(fixture, marker, '''        if (const char * dir = std::getenv("REPACK_TEST_OUTPUT_DIR")) {
            const std::string path = std::string(dir) + "/" + std::to_string(k) + "-" + std::to_string(rows) + "-" +
                std::to_string(tokens) + "-" + std::to_string(fused) + ".f32";
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char *>(got.values.data()), got.values.size() * sizeof(float));
            if (!output) std::abort();
        }
''' + marker)
        marker = '    ggml_backend_load_all();'
        fixture = replace_once(fixture, marker, marker + '''
    Dl_info loaded{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &loaded)) std::abort();
    std::printf("CPU_LIBRARY %s\\n", loaded.dli_fname);
''')
        fixture_path, binary = OUT / 'expert-check.cpp', OUT / 'expert-check'
        fixture_path.write_text(fixture)
        flags = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I' + str(ENGINE / p) for p in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        flags += [str(fixture_path), '-L' + str(PINNED), '-Wl,-rpath,' + str(PINNED),
                  '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
        run(flags, 'fixture-compile')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
        env.update(GGML_CPU_X16_Q8_0='1', GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_X16_CHUNK_MAX='16',
                   REPACK_TEST_DOWN='1', REPACK_TEST_THREADS='15', REPACK_TEST_REPEATS='20',
                   REPACK_TEST_TIMING_MEDIAN='1', REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1')
        for index, (mode, padded) in enumerate([('reference', False), ('off', False), ('on', False),
                                               ('on', False), ('off', False), ('reference', True), ('on', True)]):
            label = f'run-{index}-{mode}'
            values_dir = OUT / label
            values_dir.mkdir()
            env['REPACK_TEST_OUTPUT_DIR'] = str(values_dir)
            env['LD_LIBRARY_PATH'] = str(parent_library.parent if mode == 'reference' else PRIVATE) + ':' + str(PINNED)
            env['GGML_CPU_X16_Q8_EXPERTS'] = '1' if mode == 'on' else '0'
            env.pop('REPACK_TEST_PADDED', None)
            if padded:
                env['REPACK_TEST_PADDED'] = '1'
            log = run(['taskset', '-c', '48-62', str(binary), 'q8'], label, env)
            mapped, = re.findall(r'^CPU_LIBRARY (.+)$', log, re.M)
            assert Path(mapped).resolve() == (parent_library if mode == 'reference' else library).resolve()
            rows = [dict(item.split('=', 1) for item in line.split()[1:])
                    for line in log.splitlines() if line.startswith('PASS ')]
            assert len(rows) == 9, (label, len(rows))
            result['runs'].append(dict(label=label, mode=mode, padded=padded, rows=rows, values=str(values_dir)))
            save()
        references = {x['padded']: x for x in result['runs'] if x['mode'] == 'reference'}
        for item in result['runs'][1:]:
            if item['mode'] == 'reference':
                continue
            reference = references[item['padded']]
            for path in sorted(Path(item['values']).glob('*.f32')):
                baseline = Path(reference['values']) / path.name
                a, b = array.array('f'), array.array('f')
                a.frombytes(baseline.read_bytes())
                b.frombytes(path.read_bytes())
                assert len(a) == len(b)
                scaled = [abs(x - y) / (1 + abs(x)) for x, y in zip(a, b)]
                maximum = max(scaled)
                assert all(math.isfinite(x) for x in scaled) and maximum <= 2e-4, (item['label'], path.name, maximum)
                exact = baseline.read_bytes() == path.read_bytes()
                if item['mode'] == 'off' and not item['padded']:
                    assert exact, (item['label'], path.name)
                result['comparisons'].append(dict(run=item['label'], case=path.stem, values=len(a),
                                                  bit_exact=exact, max_scaled_error=maximum))
        assert all(sha256(path) == digest for path, digest in original_hashes.items())
        guard.assert_idle()
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
