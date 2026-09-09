#!/usr/bin/env python3
"""Port the exact dense Q8 batching kernel to a private Flash CPU library."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'
OUT = BASE / 'results/glm-flash-q8-batch-0908'
PRIVATE = OUT / 'private-cpu'


def main():
    OUT.mkdir(exist_ok=False)
    PRIVATE.mkdir()
    state = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
    pid = state['current']['pid']
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.assert_idle()
    cpu_source = ENGINE / 'ggml/src/ggml-cpu/repack.cpp'
    arch_source = ENGINE / 'ggml/src/ggml-cpu/arch/x86/repack.cpp'
    header = BASE / 'qwen-q8-batch.h'
    micro_path = BASE / 'results/qwen-q8-chains-batch-shapes-0907/result.json'
    micro = json.loads(micro_path.read_text())
    assert micro['passed'] and micro['rows'][0]['bit_exact_values'] == 168480
    assert sha256(header) == micro['batch_header_sha256']
    assert sha256(arch_source) == micro['source_sha256']
    text = cpu_source.read_text()
    marker = '                for (int64_t i11 = 0; i11 < ne11;) {'
    assert text.count(marker) == 1
    changed = text.replace(marker, '''                static const bool q8_batch = [] {
                    const char * value = getenv("GGML_CPU_X16_Q8_BATCH");
                    return value && atoi(value) == 1;
                }();
''' + marker)
    old = 'const int nr = triple ? 3 : (pair_enabled || compact_batch) && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;'
    assert changed.count(old) == 1
    changed = changed.replace(old,
        'const int nr = q8_batch && sp.dst_type == GGML_TYPE_Q8_0 ? (int) std::min<int64_t>(4, ne11 - i11) : triple ? 3 : (pair_enabled || compact_batch) && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;')
    old = '        std::vector<float> gtmp(chunk), utmp(chunk);'
    assert changed.count(old) == 1
    changed = changed.replace(old, '''        static const bool q8_batch = [] {
            const char * value = getenv("GGML_CPU_X16_Q8_BATCH");
            return value && atoi(value) == 1;
        }();
        const int max_batch = q8_batch && sp.dst_type == GGML_TYPE_Q8_0 ? 4 : 1;
        std::vector<float> gtmp(chunk * max_batch), utmp(chunk * max_batch);''')
    old = '''            for (int64_t i = 0; i < n1; i++) {
                float * out = (float *) ((char *) dst->data + i * dst->nb[1]) + r0;
                sp.gemv((int) k, gtmp.data(), 0, gw, wdata + (size_t) i * row_bytes, 1, (int) (r1 - r0));
                sp.gemv((int) k, utmp.data(), 0, uw, wdata + (size_t) i * row_bytes, 1, (int) (r1 - r0));
                ggml_vec_swiglu_f32(r1 - r0, out, gtmp.data(), utmp.data());
            }'''
    assert changed.count(old) == 1
    changed = changed.replace(old, '''            for (int64_t i = 0; i < n1;) {
                const int nr = (int) std::min<int64_t>(max_batch, n1 - i);
                sp.gemv((int) k, gtmp.data(), chunk, gw, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                sp.gemv((int) k, utmp.data(), chunk, uw, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                for (int t = 0; t < nr; ++t) {
                    float * out = (float *) ((char *) dst->data + (i + t) * dst->nb[1]) + r0;
                    ggml_vec_swiglu_f32(r1 - r0, out, gtmp.data() + t * chunk, utmp.data() + t * chunk);
                }
                i += nr;
            }''')
    arch_text = arch_source.read_text()
    signature = 'void ggml_gemv_q8_0_x16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {'
    avx = '#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)'
    assert arch_text.count(signature) == 1
    arch_changed = arch_text.replace(signature, avx + '\n#include "qwen-q8-batch.h"\n#endif\n\n' + signature)
    entry = signature + '\n' + avx + '\n'
    assert arch_changed.count(entry) == 1
    arch_changed = arch_changed.replace(entry, entry + '    if (nr > 1) return qwen_q8_batch_dispatch(n, s, bs, vx, vy, nr, nc);\n')
    assert arch_changed.count('#include "../../repack.h"') == 1
    arch_changed = arch_changed.replace('#include "../../repack.h"', '#include "' + str(cpu_source.with_name('repack.h')) + '"')
    private_main, private_arch = PRIVATE / 'repack.cpp', PRIVATE / 'repack-x86.cpp'
    private_main.write_text(changed)
    private_arch.write_text(arch_changed)
    (PRIVATE / header.name).write_bytes(header.read_bytes())
    patch = ''.join(difflib.unified_diff(text.splitlines(True), changed.splitlines(True), fromfile=str(cpu_source), tofile=str(private_main)))
    patch += ''.join(difflib.unified_diff(arch_text.splitlines(True), arch_changed.splitlines(True), fromfile=str(arch_source), tofile=str(private_arch)))
    (PRIVATE / 'q8-batch.patch').write_text(patch)
    commands_path = ENGINE / 'build-goal/compile_commands.json'
    commands = json.loads(commands_path.read_text())
    compile_commands, replacements = [], {}
    for source, target in ((cpu_source, private_main), (arch_source, private_arch)):
        entry, = [e for e in commands if e['file'] == str(source)]
        command = shlex.split(entry['command'])
        old_object = str((Path(entry['directory']) / command[command.index('-o') + 1]).resolve())
        new_object = target.with_suffix('.cpp.o')
        command[command.index('-o') + 1] = str(new_object)
        command[command.index(str(source))] = str(target)
        command.insert(1, '-I' + str(source.parent))
        compile_commands.append(command)
        replacements[old_object] = str(new_object)
    link_dir = ENGINE / 'build-goal/ggml/src'
    link_path = link_dir / 'CMakeFiles/ggml-cpu.dir/link.txt'
    link = shlex.split(link_path.read_text())
    link = [str((link_dir / x).resolve()) if not x.startswith('-') and x.endswith(('.o', '.so.0.22.0')) else x for x in link]
    assert all(x in link for x in replacements)
    original_library = ENGINE / 'build-goal/bin/libggml-cpu.so.0.22.0'
    reference_library = PINNED / 'libggml-cpu.so.0.22.0'
    assert sha256(reference_library) == '4254ca6c9537ba258eb8d6ae1c025d43741bad71c82de3a69c4a06257c558fbe'
    fixture_source = BASE / 'iq2-repack-check.cpp'
    original_paths = {str(p) for p in (cpu_source, arch_source, header, micro_path, commands_path,
                                      link_path, original_library, reference_library, fixture_source, Path(__file__))}
    original_paths.update(x for x in link if x.endswith(('.o', '.so.0.22.0', '.a')))
    original_paths.remove(link[link.index('-o') + 1])
    original_paths.add(str(original_library))
    inputs = {p: sha256(p) for p in original_paths}
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    result = dict(started=time.time(), passed=False, steps=[], checks=[],
                  parent_cpu_sha256=sha256(original_library), reference_cpu_sha256=sha256(reference_library),
                  preserved_kernel_check_values=168480)

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(command, label, environment=None):
        guard.assert_idle()
        with (OUT / (label + '.log')).open('w') as stream:
            process = subprocess.Popen(command, cwd=link_dir, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 300
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert process.returncode == 0, (label, process.returncode)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result['steps'].append(dict(label=label, command=command))
        save()
        print(json.dumps({'completed': label}), flush=True)
        return (OUT / (label + '.log')).read_text()

    save()
    try:
        baseline_link = list(link)
        baseline_link[baseline_link.index('-o') + 1] = str(PRIVATE / 'baseline.so')
        run(baseline_link, 'baseline-link')
        assert sha256(PRIVATE / 'baseline.so') == sha256(original_library), 'Link inputs do not reproduce the existing build'
        result['baseline_link_identical'] = True
        for index, command in enumerate(compile_commands):
            run(command, f'compile-{index}')
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        candidate_link = [replacements.get(x, x) for x in link]
        candidate_link[candidate_link.index('-o') + 1] = str(library)
        run(candidate_link, 'candidate-link')
        (PRIVATE / 'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(library=str(library), library_sha256=sha256(library), input_sha256=inputs,
                        parent_cpu_sha256=sha256(original_library), reference_cpu_sha256=sha256(reference_library),
                        compile_commands=compile_commands, link_command=candidate_link,
                        private_source_sha256={str(p): sha256(p) for p in (private_main, private_arch, PRIVATE / header.name)},
                        scope='Exact Q8 dense and fused gate/up batching through four activation rows; no tensor requantization.')
        (PRIVATE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result.update(library=str(library), library_sha256=sha256(library))
        fixture = fixture_source.read_text().replace('ggml_set_name(w, "blk.0.ffn_gate_exps.weight");',
                                                     'ggml_set_name(w, "blk.0.attn_output.weight");')
        fixture = fixture.replace('ggml_set_name(u, "blk.0.ffn_up_exps.weight");',
                                  'ggml_set_name(u, "blk.0.attn_up.weight");')
        fixture = fixture.replace('q8 ? std::vector<std::pair<int, int>>{{320, 512}, {10240, 320}}',
            'q8 ? std::vector<std::pair<int, int>>{{256, 512}, {512, 256}, {1536, 384}, {2048, 512}, {4096, 512}, {16384, 64}}')
        fixture = fixture.replace('std::vector<int>{1, 2, 3, 4, 9}', 'std::vector<int>{1, 2, 3, 4, 5, 9}')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
        env.update(GGML_CPU_X16_Q8_0='1', GGML_CPU_X16_Q4_K='1', GGML_CPU_X16_CHUNK_MAX='16',
                   REPACK_TEST_DENSE_WORK_SHARING='1', REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
        flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I' + str(ENGINE / p) for p in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        links = ['-L' + str(PINNED), '-Wl,-rpath,' + str(PINNED), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl']
        for kind in ('q8', 'q4'):
            current_fixture = fixture
            expected = 216
            if kind == 'q4':
                current_fixture = current_fixture.replace('GGML_TYPE_Q5_K', 'GGML_TYPE_Q4_K').replace('quantize_row_q5_K_ref', 'quantize_row_q4_K_ref').replace('block_q5_K', 'block_q4_K')
                current_fixture = current_fixture.replace('q8 ? std::vector<ggml_type>{GGML_TYPE_Q8_0}', 'q8 ? std::vector<ggml_type>{GGML_TYPE_Q4_K}')
                current_fixture = current_fixture.replace('{{256, 512}, {512, 256}, {1536, 384}, {2048, 512}, {4096, 512}, {16384, 64}}', '{{4096, 64}, {4096, 128}}')
                current_fixture = current_fixture.replace('std::vector<int>{1, 2, 3, 4, 5, 9}', 'std::vector<int>{1, 3}')
                env['REPACK_TEST_THREADS'] = '4'
                expected = 8
            source, binary = OUT / (kind + '-check.cpp'), OUT / (kind + '-check')
            source.write_text(current_fixture)
            run(flags + [str(source)] + links + ['-o', str(binary)], kind + '-compile')
            for padded in (False, True):
                env.pop('REPACK_TEST_PADDED', None)
                if padded:
                    env['REPACK_TEST_PADDED'] = '1'
                hashes = []
                modes = ('reference', 'off', 'on') if kind == 'q8' else ('reference', 'on')
                for mode in modes:
                    env['LD_LIBRARY_PATH'] = str(PINNED) if mode == 'reference' else str(PRIVATE) + ':' + str(PINNED)
                    env['GGML_CPU_X16_Q8_BATCH'] = '1' if mode == 'on' else '0'
                    log = run(['taskset', '-c', '48-62', str(binary), 'q8'], f'{kind}-{int(padded)}-{mode}', env)
                    rows = re.findall(r'^PASS .* hash=([^\n]+)', log, re.M)
                    assert len(rows) == expected, (kind, mode, len(rows))
                    hashes.append(rows)
                assert all(h == hashes[0] for h in hashes), (kind, padded, 'Output bits changed')
                result['checks'].append(dict(type=kind, padded=padded, bit_exact_cases=expected, compared_modes=modes))
                save()
        assert all(sha256(p) == value for p, value in inputs.items()), 'An original build input changed'
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
