#!/usr/bin/env python3
"""Build an opt-in exact Q8 activation-sum path on Qwen's selected CPU library."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
OUT = BASE / 'results/qwen-q8-sums-build-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    proof_path = BASE / 'results/qwen-q8-sums-proof-0909/result.json'
    parent, proof = [json.loads(path.read_text()) for path in (parent_path, proof_path)]
    assert sha256(parent['library']) == parent['library_sha256'] == proof['cpu_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert proof['passed'] and proof['correctness']['matrix_cases'] == 240
    assert all(sha256(path) == value for record in (parent, proof) for path, value in record['input_sha256'].items())
    assert all(sha256(path) == value for path, value in parent['private_source_sha256'].items())
    command = list(parent['compile_commands'][1])
    source_path = Path(command[-1])
    original_object = command[command.index('-o') + 1]
    assert original_object in parent['link_command']
    original = source_path.read_text()
    marker = 'void ggml_gemv_q8_0_x16_q8_0('
    assert original.count(marker) == 1
    changed = original.replace(marker, '#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)\n#include "qwen-q8-sums.h"\n#endif\n\n' + marker)
    marker = '    GGML_ASSERT(nr == 1 && n % QK8_0 == 0 && nc % 16 == 0);\n'
    assert changed.count(marker) == 1
    changed = changed.replace(marker, marker + '''    static const bool fast_sum = [] {
        const char * value = std::getenv("GGML_CPU_Q8_FAST_SUM");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (fast_sum) return qwen_q8_sum_candidate<2>(n, s, bs, vx, vy, nr, nc);
''')
    header_path = proof_path.parent / 'q8-sums-candidates.h'
    header = header_path.read_text().replace('#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))\n', '')
    header = header.replace('#undef GGML_X16_BC32\n', '')
    header = header.replace('candidate_q8', 'qwen_q8_sum_candidate').replace('q8_activation_sum', 'qwen_q8_activation_sum')
    assert 'GGML_X16_BC32' in header and '#define GGML_X16_BC32' not in header
    graph_source = source_path.parent.parent / 'graph-check.cpp'
    graph_text = graph_source.read_text()
    marker = 'int main(int argc, char ** argv) {\n'
    assert graph_text.count(marker) == 1
    graph_text = graph_text.replace(marker, marker + '''    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\\n", runtime.dli_fname);
''')
    fixture = BASE / 'flash-q8-sums-check-0908.cpp'
    direct_source = fixture.read_text()
    marker = '    for (int k : {128, 512, 2048, 4096, 16384})'
    assert direct_source.count(marker) == 1
    direct_source = direct_source.replace(marker, '''    if (std::getenv("Q8_SUMS_CHECK_ONLY")) {
        std::printf("{\\"event\\":\\"done\\",\\"passed\\":true}\\n");
        return 0;
    }
''' + marker)
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None

    def interrupt(signum, frame):
        raise InterruptedError('Stop only this owned private Q8 build/check')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-cpu'
        private.mkdir()
        (private / 'repack-x86.cpp').write_text(changed)
        (private / 'qwen-q8-sums.h').write_text(header)
        (private / 'qwen-q8-batch.h').write_bytes((source_path.parent / 'qwen-q8-batch.h').read_bytes())
        (private / 'q8-sums.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile=str(source_path), tofile='private/repack-x86.cpp')))
        (OUT / 'direct-check.cpp').write_text(direct_source)
        (OUT / 'q8-sums-candidates.h').write_bytes(header_path.read_bytes())
        (OUT / 'graph-check.cpp').write_text(graph_text)
        inputs = {str(path):sha256(path) for path in (Path(__file__).resolve(), parent_path, proof_path,
                  source_path, Path(original_object), header_path, graph_source, fixture, Path(parent['library']),
                  BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py')}
        inputs.update(parent['input_sha256'])
        inputs.update(parent['private_source_sha256'])
        inputs.update(proof['input_sha256'])
        source_files = [private / 'repack-x86.cpp', private / 'qwen-q8-sums.h', private / 'qwen-q8-batch.h',
                        OUT / 'direct-check.cpp', OUT / 'q8-sums-candidates.h', OUT / 'graph-check.cpp']
        source_hashes = {str(path):sha256(path) for path in source_files}
        result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, build_completed=False,
                      input_sha256=inputs, private_source_sha256=source_hashes, steps=[], checks=[],
                      parent_sha256=parent['library_sha256'], model_loaded=False,
                      scope='Only the x86 repack object changes. GGML_CPU_Q8_FAST_SUM=1 selects exact SAD sums for NR=1, with inline preparation for NC=16. Wider activation batches and quantized layouts stay unchanged.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(args, label, env=None):
            nonlocal owned
            guard.assert_idle()
            with (OUT / (label + '.log')).open('w') as log:
                owned = subprocess.Popen(args, cwd=ENGINE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, label=label)
                save()
                deadline = time.monotonic() + 600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=args, exit_code=owned.returncode))
            save()
            assert owned.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)
            return (OUT / (label + '.log')).read_text()

        save()
        try:
            baseline = OUT / 'baseline-repack-x86.cpp.o'
            baseline_command = list(command)
            baseline_command[baseline_command.index('-o') + 1] = str(baseline)
            run(baseline_command, 'baseline-compile')
            result['baseline_object_identical'] = sha256(baseline) == sha256(original_object)
            assert result['baseline_object_identical']
            link = [str(baseline) if value == original_object else value for value in parent['link_command']]
            link[link.index('-o') + 1] = str(OUT / 'parent-link.so')
            run(link, 'parent-link')
            result['baseline_link_identical'] = sha256(OUT / 'parent-link.so') == parent['library_sha256']
            assert result['baseline_link_identical']
            command[-1] = str(private / 'repack-x86.cpp')
            command[command.index('-o') + 1] = str(private / 'repack-x86.cpp.o')
            run(command, 'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [command[command.index('-o') + 1] if value == original_object else value for value in parent['link_command']]
            link[link.index('-o') + 1] = str(library)
            run(link, 'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            result.update(build_completed=True, library=str(library), library_sha256=sha256(library))
            environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
            environment.update(LD_LIBRARY_PATH=str(private) + ':' + str(PINNED), GGML_CPU_X16_Q8_0='1',
                               GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096', Q8_SUMS_CHECK_ONLY='1')
            direct = OUT / 'direct-check'
            direct_command = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-I' + str(OUT)]
            direct_command += ['-I' + str(ENGINE / path) for path in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            direct_command += [str(OUT / 'direct-check.cpp'), str(library), str((PINNED / 'libggml-base.so.0').resolve()), '-ldl', '-pthread', '-o', str(direct)]
            run(direct_command, 'direct-compile')
            for enabled in ('0', '1'):
                output = OUT / ('direct-' + enabled + '.bin')
                log = run(['taskset', '-c', '48', str(direct), str(output)], 'direct-' + enabled, dict(environment, GGML_CPU_Q8_FAST_SUM=enabled))
                events = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
                assert Path(events[0]['path']).resolve() == library.resolve()
                assert events[-1] == dict(event='done', passed=True)
                assert events[1] == proof['correctness']
                assert sha256(output) == proof['output_sha256']
                result['checks'].append(dict(kind='direct', enabled=enabled, cases=240, compared_values=57600,
                                             output_sha256=sha256(output), cpu_library=str(library)))
                save()
            graph = OUT / 'graph-check'
            graph_command = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
            graph_command += ['-I' + str(ENGINE / path) for path in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            graph_command += [str(OUT / 'graph-check.cpp'), '-L' + str(private), '-L' + str(PINNED),
                              '-Wl,-rpath,' + environment['LD_LIBRARY_PATH'], '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(graph)]
            run(graph_command, 'graph-compile')
            environment.update(REPACK_TEST_DENSE_WORK_SHARING='1', REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
            for padded in (False, True):
                reference = graph_source.parent / ('graph-' + ('padded' if padded else 'standard') + '-batch1.log')
                expected = [line.split(' hash=')[-1] for line in reference.read_text().splitlines() if line.startswith('PASS ')]
                assert len(expected) == 324
                inputs[str(reference)] = sha256(reference)
                for chunk in (16, 64):
                    for enabled in ('0', '1'):
                        label = f'graph-pad{int(padded)}-chunk{chunk}-sum{enabled}'
                        env = dict(environment, GGML_CPU_X16_CHUNK_MAX=str(chunk), GGML_CPU_Q8_FAST_SUM=enabled)
                        if padded:
                            env['REPACK_TEST_PADDED'] = '1'
                        log = run(['taskset', '-c', '48-62', str(graph), 'q8'], label, env)
                        loaded, = re.findall(r'^CPU_LIBRARY (.+)$', log, re.M)
                        assert Path(loaded).resolve() == library.resolve()
                        actual = [line.split(' hash=')[-1] for line in log.splitlines() if line.startswith('PASS ')]
                        assert actual == expected, label
                        result['checks'].append(dict(kind='graph', enabled=enabled, padded=padded, chunk=chunk,
                                                     cases=len(actual), all_hashes_match=True, reference=str(reference)))
                        save()
            assert all(sha256(path) == value for path, value in {**inputs, **source_hashes}.items())
            result.update(passed=True, graph_case_executions=sum(row['cases'] for row in result['checks'] if row['kind'] == 'graph'))
            manifest = dict(input_sha256=inputs, library=str(library), library_sha256=sha256(library),
                            parent_manifest=str(parent_path), parent_sha256=parent['library_sha256'],
                            compile_command=command, link_command=link, private_source_sha256=source_hashes,
                            baseline_object_identical=True, baseline_link_identical=True, scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(dict(passed=True, library_sha256=sha256(library), graph_cases=result['graph_case_executions'])), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=10)
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            result['peer_after'] = read_service(18095)
            save()


if __name__ == '__main__':
    main()
