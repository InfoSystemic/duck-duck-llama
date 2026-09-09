#!/usr/bin/env python3
"""Validate block scheduling through dense, expert, and four-NUMA graphs."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from glm_flash_q8_trial import memory_status
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
OUT = BASE / 'results/qwen-quantize-blocks-validation-0909'

AUDIT = r'''
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("QUANT_CPU_LIBRARY %s\n", runtime.dli_fname);
    std::atexit([] {
        using counter_fn = uint64_t (*)(int);
        const auto counter = reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_quantize_blocks_count"));
        std::printf("QUANT_BLOCK_CALLS %llu %llu %llu\n", (unsigned long long) (counter ? counter(0) : 0),
                    (unsigned long long) (counter ? counter(1) : 0), (unsigned long long) (counter ? counter(2) : 0));
    });
'''


def graph_fixture(source):
    marker = 'int main(int argc, char ** argv) {\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + AUDIT + '''
    const char * output_path = std::getenv("REPACK_TEST_OUTPUT_PATH");
    FILE * output_file = output_path ? std::fopen(output_path, "wb") : nullptr;
    if (!output_file) return 2;
''')
    source, count = re.subn(r': q8 \? std::vector<std::pair<int, int>>\{\{[^\n]+', ': q8 ? std::vector<std::pair<int, int>>{{256, 64}, {2560, 160}, {10240, 64}}', source)
    assert count == 1
    source, count = re.subn(r'    const std::vector<int> token_counts =.*?;\n', '    const std::vector<int> token_counts = {1, 3, 5, 9};\n', source, flags=re.S)
    assert count == 1
    source, count = re.subn(r'    std::vector<int> thread_counts = [^\n]+', '    std::vector<int> thread_counts = {1, 4, 15};', source)
    assert count == 1
    marker = '        auto got = run(true, k, rows, tokens, threads, moe, fused, type);\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '        if (std::fwrite(got.values.data(), sizeof(float), got.values.size(), output_file) != got.values.size()) std::abort();\n')
    marker = '    return failures ? 1 : 0;'
    assert source.count(marker) == 1
    return source.replace(marker, '    if (std::fclose(output_file)) return 2;\n' + marker)


def equal_files(left, right):
    assert left.stat().st_size == right.stat().st_size
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            chunk = a.read(1024 * 1024)
            assert chunk == b.read(1024 * 1024), (left, right)
            if not chunk:
                return


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    build_path = BASE / 'results/qwen-quantize-blocks-build-0909/result.json'
    manifest_path = build_path.parent / 'private-cpu/manifest.json'
    build, manifest = [json.loads(path.read_text()) for path in (build_path, manifest_path)]
    assert build['passed'] and build['build_completed'] and build['baseline_object_identical'] and build['baseline_link_identical']
    assert all(sha256(path) == value for path, value in build['input_sha256'].items())
    assert all(sha256(path) == value for path, value in manifest['private_source_sha256'].items())
    library = Path(manifest['library'])
    assert sha256(library) == manifest['library_sha256'] == build['library_sha256']
    parent = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    assert sha256(parent) == build['parent_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    llama = BASE / 'results/qwen-hc-norm-flat-build-0908/private-llama/libllama.so.0.3.0'
    assert sha256(llama) == 'd0b2321eae443255dd84a5ad98cda8d3d9bf5540a31891bdc5c03be91061d647'
    preset_path = BASE / 'qwen-flash-20tps.json'
    sources = dict(q6=BASE / 'results/qwen-q6-simple-barrier-0907/q6-check.cpp',
                   q8=BASE / 'results/qwen-q6-q8-wide-batch-0907/graph-check.cpp',
                   numa=BASE / 'results/qwen-q6-512-expert-validation-0907/numeric.cpp')
    inputs = [Path(__file__).resolve(), build_path, manifest_path, library, parent, llama, preset_path,
              *sources.values(), BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py', BASE / 'glm_flash_q8_trial.py']
    result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, steps=[], checks=[], numa_checks=[],
                  input_sha256={str(path):sha256(path) for path in inputs}, cpu_sha256=sha256(library),
                  fixture_sha256={}, model_loaded=False,
                  scope='Integrated graph correctness and opt-in path counters. Full output files are compared byte for byte; timings are not model-performance evidence.')
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_', 'COLD_GRAPH_'))}
    environment.update(json.loads(preset_path.read_text())['runtime_env'])
    environment.update(GGML_CPU_X16_QUANTIZE_BLOCKS_AUDIT='1')
    owned = None

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def interrupt(signum, frame):
        raise InterruptedError('Stop only owned graph validation')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)

    def run(command, label, env=None, input_text=None):
        nonlocal owned
        guard.assert_idle()
        assert memory_status()['MemAvailable'] > 32 << 30
        with (OUT / (label + '.log')).open('w') as log:
            owned = subprocess.Popen(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE if input_text else subprocess.DEVNULL, start_new_session=True)
            result['owned_component'] = dict(pid=owned.pid, label=label)
            save()
            if input_text:
                owned.stdin.write(input_text.encode())
                owned.stdin.close()
            deadline = time.monotonic() + 600
            while owned.poll() is None:
                guard.assert_idle()
                assert time.monotonic() < deadline, label
                time.sleep(.25)
        result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode))
        save()
        assert owned.returncode == 0, (label, owned.returncode)
        print(json.dumps(dict(completed=label)), flush=True)
        return (OUT / (label + '.log')).read_text()

    def compile_source(source, name, flags=()):
        result['fixture_sha256'][str(source)] = sha256(source)
        command = ['g++', '-O3', '-std=c++17', '-march=native', '-fopenmp', *flags]
        command += ['-I' + str(ENGINE / path) for path in ('include', 'src', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += [str(source), '-L' + str(PINNED), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread', '-o', str(OUT / name)]
        run(command, 'compile-' + name, environment)

    def check_runtime(log, expected, enabled):
        loaded, = re.findall(r'^QUANT_CPU_LIBRARY (.+)$', log, re.M)
        assert Path(loaded).resolve() == expected.resolve()
        counters, = re.findall(r'^QUANT_BLOCK_CALLS (\d+) (\d+) (\d+)$', log, re.M)
        counters = list(map(int, counters))
        if not enabled:
            assert counters == [0, 0, 0]
        return counters

    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        save()
        try:
            result['idle_gate'] = guard.wait_idle(OUT / 'idle.json', quiet_seconds=15)
            for kind in ('q6', 'q8'):
                source = OUT / (kind + '-check.cpp')
                source.write_text(graph_fixture(sources[kind].read_text()))
                compile_source(source, kind + '-check')
                graph_env = {k:v for k,v in environment.items() if not k.startswith('GGML_CPU_NUMA_')}
                graph_env.update(REPACK_TEST_REPEATS='1', REPACK_TEST_DOWN='1')
                for padded in (False, True):
                    reference = None
                    for arm, expected, enabled in (('parent', parent, False), ('off', library, False), ('on', library, True)):
                        label = f'{kind}-{arm}-padded{int(padded)}'
                        output = OUT / (label + '.f32')
                        trial = dict(graph_env, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(PINNED),
                                     GGML_CPU_X16_QUANTIZE_BLOCKS=str(int(enabled)), REPACK_TEST_OUTPUT_PATH=str(output))
                        if padded:
                            trial['REPACK_TEST_PADDED'] = '1'
                        log = run(['taskset', '-c', '48-62', str(OUT / (kind + '-check')), 'q8'], label, trial)
                        counts = check_runtime(log, expected, enabled)
                        assert len(re.findall(r'^PASS ', log, re.M)) == 144
                        if enabled:
                            assert min(counts) > 0, (kind, counts)
                        if reference is None:
                            reference = output
                        else:
                            equal_files(reference, output)
                        result['checks'].append(dict(label=label, kind=kind, padded=padded, enabled=enabled,
                                                     cases=144, output_sha256=sha256(output), bytes=output.stat().st_size,
                                                     full_output_matches_parent=True, path_counts=counts))
                        save()
            numa = sources['numa'].read_text()
            marker = 'int main(int argc, char ** argv) {'
            assert numa.count(marker) == 1
            numa = numa.replace(marker, marker + AUDIT)
            old = '(u + 10 * t + (m % 2 ? 490 : 250) + probe) % experts'
            assert numa.count(old) == 1
            numa = numa.replace(old, '(u + 2 * t + (m % 2 ? 508 : 250) + probe) % experts')
            source = OUT / 'numa-check.cpp'
            source.write_text(numa)
            for kind in ('q6', 'q8'):
                compile_source(source, 'numa-' + kind, ['-DQWEN_MOE_CHECK', '-DQWEN_' + kind.upper() + '_CHECK'])
                cases = [(1, False), (3, False), (5, False), (64, False), (5, True)] if kind == 'q6' else [(5, False), (5, True)]
                for tokens, unfused in cases:
                    reference = None
                    for enabled in (False, True):
                        expected = library if enabled else parent
                        label = f'numa-{kind}-t{tokens}-unfused{int(unfused)}-on{int(enabled)}'
                        output = OUT / (label + '.f32')
                        trial = dict(environment, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(llama.parent) + ':' + str(PINNED),
                                     GGML_CPU_X16_QUANTIZE_BLOCKS=str(int(enabled)), GGML_CPU_DISABLE_FUSION=str(int(unfused)),
                                     GGML_CPU_MOE_GATE_UP_FUSION=str(int(not unfused)), GGML_Q4E_EXPERT_EVEN_SPLIT='1',
                                     COLD_GRAPH_K_PER_SOCKET='2560', COLD_GRAPH_ROWS='640', COLD_GRAPH_MATRICES='4',
                                     COLD_GRAPH_TOKENS=str(tokens), COLD_GRAPH_OUTPUT_PATH=str(output))
                        log = run(['taskset', '-c', '0-127', str(OUT / ('numa-' + kind)), '32', '15', 'fixture', '3'], label, trial, 'exit\n')
                        counts = check_runtime(log, expected, enabled)
                        ready, = [json.loads(line) for line in log.splitlines() if line.startswith('{') and json.loads(line).get('event') == 'ready']
                        assert ready['experts'] == 512 and ready['down_checked'] and ready['weight_type'] == {'q6':'q6_K', 'q8':'q8_0'}[kind]
                        assert Path(ready['policy_library']).resolve() == llama.resolve()
                        assert output.stat().st_size == 3 * 4 * 10 * tokens * 2560 * 4
                        if enabled and kind == 'q6':
                            assert counts[1 if unfused else 2] > 0, (label, counts)
                        if reference is None:
                            reference = output
                        else:
                            equal_files(reference, output)
                        result['numa_checks'].append(dict(label=label, kind=kind, tokens=tokens, unfused=unfused,
                                                          enabled=enabled, output_sha256=sha256(output), bytes=output.stat().st_size,
                                                          full_output_matches_parent=True, path_counts=counts, reference=ready))
                        save()
            assert all(sha256(path) == value for path, value in {**result['input_sha256'], **result['fixture_sha256']}.items())
            result.update(passed=True, bit_exact=True, graph_case_executions=sum(row['cases'] for row in result['checks']),
                          binary_sha256={name:sha256(OUT / name) for name in ('q6-check', 'q8-check', 'numa-q6', 'numa-q8')})
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
            print(json.dumps(dict(passed=result['passed'], error=result.get('error'))), flush=True)


if __name__ == '__main__':
    main()
