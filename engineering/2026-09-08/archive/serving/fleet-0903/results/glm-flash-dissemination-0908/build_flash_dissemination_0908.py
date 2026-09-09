#!/usr/bin/env python3
"""Build and validate a private child of the retained ordered-Q8 runtime."""
import difflib
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from flash_dissemination_transform_0908 import transform, fixture_audit
from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE / 'results/glm-flash-dissemination-0908'
PRIVATE = OUT / 'private-cpu'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'


def main(build_only=False, resume=False):
    os.umask(0o077)
    manager = Manager()
    if build_only:
        current = manager.state['current']
        retained = json.loads((BASE / 'results/glm-flash-q8-r8-ordered-k-post-model-0908.json').read_text())
        assert retained['passed'] and retained['current'] == current
        guard = None
    else:
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
    def validate():
        if guard is not None:
            manager.validate_current()
            guard.assert_idle()
        else:
            # Compilation only, on the controller's reserved CPU. This checks
            # the recorded parent files and does not claim a live Flash model.
            assert Manager().state['current'] == current
            assert sha256(current['cpu_library']) == current['cpu_sha256']
    assert current['ordered_k'] and current['drafts'] == 0
    parent_path = BASE / 'results/glm-flash-q8-r8-ordered-k-0908/private-cpu/manifest.json'
    cpu_path = BASE / 'results/glm-flash-q8-pool-0908c/private-cpu/manifest.json'
    probe_path = BASE / 'results/glm-flash-local-barrier-probe-0908b/result.json'
    parent, cpu, probe = [json.loads(p.read_text()) for p in (parent_path, cpu_path, probe_path)]
    assert probe['passed'] and all(sha256(p) == h for p, h in probe['input_sha256'].items())
    assert probe['cpu_sha256'] == parent['library_sha256'] == current['cpu_sha256'] == sha256(parent['library'])
    assert all(row['change_percent'] < 0 for row in probe['comparisons'] if row['mode'] == 'dissemination')
    original_command = list(cpu['compile_commands'][0])
    source_path = Path(original_command[-1])
    old_object = original_command[original_command.index('-o') + 1]
    assert old_object in parent['link_command']
    original = source_path.read_text()
    changed = transform(original)
    source_fixtures = {
        'graph': BASE / 'results/glm-flash-q8-r8-ordered-k-0908/ordered-k-graph-check.cpp',
        'threads': BASE / 'numa-thread-limit-check.cpp',
        'reduce': BASE / 'numa-reduce-check.cpp',
    }
    paths = [Path(__file__), BASE / 'flash_dissemination_transform_0908.py',
             BASE / 'flash-local-barrier-0908.h', parent_path, cpu_path, probe_path,
             source_path, Path(parent['library']), *source_fixtures.values()]
    inputs = {str(p): sha256(p) for p in paths}
    for value in parent['link_command']:
        if value.endswith(('.o', '.a', '.so.0.22.0', '/libgomp.so')) and Path(value).exists():
            inputs[value] = sha256(value)
    for root in ('ggml/src', 'ggml/include'):
        for p in (ENGINE / root).rglob('*.h'):
            inputs[str(p)] = sha256(p)
    OUT.mkdir(exist_ok=resume)
    PRIVATE.mkdir(exist_ok=resume)
    def publish(path, data):
        if path.exists():
            assert resume and path.read_bytes() == data, str(path)
        else:
            path.write_bytes(data)
    publish(PRIVATE / 'ggml-cpu.c', changed.encode())
    publish(PRIVATE / 'flash-local-barrier-0908.h', (BASE / 'flash-local-barrier-0908.h').read_bytes())
    publish(PRIVATE / 'cpu.patch', ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True))).encode())
    for name, path in source_fixtures.items():
        output = OUT / (name + '-check.cpp')
        fixture = path.read_text()
        if name == 'graph':
            marker = '        if (std::fclose(dump)) ++failures;'
            assert fixture.count(marker) == 1
            fixture = fixture.replace(marker, '''        for (int threads : {3,7,64,65}) {
            test_case c; c.k=1024; c.rows=64; c.tokens=1; c.threads=threads;
            c.type=GGML_TYPE_F32; c.packed=false;
            failures += !run_case(c,dump,false,cases++);
        }
''' + marker)
            marker = '    bool good = true;'
            assert fixture.count(marker) == 1
            fixture = fixture.replace(marker, '''    const auto barrier_count = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT,"ggml_cpu_dissemination_count"));
    const bool barrier_audit = enabled("GGML_CPU_DISSEMINATION_AUDIT");
    uint64_t barriers_used = 0;
''' + marker)
            marker = '        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) good = false;'
            assert fixture.count(marker) == 1
            fixture = fixture.replace(marker, '''        const uint64_t before_barrier = barrier_count ? barrier_count(0) : 0;
''' + marker + '''
        const uint64_t used_barriers = barrier_count ? barrier_count(0)-before_barrier : 0;
        if (barrier_audit) {
            const bool expected = enabled("GGML_CPU_DISSEMINATION_BARRIER") && c.threads > 1 && c.threads <= 64;
            if ((used_barriers > 0) != expected) good = false;
        }
        barriers_used += used_barriers;''')
            marker = '    std::fflush(stdout);'
            assert fixture.count(marker) == 1
            fixture = fixture.replace(marker, '''    std::printf("BARRIER_CASE id=%d threads=%d calls=%llu\\n",id,c.threads,(unsigned long long)barriers_used);
''' + marker)
        if name == 'threads':
            marker = '            failures += !ok;'
            assert fixture.count(marker) == 1
            fixture = fixture.replace(marker, '''            uint64_t hash = 14695981039346656037ULL;
            const auto bytes = reinterpret_cast<const unsigned char *>(data);
            for (size_t i = 0; i < ggml_nbytes(product); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
            std::printf("THREAD_HASH node=%d setting=%s hash=%016llx\\n",node,setting.c_str(),(unsigned long long)hash);
''' + marker)
        publish(output, fixture_audit(fixture).encode())
        inputs[str(output)] = sha256(output)
    for p in (Path(__file__), BASE / 'flash_dissemination_transform_0908.py', BASE / 'glm_flash_q8_trial.py'):
        if p.name != 'glm_flash_q8_trial.py' or not resume:
            publish(OUT / p.name, p.read_bytes())
    result = dict(started=time.time(), passed=False, parent_sha256=parent['library_sha256'],
                  input_sha256=inputs, steps=[], checks=[], numa_checks=[], timings=[])
    if resume:
        result = json.loads((OUT / 'result.json').read_text())
        assert result['build_completed'] and not result['passed'] and not result.get('error')
        assert result['input_sha256'] == inputs and not result['checks'] and not result['numa_checks']
        result['validation_started'] = time.time()
    result['mode'] = 'compile_only' if build_only else 'build_and_validate'
    cwd = ENGINE / 'build-goal/ggml/src'
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(command, label, env=None):
        validate()
        log = OUT / (label + '.log')
        prior = [step for step in result['steps'] if step['label'] == label]
        if prior:
            assert resume and len(prior) == 1 and prior[0]['command'] == command and prior[0]['exit_code'] == 0
            return log.read_text()
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            result['owned_process'] = dict(pid=process.pid, label=label)
            save()
            try:
                deadline = time.monotonic() + 600
                while process.poll() is None:
                    if guard is not None:
                        guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.5)
                assert process.returncode == 0, (label, process.returncode)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result['steps'].append(dict(label=label, command=command, exit_code=process.returncode))
        save()
        print(json.dumps(dict(completed=label)), flush=True)
        return log.read_text()
    def verify_loaded(log, expected, enabled):
        loaded, = re.findall(r'^BARRIER_CPU_LIBRARY (.+)$', log, re.M)
        assert Path(loaded).resolve() == expected.resolve()
        calls, = map(int, re.findall(r'^DISSEMINATION_CALLS (\d+)$', log, re.M))
        assert (calls > 0) == enabled
        return calls
    save()
    try:
        link = list(parent['link_command'])
        link[link.index('-o') + 1] = str(PRIVATE / 'parent-link.so')
        run(link, 'parent-link')
        assert sha256(PRIVATE / 'parent-link.so') == parent['library_sha256']
        command = list(original_command)
        command[command.index('-o') + 1] = str(PRIVATE / 'baseline.o')
        run(command, 'baseline-compile')
        for label, path in [('original', old_object), ('rebuilt', str(PRIVATE / 'baseline.o'))]:
            run(['objcopy', '--dump-section', '.text=' + str(PRIVATE / (label + '.text')),
                 path, str(PRIVATE / (label + '.copy.o'))], label + '-text')
        assert (PRIVATE / 'original.text').read_bytes() == (PRIVATE / 'rebuilt.text').read_bytes()
        command[-1] = str(PRIVATE / 'ggml-cpu.c')
        command[command.index('-o') + 1] = str(PRIVATE / 'ggml-cpu.c.o')
        command[1:1] = ['-I' + str(PRIVATE)]
        run(command, 'private-compile')
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        link[link.index(old_object)] = command[command.index('-o') + 1]
        link[link.index('-o') + 1] = str(library)
        run(link, 'private-link')
        for name, target in [('libggml-cpu.so.0', library.name), ('libggml-cpu.so', 'libggml-cpu.so.0')]:
            p = PRIVATE / name
            if p.exists():
                assert resume and p.is_symlink() and p.resolve() == library.resolve()
            else:
                p.symlink_to(target)
        manifest = dict(library=str(library), library_sha256=sha256(library), parent_manifest=str(parent_path),
                        parent_sha256=parent['library_sha256'], compile_command=command, link_command=link,
                        input_sha256=inputs, baseline_link_identical=True, unpatched_text_identical=True,
                        private_source_sha256={str(p): sha256(p) for p in (PRIVATE / 'ggml-cpu.c', PRIVATE / 'flash-local-barrier-0908.h')})
        publish(PRIVATE / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
        result.update(library=str(library), library_sha256=sha256(library), baseline_link_identical=True, unpatched_text_identical=True)
        pinned = Path(current['pinned_directory'])
        for name in source_fixtures:
            flags = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
            flags += ['-I' + str(ENGINE / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            flags += [str(OUT / (name + '-check.cpp')), '-L' + str(PRIVATE), '-L' + str(pinned),
                      '-Wl,-rpath,' + str(PRIVATE) + ':' + str(pinned), '-lggml-cpu', '-lggml-base',
                      '-ldl', '-pthread', '-o', str(OUT / (name + '-check'))]
            run(flags, name + '-fixture-compile')
        result['binary_sha256'] = {name: sha256(OUT / (name + '-check')) for name in source_fixtures}
        result['build_completed'] = True
        if build_only:
            assert all(sha256(p) == digest for p, digest in inputs.items())
            validate()
            print(json.dumps(dict(build_completed=True, validated=False, library_sha256=sha256(library))), flush=True)
            return
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_', 'NUMA_REDUCE_TEST_'))}
        env.update(current['runtime_env'])
        env['LD_LIBRARY_PATH'] = str(PRIVATE) + ':' + str(pinned)
        # Ordinary graph fixture uses its own explicit affinity and CPU backend.
        graph_env = {k: v for k, v in env.items() if not k.startswith('GGML_CPU_NUMA_')}
        outputs = []
        for label, enabled, disabled, is_parent in [('parent', False, False, True), ('off', False, False, False),
                                                    ('on', True, False, False), ('no-fusion', True, True, False)]:
            trial = dict(graph_env, GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)),
                         GGML_CPU_DISSEMINATION_AUDIT='1', GGML_CPU_DISABLE_FUSION=str(int(disabled)))
            expected = Path(parent['library']) if is_parent else library
            if is_parent:
                trial['LD_LIBRARY_PATH'] = str(expected.parent) + ':' + str(pinned)
            output = OUT / (label + '.bin')
            log = run(['taskset', '-c', '0-127', str(OUT / 'graph-check'), str(output)], 'graph-' + label, trial)
            calls = verify_loaded(log, expected, enabled)
            assert 'SUMMARY cases=98 failures=0' in log
            barrier_rows = re.findall(r'^BARRIER_CASE id=(\d+) threads=(\d+) calls=(\d+)$', log, re.M)
            assert len(barrier_rows) == 98
            assert all((int(calls) > 0) == (enabled and 1 < int(threads) <= 64) for _, threads, calls in barrier_rows)
            outputs.append(output)
            result['checks'].append(dict(label=label, cases=98, samples_per_case=3, barrier_calls=calls,
                                         output_bytes=output.stat().st_size, output_sha256=sha256(output)))
            save()
        assert all(p.read_bytes() == outputs[0].read_bytes() for p in outputs[1:])
        result['bit_exact'] = True
        for label, enabled, is_parent in [('parent', False, True), ('off', False, False), ('on', True, False)]:
            expected = Path(parent['library']) if is_parent else library
            trial = dict(env, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(pinned),
                         GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)), GGML_CPU_DISSEMINATION_AUDIT='1')
            for name, args, expected_summary in [('threads', [str(OUT / ('threads-' + label + '.txt'))], 'cases=40 failures=0'),
                                                 ('reduce', ['--fused-graph'], 'Fused graph: 32 cases, 0 failures')]:
                log = run(['taskset', '-c', '0-127', str(OUT / (name + '-check')), *args], name + '-' + label, trial)
                calls = verify_loaded(log, expected, enabled)
                assert expected_summary in log
                hashes = re.findall(r' hash=([0-9a-f]+)$', log, re.M)
                result['numa_checks'].append(dict(label=label, name=name, barrier_calls=calls, hashes=hashes, passed=True))
                save()
        hash_groups = [x['hashes'] for x in result['numa_checks'] if x['name'] == 'reduce']
        assert all(len(x) == 32 and x == hash_groups[0] for x in hash_groups)
        hash_groups = [x['hashes'] for x in result['numa_checks'] if x['name'] == 'threads']
        assert all(len(x) == 40 and x == hash_groups[0] for x in hash_groups)
        for index, enabled in enumerate((False, True, True, False)):
            trial = dict(graph_env, GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)))
            log = run(['taskset', '-c', '0-127', str(OUT / 'graph-check'), '--timing'], f'timing-{index}-{int(enabled)}', trial)
            rows = re.findall(r'^PASS .* tokens=(\d+) .* weighted=(\d+) .* ms=([\d.]+)$', log, re.M)
            assert len(rows) == 4
            result['timings'].append(dict(enabled=enabled, graph_ms={f'{w}-{t}': float(ms) for t, w, ms in rows}))
        assert all(sha256(p) == digest for p, digest in inputs.items())
        validate()
        result['passed'] = True
        print(json.dumps(dict(passed=True, library_sha256=sha256(library), checks=result['checks'],
                              numa_checks=result['numa_checks'], timings=result['timings'])), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--build-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(options.build_only, options.resume)
