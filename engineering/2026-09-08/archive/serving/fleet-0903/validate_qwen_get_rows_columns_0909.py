#!/usr/bin/env python3
"""Validate gather correctness, worker coverage, and isolated CPU graph behavior."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
RUNTIME = BASE / 'results/qwen-shared-dispatch-runtime-0909/bin'


def main(label):
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    build_path = BASE / 'results/qwen-get-rows-columns-build-0909/result.json'
    manifest_path = build_path.parent / 'private-cpu/manifest.json'
    build, manifest = [json.loads(path.read_text()) for path in (build_path, manifest_path)]
    assert build['passed'] and build['baseline_object_identical'] and build['baseline_library_identical']
    assert all(sha256(path) == digest for path, digest in manifest['input_sha256'].items())
    assert all(sha256(path) == digest for path, digest in manifest['private_source_sha256'].items())
    fixed = Path(manifest['library'])
    assert sha256(fixed) == manifest['library_sha256'] == '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
    parent = Path(json.loads(Path(manifest['parent_manifest']).read_text())['libraries']['cpu']['path'])
    assert sha256(parent) == manifest['parent_sha256']
    base = Path(manifest['base']['path'])
    assert sha256(base) == manifest['base']['sha256']
    direct_source = BASE / 'check-qwen-get-rows-columns-0909.cpp'
    graph_source = BASE / 'parallel-copy-check.cpp'
    text = graph_source.read_text()
    marker = '    std::printf("Parallel copy: %d cases, %d failures\\n", cases, failures);'
    assert text.count(marker) == 1
    graph_text = text.replace(marker, '    for (int64_t nr : {1, 5}) { ++cases; failures += !gather_case(786432, nr, true, 15, 1); }\n' + marker)
    assert graph_text.count('int main() {') == 1
    graph_text = '#include "ops.h"\n#include <dlfcn.h>\n' + graph_text.replace('int main() {', '''int main() {
    Dl_info loaded = {};
    GGML_ASSERT(dladdr(reinterpret_cast<void *>(ggml_compute_forward_get_rows), &loaded));
    std::printf("GATHER_LIBRARY %s\\n", loaded.dli_fname);''')
    paths = [Path(__file__).resolve(), direct_source, graph_source, build_path, manifest_path, fixed, parent, base,
             BASE / 'benchmark_qwen_q6.py', BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py']
    inputs = {str(path):sha256(path) for path in paths}
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None
    def interrupted(signum, frame):
        raise InterruptedError('Stop only the owned gather validation')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        out = BASE / 'results' / label
        out.mkdir()
        (out / direct_source.name).write_bytes(direct_source.read_bytes())
        (out / 'graph-check.cpp').write_text(graph_text)
        result = dict(started=time.time(), passed=False, model_loaded=False, input_sha256=inputs,
                      private_source_sha256={str(path):sha256(path) for path in [out / direct_source.name, out / 'graph-check.cpp']},
                      parent_cpu_sha256=sha256(parent), fixed_cpu_sha256=sha256(fixed), steps=[], direct_checks=[], graph_checks=[],
                      scope='No model benchmark. Direct canonical row references and disjoint per-worker coverage, followed by the existing CPU graph copy suite extended to actual recurrent-state dimensions. Quantized input buffers include backing padding so the known parent defect can be demonstrated without an out-of-allocation read.')
        def save():
            (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        def run(command, name, environment=None, timeout=600):
            nonlocal owned
            guard.assert_idle()
            log_path = out / (name + '.log')
            with log_path.open('w') as log:
                owned = subprocess.Popen(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, step=name)
                save()
                deadline = time.monotonic()+timeout
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, name
                    time.sleep(.25)
            result['steps'].append(dict(name=name, command=command, exit_code=owned.returncode, log_sha256=sha256(log_path)))
            save()
            assert owned.returncode == 0, name
            print(json.dumps(dict(completed=name)), flush=True)
            return log_path.read_text()
        save()
        try:
            compile_prefix = ['/usr/bin/c++', '-std=c++17', '-O3', '-march=native', '-fopenmp', '-DGGML_USE_OPENMP',
                              '-I'+str(ENGINE/'ggml/include'), '-I'+str(ENGINE/'ggml/src'), '-I'+str(ENGINE/'ggml/src/ggml-cpu')]
            links = ['-L'+str(fixed.parent), '-L'+str(RUNTIME), '-Wl,-rpath,'+str(fixed.parent)+':'+str(RUNTIME),
                     '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
            for name, source in [('direct-check', out / direct_source.name), ('graph-check', out / 'graph-check.cpp')]:
                run([*compile_prefix, str(source), *links, '-o', str(out/name)], 'compile-'+name)
            result['idle_gate'] = guard.wait_idle(out/'waiting-for-idle.json')
            result['background_gate'] = wait_background(guard, 1219506, 4, out/'background-wait.json')
            env_base = {key:value for key,value in os.environ.items() if key not in runtime_environment(os.environ)}
            for name, library, flag in [('parent-off', parent, '0'), ('parent-on', parent, '1'),
                                        ('fixed-off', fixed, '0'), ('fixed-on', fixed, '1')]:
                environment = dict(env_base, LD_LIBRARY_PATH=str(library.parent)+':'+str(RUNTIME), GGML_CPU_PARALLEL_COPY=flag)
                text = run([str(out/'direct-check'), 'fixed' if library == fixed else 'parent'], name, environment)
                rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
                loaded, = [row for row in rows if 'cpu_library' in row]
                assert Path(loaded['cpu_library']).resolve() == library.resolve()
                summary, = [row for row in rows if row.get('summary')]
                assert summary['cases'] == 114 and summary['unexpected_failures'] == 0 and summary['inputs_preserved']
                assert summary['known_quantized_failures'] == (24 if name == 'parent-on' else 0)
                ownership = [row for row in rows if row.get('ownership')]
                assert len(ownership) == 2 and all(row['exact'] for row in ownership)
                expected = [15, 15] if name == 'fixed-on' else [1, 5]
                assert [row['active_workers'] for row in ownership] == expected
                result['direct_checks'].append(dict(name=name, **summary, ownership=ownership, cpu_library=str(library)))
                save()
            for index, (name, library) in enumerate([('parent',parent),('fixed',fixed),('fixed',fixed),('parent',parent)]):
                wait_background(guard, 1219506, 4, out/'graph-background-wait.json')
                environment = dict(env_base, LD_LIBRARY_PATH=str(library.parent)+':'+str(RUNTIME),
                                   GGML_CPU_PARALLEL_COPY='1', GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096')
                text = run(['taskset','-c','0-14',str(out/'graph-check')], f'graph-{index}-{name}', environment)
                assert 'Parallel copy: 46 cases, 0 failures' in text and '\nFAIL ' not in text
                mapped, = re.findall(r'^GATHER_LIBRARY (.+)$', text, re.M)
                assert Path(mapped).resolve() == library.resolve()
                targets = []
                for columns, rows, groups, padded, threads, elements, ms in re.findall(
                        r'PASS gather nc=(\d+) nr=(\d+) groups=(\d+) padded=(\d+) threads=(\d+) elements=(\d+) ms=([0-9.]+)', text):
                    if int(columns) == 786432:
                        targets.append(dict(columns=int(columns), rows=int(rows), threads=int(threads), elements=int(elements), graph_ms=float(ms)))
                assert len(targets) == 2
                result['graph_checks'].append(dict(name=name, index=index, cases=46, bit_exact=True, targets=targets))
                save()
            assert all(sha256(path) == digest for path,digest in {**inputs, **result['private_source_sha256']}.items())
            result.update(passed=True, corrected_reference_cases=342, parent_known_bug_cases=24,
                          all_float_outputs_exact=True, graph_cases=184,
                          model_gain_established=False, target_reached=False)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=15)
            result.update(finished=time.time(), peer_preserved=process_info(1219506)['start']=='103969952',
                          peer_service=read_service(18095))
            save()
            print(json.dumps({key:result[key] for key in ['passed','direct_checks','graph_checks','error'] if key in result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-get-rows-columns-validation-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    main(args.label)
