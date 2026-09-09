#!/usr/bin/env python3
"""Build and compare a private, opt-in four-member attention pooling fusion."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from glm_flash_q8_trial import Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'
OUT = BASE / 'results/glm-flash-q8-pool-0908'
PRIVATE = OUT / 'private-cpu'

DISPATCH = '''    if (ggml_cpu_softmax_pool_fusion && node->op == GGML_OP_SOFT_MAX && node_n + 2 < cgraph->n_nodes) {
        const enum ggml_op ops[] = { GGML_OP_SOFT_MAX, GGML_OP_MUL, GGML_OP_SUM_ROWS };
        const int outputs[] = { node_n + 2 };
        if (ggml_can_fuse_subgraph(cgraph, node_n, 3, ops, outputs, 1)) {
            struct ggml_tensor * mul = cgraph->nodes[node_n + 1];
            struct ggml_tensor * dst = cgraph->nodes[node_n + 2];
            const struct ggml_tensor * gate = node->src[0];
            const struct ggml_tensor * keys = mul->src[0] == node ? mul->src[1] : mul->src[1] == node ? mul->src[0] : NULL;
            float scale, max_bias;
            memcpy(&scale, (float *) node->op_params, sizeof(float));
            memcpy(&max_bias, (float *) node->op_params + 1, sizeof(float));
            if (keys && keys != node && gate && !node->src[1] && !node->src[2] &&
                    scale == 1.0f && max_bias == 0.0f && node->ne[0] == 4 &&
                    node->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
                    keys->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
                    dst->src[0] == mul && dst->ne[0] == 1 &&
                    dst->ne[1] == node->ne[1] && dst->ne[2] == node->ne[2] && dst->ne[3] == node->ne[3] &&
                    ggml_are_same_shape(gate, node) && ggml_are_same_shape(keys, node) && ggml_are_same_shape(mul, node) &&
                    ggml_is_contiguous(gate) && ggml_is_contiguous(keys) && ggml_is_contiguous(node) &&
                    ggml_is_contiguous(mul) && ggml_is_contiguous(dst) &&
                    !node->view_src && !mul->view_src && !dst->view_src) {
                ggml_compute_forward_softmax_pool_fused(params, node, mul, dst);
                return 2;
            }
        }
    }

'''

DECLARATION = 'void ggml_compute_forward_softmax_pool_fused(const struct ggml_compute_params * params, const struct ggml_tensor * probs, const struct ggml_tensor * mul, struct ggml_tensor * dst);\n'

KERNEL = '''void ggml_compute_forward_softmax_pool_fused(
        const ggml_compute_params * params,
        const ggml_tensor * probs,
        const ggml_tensor * mul,
        ggml_tensor * dst) {
    const ggml_tensor * gate = probs->src[0];
    const bool probs_first = mul->src[0] == probs;
    const ggml_tensor * keys = mul->src[probs_first ? 1 : 0];
    const int64_t rows = ggml_nrows(gate);
    const int64_t begin = rows * params->ith / params->nth;
    const int64_t end = rows * (params->ith + 1) / params->nth;
    float scale;
    memcpy(&scale, (const float *) probs->op_params, sizeof(float));
    for (int64_t row = begin; row < end; ++row) {
        float work[4], probability[4], product[4];
        const float * gp = (const float *) gate->data + row * 4;
        const float * kp = (const float *) keys->data + row * 4;
        ggml_vec_cpy_f32(4, work, gp);
        ggml_vec_scale_f32(4, work, scale);
        float max = -INFINITY;
        ggml_vec_max_f32(4, &max, work);
        ggml_float sum = ggml_vec_soft_max_f32(4, probability, work, max);
        sum = 1.0 / sum;
        ggml_vec_scale_f32(4, probability, sum);
        ggml_vec_mul_f32(4, product, probs_first ? probability : kp, probs_first ? kp : probability);
        ggml_vec_sum_f32(4, (float *) dst->data + row, product);
    }
}

'''


def main():
    manager = Manager()
    current = manager.validate_current()
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
    guard.assert_idle()
    OUT.mkdir(exist_ok=False)
    PRIVATE.mkdir()
    parent_path = BASE / 'results/glm-flash-q8-clamp-0908/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    assert parent['library_sha256'] == current['cpu_sha256'] == sha256(parent['library'])
    inputs = dict(parent['input_sha256'])
    for path, digest in parent['private_source_sha256'].items():
        assert sha256(path) == digest
        inputs[path] = digest
    fixture = BASE / 'flash-pool-check-0908.cpp'
    for p in (parent_path, Path(parent['library']), Path(__file__), fixture): inputs[str(p)] = sha256(p)
    source_dir = ENGINE / 'ggml/src/ggml-cpu'
    originals = {name: (source_dir/name).read_text() for name in ('ggml-cpu.c', 'ops.cpp', 'ops.h')}
    changed = dict(originals)
    marker = 'static struct ggml_state g_state = {0};'
    assert changed['ggml-cpu.c'].count(marker) == 1
    changed['ggml-cpu.c'] = changed['ggml-cpu.c'].replace(marker, marker + '\nstatic bool ggml_cpu_softmax_pool_fusion = false;')
    marker = '    if (node->op == GGML_OP_RMS_NORM) {\n        // RMS_NORM + MUL fusion'
    assert changed['ggml-cpu.c'].count(marker) == 1
    changed['ggml-cpu.c'] = changed['ggml-cpu.c'].replace(marker, DISPATCH + marker)
    marker = '        {\n            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
    assert changed['ggml-cpu.c'].count(marker) == 1
    changed['ggml-cpu.c'] = changed['ggml-cpu.c'].replace(marker, '''        {
            const char * env = getenv("GGML_CPU_SOFTMAX_POOL_FUSION");
            ggml_cpu_softmax_pool_fusion = env != NULL && atoi(env) == 1;
        }
''' + marker)
    marker = '// ggml_compute_forward_soft_max\n'
    assert changed['ops.cpp'].count(marker) == 1
    changed['ops.cpp'] = changed['ops.cpp'].replace(marker, KERNEL + marker)
    marker = 'void ggml_compute_forward_soft_max(const struct ggml_compute_params * params, struct ggml_tensor * dst);\n'
    assert changed['ops.h'].count(marker) == 1
    changed['ops.h'] = changed['ops.h'].replace(marker, DECLARATION + marker)
    for name, value in changed.items():
        p = source_dir/name
        if str(p) in inputs: assert inputs[str(p)] == sha256(p)
        inputs[str(p)] = sha256(p)
        (PRIVATE/name).write_text(value)
        (PRIVATE/(name+'.patch')).write_text(''.join(difflib.unified_diff(originals[name].splitlines(True), value.splitlines(True), fromfile=str(p), tofile=str(PRIVATE/name))))
    entries = json.loads((ENGINE/'build-goal/compile_commands.json').read_text())
    cwd = ENGINE/'build-goal/ggml/src'
    result = dict(passed=False, started=time.time(), parent_sha256=parent['library_sha256'], steps=[], checks=[], timings=[])
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUT/fixture.name).write_bytes(fixture.read_bytes())
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
    environment.update(LD_LIBRARY_PATH=str(PRIVATE)+':'+str(PINNED), GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096')

    def save(): (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def run(cmd, label, env=None):
        manager.validate_current(); guard.assert_idle()
        log = OUT/(label+'.log')
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+240
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert proc.returncode == 0, (label, proc.returncode)
            finally:
                if proc.poll() is None: proc.terminate(); proc.wait(timeout=10)
        result['steps'].append(dict(label=label, command=cmd)); save()
        print(json.dumps(dict(completed=label)), flush=True)
        return log.read_text()

    save()
    try:
        assert all(sha256(p) == h for p,h in inputs.items())
        link = list(parent['link_command'])
        compiled = []
        for name in ('ggml-cpu.c', 'ops.cpp'):
            source = source_dir/name
            entry, = [e for e in entries if e['file'] == str(source)]
            assert Path(entry['directory']) == cwd
            command = shlex.split(entry['command'])
            old_obj = str((cwd/command[command.index('-o')+1]).resolve())
            assert old_obj in link
            inputs[old_obj] = sha256(old_obj)
            baseline = PRIVATE/(name+'.baseline.o')
            command[command.index('-o')+1] = str(baseline)
            run(command, name+'-baseline-compile')
            sections = []
            for obj, label in ((old_obj, 'original'), (str(baseline), 'rebuilt')):
                section = PRIVATE/(name+'.'+label+'.text'); sections.append(section)
                run(['objcopy', '--dump-section', '.text='+str(section), obj, str(PRIVATE/(name+'.'+label+'.copy.o'))], name+'-'+label+'-text')
            assert sections[0].read_bytes() == sections[1].read_bytes(), name+' source no longer matches installed instructions'
            obj = PRIVATE/(name+'.o')
            command[command.index('-o')+1] = str(obj)
            command[command.index(str(source))] = str(PRIVATE/name)
            command[1:1] = ['-I'+str(PRIVATE), '-I'+str(source_dir)]
            run(command, name+'-private-compile')
            compiled.append(command)
            link[link.index(old_obj)] = str(obj)
        library = PRIVATE/'libggml-cpu.so.0.22.0'
        link[link.index('-o')+1] = str(library)
        run(link, 'private-link')
        (PRIVATE/'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(library=str(library), library_sha256=sha256(library), parent_manifest=str(parent_path), parent_sha256=parent['library_sha256'], input_sha256=inputs, private_source_sha256={str(PRIVATE/n):sha256(PRIVATE/n) for n in changed}, compile_commands=compiled, link_command=link, unpatched_text_identical=True)
        (PRIVATE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        result.update(library=str(library), library_sha256=sha256(library), unpatched_text_identical=True)
        binary = OUT/'flash-pool-check'
        flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I'+str(ENGINE/p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        links = ['-L'+str(PRIVATE), '-L'+str(PINNED), '-Wl,-rpath,'+environment['LD_LIBRARY_PATH'], '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
        run(flags+[str(fixture)]+links+['-o',str(binary)], 'fixture-compile')
        output_paths = []
        for label, enabled, disabled, ref in [('parent', '0', '0', True), ('off', '0', '0', False), ('on', '1', '0', False), ('disabled', '1', '1', False)]:
            env = dict(environment, GGML_CPU_SOFTMAX_POOL_FUSION=enabled, GGML_CPU_DISABLE_FUSION=disabled)
            if ref: env['LD_LIBRARY_PATH'] = str(Path(parent['library']).parent)+':'+str(PINNED)
            output = OUT/(label+'.bin'); output_paths.append(output)
            log = run(['taskset', '-c', '48-62', str(binary), str(output)], 'check-'+label, env)
            assert 'SUMMARY cases=25 failures=0' in log
            result['checks'].append(dict(label=label, cases=25, samples_per_case=3, output_bytes=output.stat().st_size, output_sha256=sha256(output)))
        original = output_paths[0].read_bytes()
        assert all(p.read_bytes() == original for p in output_paths[1:]), 'Pooling outputs differ'
        result['bit_exact'] = True
        for index, enabled in enumerate(('0','1','1','0')):
            env = dict(environment, GGML_CPU_SOFTMAX_POOL_FUSION=enabled)
            log = run(['taskset', '-c', '48-62', str(binary), '--timing'], f'timing-{index}-{enabled}', env)
            samples = re.findall(r'^PASS .* pools=(\d+).* ms=([\d.]+)$', log, re.M)
            assert len(samples) == 2
            result['timings'].append(dict(enabled=enabled=='1', graph_ms={n:float(ms) for n,ms in samples}))
        assert all(sha256(p)==h for p,h in inputs.items())
        manager.validate_current(); guard.assert_idle()
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time(); save()


if __name__ == '__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
