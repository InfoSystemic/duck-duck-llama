#!/usr/bin/env python3
"""Build and validate opt-in Q8 row batching without replacing installed code."""
import difflib
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q6-q8-batch-0907'
PRIVATE = OUT / 'private-cpu'
PINNED = ENGINE / 'validated-iq-batch3-bin'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    global OUT, PRIVATE
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--fused', action='store_true', help='Also batch the fused dense gate/up path')
    variants.add_argument('--experts', action='store_true', help='Enable an opt-in x16 selector for Q8 experts')
    variants.add_argument('--dense-extra', action='store_true', help='Enable x16 on selected additional dense Q8 projections')
    variants.add_argument('--wide', action='store_true', help='Batch up to eight dense Q8 rows')
    args = parser.parse_args()
    if args.fused:
        OUT = BASE / 'results/qwen-q6-q8-fused-batch-0907'
        PRIVATE = OUT / 'private-cpu'
    if args.experts:
        OUT = BASE / 'results/qwen-q6-q8-experts-0907'
        PRIVATE = OUT / 'private-cpu'
    if args.dense_extra:
        OUT = BASE / 'results/qwen-q6-q8-dense-extra-0907'
        PRIVATE = OUT / 'private-cpu'
    if args.wide:
        OUT = BASE / 'results/qwen-q6-q8-wide-batch-0907'
        PRIVATE = OUT / 'private-cpu'
    OUT.mkdir(exist_ok=False)
    PRIVATE.mkdir()
    parent_path = BASE / 'results/qwen-q6-expert-capacity-0907/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    assert digest(parent['library']) == parent['library_sha256']
    micro_path = BASE / ('results/qwen-q8-chains-wide-0907/result.json' if args.wide else
                         'results/qwen-q8-chains-batch-shapes-0907/result.json')
    micro = json.loads(micro_path.read_text())
    expected_kernel_values = 842400 if args.wide else 168480
    assert micro['passed'] and micro['batch'] and micro['rows'][0]['bit_exact_values'] == expected_kernel_values
    assert bool(micro.get('wide')) == args.wide
    header_path = Path(micro['batch_header']) if args.wide else BASE / 'qwen-q8-batch.h'
    assert digest(header_path) == micro['batch_header_sha256']
    result = dict(started=time.time(), passed=False, steps=[], fused=args.fused, experts=args.experts,
                  dense_extra=args.dense_extra, wide=args.wide,
                  parent_cpu_sha256=parent['library_sha256'])
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(command, label, environment=None):
        with (OUT / (label + '.log')).open('w') as log:
            completed = subprocess.run(command, cwd=ENGINE, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=240)
        result['steps'].append(dict(label=label, command=command, exit_code=completed.returncode))
        save()
        assert completed.returncode == 0, (label, completed.returncode)
        print(json.dumps({'completed': label}), flush=True)

    try:
        main_path = Path(parent['compile_command'][-1])
        arch_path = ENGINE / 'ggml/src/ggml-cpu/arch/x86/repack.cpp'
        main_source, arch_source = main_path.read_text(), arch_path.read_text()
        assert digest(arch_path) == micro['source_sha256']
        marker = '                for (int64_t i11 = 0; i11 < ne11;) {'
        assert main_source.count(marker) == 1
        main_changed = main_source.replace(marker, '''                static const bool q8_batch = [] {
                    const char * value = getenv("GGML_CPU_X16_Q8_BATCH");
                    return value && atoi(value) == 1;
                }();
''' + marker)
        old_nr = 'const int nr = triple ? 3 : pair_enabled && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;'
        assert main_changed.count(old_nr) == 1
        main_changed = main_changed.replace(old_nr,
            'const int nr = q8_batch && sp.dst_type == GGML_TYPE_Q8_0 ? (int) std::min<int64_t>(4, ne11 - i11) : triple ? 3 : pair_enabled && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;')
        if args.wide:
            main_changed = main_changed.replace('std::min<int64_t>(4, ne11 - i11)',
                                                'std::min<int64_t>(8, ne11 - i11)')
        if args.fused:
            old = '        std::vector<float> gtmp(chunk), utmp(chunk);'
            assert main_changed.count(old) == 1
            main_changed = main_changed.replace(old, '''        static const bool q8_batch = [] {
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
            assert main_changed.count(old) == 1
            main_changed = main_changed.replace(old, '''            for (int64_t i = 0; i < n1;) {
                const int nr = (int) std::min<int64_t>(max_batch, n1 - i);
                sp.gemv((int) k, gtmp.data(), chunk, gw, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                sp.gemv((int) k, utmp.data(), chunk, uw, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                for (int t = 0; t < nr; ++t) {
                    float * out = (float *) ((char *) dst->data + (i + t) * dst->nb[1]) + r0;
                    ggml_vec_swiglu_f32(r1 - r0, out, gtmp.data() + t * chunk, utmp.data() + t * chunk);
                }
                i += nr;
            }''')
        if args.experts:
            old = '    const bool is_attn  = nd == 2 && strstr(cur->name, ".attn_") != nullptr;'
            assert main_changed.count(old) == 1
            main_changed = main_changed.replace(old, '''    static const bool q8_experts = [] {
        const char * value = getenv("GGML_CPU_X16_Q8_EXPERTS");
        return value && atoi(value) == 1;
    }();
    if (q8_experts && nd == 3 && cur->type == GGML_TYPE_Q8_0 &&
            strstr(cur->name, "_exps.weight") != nullptr && cur->ne[0] % QK8_0 == 0) {
        return &t_q8_0;
    }
''' + old)
        if args.dense_extra:
            old = '    const bool is_attn  = nd == 2 && strstr(cur->name, ".attn_") != nullptr;'
            assert main_changed.count(old) == 1
            main_changed = main_changed.replace(old, '''    static const bool q8_dense_extra = [] {
        const char * value = getenv("GGML_CPU_X16_Q8_DENSE_EXTRA");
        return value && atoi(value) == 1;
    }();
    if (q8_dense_extra && nd == 2 && cur->type == GGML_TYPE_Q8_0 && cur->ne[0] % QK8_0 == 0 &&
            (strstr(cur->name, ".ssm_out.weight") || strstr(cur->name, ".ple_key.weight") ||
             strstr(cur->name, ".ple_value.weight") || strstr(cur->name, ".hc_attn_down.weight") ||
             strstr(cur->name, ".hc_ffn_down.weight") || strstr(cur->name, ".hc_head_down.weight") ||
             strcmp(cur->name, "output_hc_down.weight") == 0)) {
        return &t_q8_0;
    }
''' + old)
        signature = 'void ggml_gemv_q8_0_x16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {'
        avx = '#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)'
        assert arch_source.count(signature) == 1
        include = avx + '\n#include "qwen-q8-batch.h"\n#endif\n\n'
        arch_changed = arch_source.replace(signature, include + signature)
        entry = signature + '\n' + avx + '\n'
        assert arch_changed.count(entry) == 1
        arch_changed = arch_changed.replace(entry, entry + '    if (nr > 1) return qwen_q8_batch_dispatch(n, s, bs, vx, vy, nr, nc);\n')
        # Resolve the existing relative include from the private copy's location.
        assert arch_changed.count('#include "../../repack.h"') == 1
        arch_changed = arch_changed.replace('#include "../../repack.h"',
            '#include "' + str(ENGINE / 'ggml/src/ggml-cpu/repack.h') + '"')
        private_main, private_arch = PRIVATE / 'repack.cpp', PRIVATE / 'repack-x86.cpp'
        private_main.write_text(main_changed)
        private_arch.write_text(arch_changed)
        (PRIVATE / 'qwen-q8-batch.h').write_bytes(header_path.read_bytes())
        patch = ''.join(difflib.unified_diff(main_source.splitlines(True), main_changed.splitlines(True),
                       fromfile='capacity/repack.cpp', tofile='private/repack.cpp'))
        patch += ''.join(difflib.unified_diff(arch_source.splitlines(True), arch_changed.splitlines(True),
                        fromfile='original/arch/x86/repack.cpp', tofile='private/repack-x86.cpp'))
        (PRIVATE / 'q8-batch.patch').write_text(patch)
        main_command = list(parent['compile_command'])
        main_object, arch_object = PRIVATE / 'repack.cpp.o', PRIVATE / 'repack-x86.cpp.o'
        old_main_object = main_command[main_command.index('-o') + 1]
        main_command[main_command.index('-o') + 1] = str(main_object)
        main_command[-1] = str(private_main)
        commands = json.loads((ENGINE / 'build-goal/compile_commands.json').read_text())
        entries = [e for e in commands if e['file'] == str(arch_path)]
        assert len(entries) == 1
        arch_command = shlex.split(entries[0]['command'])
        output_index = arch_command.index('-o') + 1
        old_arch_object = str((Path(entries[0]['directory']) / arch_command[output_index]).resolve())
        arch_command[output_index] = str(arch_object)
        arch_command[arch_command.index(str(arch_path))] = str(private_arch)
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        link = [str(main_object) if x == old_main_object else str(arch_object) if x == old_arch_object else x
                for x in parent['link_command']]
        assert str(main_object) in link and str(arch_object) in link
        link[link.index('-o') + 1] = str(library)
        original_paths = set(parent['input_sha256']) | {str(main_path), str(arch_path), str(parent_path),
            str(micro_path), str(header_path), parent['library'], old_main_object, old_arch_object}
        inputs = {p: digest(p) for p in original_paths}
        run(main_command, 'main-compile')
        run(arch_command, 'arch-compile')
        run(link, 'private-link')
        (PRIVATE / 'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(input_sha256=inputs, library=str(library), library_sha256=digest(library),
            parent_cpu_sha256=parent['library_sha256'], parent_manifest=str(parent_path),
            private_source_sha256={str(p): digest(p) for p in (private_main, private_arch, PRIVATE / 'qwen-q8-batch.h')},
            compile_commands=[main_command, arch_command], link_command=link,
            fused=args.fused, experts=args.experts, dense_extra=args.dense_extra, wide=args.wide,
            scope='The validated 512-expert capacity fix plus opt-in exact Q8 dense row batching'
                  + (' including fused gate/up.' if args.fused else
                     ' and an opt-in x16 Q8 expert selector.' if args.experts else
                     ' and opt-in recurrent-output, PLE, and HC-down Q8 selectors.' if args.dense_extra else
                     ' extended through eight activation rows.' if args.wide else '.'))
        (PRIVATE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
        environment['LD_LIBRARY_PATH'] = str(PRIVATE) + ':' + str(PINNED)
        kernel_command = list(micro['command'])
        kernel_command[kernel_command.index('-o') + 1] = str(OUT / 'kernel-check')
        kernel_command.insert(kernel_command.index('-L' + str(PINNED)), '-L' + str(PRIVATE))
        kernel_command = ['-Wl,-rpath,' + str(PRIVATE) + ':' + str(PINNED) if x == '-Wl,-rpath,' + str(PINNED) else x
                          for x in kernel_command]
        run(kernel_command, 'kernel-check-compile')
        run([str(OUT / 'kernel-check'), 'check-only'], 'kernel-check', environment)
        kernel_rows = [json.loads(x) for x in (OUT / 'kernel-check.log').read_text().splitlines()]
        assert kernel_rows[0]['correctness_passed']
        result['kernel_check'] = kernel_rows[0]
        fixture = (BASE / 'iq2-repack-check.cpp').read_text()
        fixture = fixture.replace('ggml_set_name(w, "blk.0.ffn_gate_exps.weight");',
                                  'ggml_set_name(w, "blk.0.attn_output.weight");')
        fixture = fixture.replace('ggml_set_name(u, "blk.0.ffn_up_exps.weight");',
                                  'ggml_set_name(u, "blk.0.attn_up.weight");')
        fixture = fixture.replace('q8 ? std::vector<std::pair<int, int>>{{320, 512}, {10240, 320}}',
            'q8 ? std::vector<std::pair<int, int>>{{320, 64}, {640, 160}, {1536, 256}, {2560, 320}, {6144, 64}, {10240, 64}}')
        fixture = fixture.replace('std::vector<int>{1, 2, 3, 4, 9}', 'std::vector<int>{1, 2, 3, 4, 5, 9}')
        if args.wide:
            fixture = fixture.replace('std::vector<int>{1, 2, 3, 4, 5, 9}',
                                      'std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}')
        fixture_path = OUT / 'graph-check.cpp'
        fixture_path.write_text(fixture)
        graph_command = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp',
            '-I' + str(ENGINE / 'ggml/include'), '-I' + str(ENGINE / 'ggml/src'),
            '-I' + str(ENGINE / 'ggml/src/ggml-cpu'), str(fixture_path),
            '-L' + str(PRIVATE), '-L' + str(PINNED), '-Wl,-rpath,' + environment['LD_LIBRARY_PATH'],
            '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(OUT / 'graph-check')]
        run(graph_command, 'graph-check-compile')
        environment.update(GGML_CPU_X16_Q8_0='1', REPACK_TEST_DENSE_WORK_SHARING='1',
                           REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
        result['graph_checks'] = []
        for padded in (False, True):
            checks = []
            if padded: environment['REPACK_TEST_PADDED'] = '1'
            for enabled in ('0', '1'):
                environment['GGML_CPU_X16_Q8_BATCH'] = enabled
                label = 'graph-' + ('padded' if padded else 'standard') + '-batch' + enabled
                run(['taskset', '-c', '48-62', str(OUT / 'graph-check'), 'q8'], label, environment)
                rows = [x for x in (OUT / (label + '.log')).read_text().splitlines() if x.startswith('PASS ')]
                assert len(rows) == (324 if args.wide else 216), (label, len(rows))
                checks.append([x.split(' hash=')[-1] for x in rows])
            assert checks[0] == checks[1], 'Batching changed graph output bits'
            result['graph_checks'].append(dict(padded=padded, bit_exact_cases=len(checks[0])))
        assert all(digest(p) == h for p, h in inputs.items()), 'An original build input changed'
        result.update(passed=True, library=str(library), library_sha256=digest(library),
                      manifest=str(PRIVATE / 'manifest.json'))
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
