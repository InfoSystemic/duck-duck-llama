#!/usr/bin/env python3
"""Check isolated Q8 R8 candidate outputs, branch coverage, and component timing."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from benchmark_flash_q4_selected_0910 import background
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, runtime_environment, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-r8-projection-validation-0910'


def fixture(source):
    source = source[:source.index('int main(int argc,char ** argv) {')]
    old = 'ggml_set_name(w, moe ? "blk.0.ffn_down_exps.weight" : k == 1536 ? "blk.0.ssm_out.weight" : "blk.0.hc_attn_down.weight");'
    new = '''ggml_set_name(w, moe ? (k==2560 ? "blk.0.ffn_gate_exps.weight" : "blk.0.ffn_down_exps.weight") :
        k==1536 ? "blk.0.ssm_out.weight" : rows==10240 ? "blk.0.hc_attn_up.weight" : "blk.0.hc_attn_down.weight");'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = 'const int src_rows = moe && !fused && std::getenv("REPACK_TEST_DOWN") ? used : 1;'
    assert source.count(old) == 1
    source = source.replace(old, 'const int src_rows = moe && k!=2560 && !fused && std::getenv("REPACK_TEST_DOWN") ? used : 1;')
    old = 'if (std::getenv("REPACK_TEST_FULL_GLM_Q5")) {\n            for (uint8_t value : weights)'
    assert source.count(old) == 1
    source = source.replace(old, 'if (true) {\n            for (uint8_t value : weights)')
    return source + r'''
int main(int argc,char ** argv) {
    if (argc!=2) return 2;
    setenv("GGML_CPU_Q8_0_REPACK","1",1);
    setenv("GGML_CPU_Q8_0_REPACK_FORCE","1",1);
    setenv("REPACK_TEST_DOWN","1",1);
    ggml_backend_load_all();
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\n",runtime.dli_fname);
    using audit_fn=uint64_t (*)(int);
    auto audit=reinterpret_cast<audit_fn>(dlsym(RTLD_DEFAULT,"ggml_cpu_q8_r8_k160_calls"));
    const bool prepare=std::getenv("GGML_CPU_Q8_R8_K160_PREP") && std::strcmp(std::getenv("GGML_CPU_Q8_R8_K160_PREP"),"1")==0;
    const bool counting=std::getenv("GGML_CPU_Q8_R8_K160_AUDIT") && std::strcmp(std::getenv("GGML_CPU_Q8_R8_K160_AUDIT"),"1")==0;
    if (prepare && !audit) std::abort();
    std::printf("AUDIT_SYMBOL %d\n",!!audit);
    FILE * dump=std::fopen(argv[1],"wb");
    if (!dump) return 2;
    const int threads=std::getenv("REPACK_TEST_THREADS") ? std::atoi(std::getenv("REPACK_TEST_THREADS")) : 15;
    int failures=0,cases=0;
    struct shape { int k,rows; bool moe; };
    for (auto dims:std::vector<shape>{{1536,2560,false},{160,2560,true},{2560,160,true},
            {10240,64,false},{10240,96,false},{64,10240,false},{96,10240,false},
            {128,64,true},{192,64,true},{160,64,false}})
    for (int tokens:{1,3,5}) {
        const int k=dims.k,rows=dims.rows;
        const bool moe=dims.moe;
        if (std::getenv("REPACK_TEST_TIMING_ONLY") && !(k==1536 || (k==160 && moe && rows==2560))) continue;
        auto ref=run(false,k,rows,tokens,threads,moe,false,GGML_TYPE_Q8_0);
        if (audit) audit(1);
        auto got=run(true,k,rows,tokens,threads,moe,false,GGML_TYPE_Q8_0);
        const uint64_t calls=audit ? audit(1) : 0;
        bool okay=ref.values.size()==got.values.size() && ref.weight_digest==got.weight_digest;
        if (counting && prepare && k==160 && moe) okay &= calls>0;
        if (!counting || !prepare || k!=160) okay &= calls==0;
        float max_abs=0,max_rel=0;
        for (size_t i=0;i<got.values.size();++i) {
            const float delta=std::abs(ref.values[i]-got.values[i]);
            max_abs=std::max(max_abs,delta);
            max_rel=std::max(max_rel,delta/(1+std::abs(ref.values[i])));
            okay &= std::isfinite(got.values[i]) && delta<=2e-4f*(1+std::abs(ref.values[i]));
        }
        if (std::fwrite(got.values.data(),sizeof(float),got.values.size(),dump)!=got.values.size()) return 2;
        std::printf("%s k=%d rows=%d threads=%d tokens=%d moe=%d native_ms=%.6f packed_ms=%.6f max_abs=%.8g max_scaled=%.8g weights=%016llx calls=%llu\n",
            okay?"PASS":"FAIL",k,rows,threads,tokens,moe,ref.ms,got.ms,max_abs,max_rel,
            (unsigned long long)got.weight_digest,(unsigned long long)calls);
        std::fflush(stdout);++cases;failures+=!okay;
    }
    if (std::fclose(dump)) return 2;
    std::printf("Q8_R8_PROJECTION cases=%d failures=%d\n",cases,failures);
    return failures?1:0;
}
'''


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    parent_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    build_path = BASE / 'results/qwen-r8-projection-build-0910/result.json'
    parent, build = [json.loads(path.read_text()) for path in [parent_path, build_path]]
    assert parent['passed'] and build['passed']
    assert sha256(parent['cpu']) == parent['cpu_sha256']
    assert sha256(build['library']) == build['library_sha256']
    source_path = BASE / 'results/qwen-r8-tiles-probe-0910/q8-r8-tiles.cpp'
    inputs = {str(path): sha256(path) for path in [Path(__file__), source_path, parent_path, build_path,
        Path(parent['cpu']), Path(build['library']), BASE / 'model_measurement_guard.py']}
    for path, digest in build['private_source_sha256'].items():
        assert sha256(path) == digest
        inputs[path] = digest
    result = dict(started=time.time(), passed=False, peer_pid=peer['pid'], input_sha256=inputs,
        model_loaded=False, runtime_promoted=False, steps=[], runs=[],
        scope='Exact candidate/parent packed outputs within equal worker count, input layout, and route schedule. Native tolerance and audit branch coverage. Separate one-socket component timings; no model or IMC claim.')
    owned = None
    def save(): atomic_json(OUT / 'result.json', result)
    def cancel(*_): raise InterruptedError('Stop only the owned Qwen R8 fixture')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        cpp, binary = OUT / 'qwen-r8-projection.cpp', OUT / 'qwen-r8-projection'
        cpp.write_text(fixture(source_path.read_text()))
        result['fixture_sha256'] = sha256(cpp)
        def run(command, label, env=None):
            nonlocal owned
            guard.assert_idle()
            log = OUT / (label + '.log')
            with log.open('w') as handle:
                owned = subprocess.Popen(command, cwd=BASE, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode, log_sha256=sha256(log)))
            save()
            assert owned.returncode == 0, (label, owned.returncode)
            return log.read_text()
        save()
        try:
            runtime_dir = Path(build['server']).parent
            command = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
            command += ['-I' + str(ENGINE / path) for path in ['ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
            command += [str(cpp), '-L' + str(runtime_dir), '-Wl,-rpath,' + str(runtime_dir),
                        '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
            run(command, 'compile')
            result['binary_sha256'] = sha256(binary)
            common_env = {k: v for k, v in os.environ.items() if k not in runtime_environment(os.environ) and not k.startswith('REPACK_TEST_')}
            schedule = [(15, False, mode, False) for mode in ['parent', 'off', 'prep', 'ssm', 'both']]
            schedule += [(threads, padded, mode, False) for threads, padded in [(1, False), (15, True), (1, True)] for mode in ['parent', 'both']]
            schedule += [(15, False, mode, True) for mode in ['off', 'both', 'both', 'off']]
            expected = {}
            for index, (threads, padded, mode, timing) in enumerate(schedule):
                env = dict(common_env)
                env.update(parent['runtime_env'] if mode == 'parent' else build['runtime_env'])
                env.update(GGML_CPU_NUMA_DEVICES='0', GGML_CPU_NUMA_SHARED_DISPATCH='0',
                    GGML_CPU_Q8_0_REPACK_TRACE='1', GGML_CPU_Q8_0_REPACK_X_TILE='1',
                    GGML_CPU_Q8_R8_K160_PREP='1' if mode in ['prep', 'both'] else '0',
                    GGML_CPU_Q8_R8_SSM_TILE8='1' if mode in ['ssm', 'both'] else '0',
                    GGML_CPU_Q8_R8_K160_AUDIT='0' if timing else '1',
                    REPACK_TEST_THREADS=str(threads), REPACK_TEST_REPEATS='31' if timing else '3',
                    REPACK_TEST_TIMING_MEDIAN='1', REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1')
                if padded: env['REPACK_TEST_PADDED'] = '1'
                if timing: env['REPACK_TEST_TIMING_ONLY'] = '1'
                label = f'run-{index:02d}-{mode}-t{threads}-pad{int(padded)}-timing{int(timing)}'
                dump = OUT / (label + '.bin')
                before = background(peer['pid'])
                log = run(['taskset', '-c', '48' if threads == 1 else '48-62', str(binary), str(dump)], label, env)
                rows = [dict(field.split('=', 1) for field in line.split()[1:]) for line in log.splitlines() if line.startswith('PASS ')]
                cases = 6 if timing else 30
                assert len(rows) == cases and f'Q8_R8_PROJECTION cases={cases} failures=0' in log
                loaded, = re.findall(r'^CPU_LIBRARY (.+)$', log, re.M)
                expected_cpu = parent['cpu'] if mode == 'parent' else build['library']
                assert Path(loaded).resolve() == Path(expected_cpu).resolve()
                assert ('AUDIT_SYMBOL 0' if mode == 'parent' else 'AUDIT_SYMBOL 1') in log
                trace, = re.findall(r'q8_0_r8: AVX-512/VNNI GEMV n=(\d+) nr=(\d+) nc=(\d+) x_tile=(\d+)', log)
                assert trace[0] == '1536' and int(trace[3]) == (8 if mode in ['ssm', 'both'] else 1)
                key = (threads, padded, timing)
                digest = sha256(dump)
                if key not in expected:
                    assert mode in ['parent', 'off']
                    expected[key] = digest
                assert digest == expected[key], 'Candidate changed packed output bits'
                result['runs'].append(dict(index=index, mode=mode, threads=threads, padded=padded, timing=timing,
                    rows=rows, output_sha256=digest, exact_parent_outputs=True, kernel_trace=trace,
                    background_before=before, background_after=background(peer['pid'])))
                save()
                print(json.dumps(dict(index=index, mode=mode, threads=threads, padded=padded, timing=timing,
                    cases=cases, exact=True, specialized_calls=sum(int(row['calls']) for row in rows))), flush=True)
            assert all(sha256(path) == digest for path, digest in inputs.items())
            assert sha256(cpp) == result['fixture_sha256']
            manager.validate_current()
            result.update(passed=True, peer_preserved=True, correctness_cases=330, timing_cases=24)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                try: owned.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(owned.pid, signal.SIGKILL); owned.wait(timeout=10)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
