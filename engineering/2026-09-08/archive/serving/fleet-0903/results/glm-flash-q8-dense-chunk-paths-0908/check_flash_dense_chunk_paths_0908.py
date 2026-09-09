#!/usr/bin/env python3
"""Check larger Q8 chunks in 3D attention, dense fusion, and prefill."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE/'results/glm-flash-q8-dense-chunk-paths-0908'
ENGINE = BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE/'validated-chunk16-bin'


def main():
    os.umask(0o077)
    manager = Manager()
    current = manager.validate_current()
    guard = ModelMeasurementGuard(current['pid'], {current['pid']:PORT}, inference_snapshot)
    guard.assert_idle()
    cpu = Path(current['cpu_library'])
    assert sha256(cpu) == current['cpu_sha256'] == 'f86a02d1d042bd2ed92b09af28983dd53a001f32d3624901599c26f7eb206191'
    OUT.mkdir(exist_ok=False)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    sources = {'attention': BASE/'results/glm-flash-q8-attention3d-0908/glm-q8-attention3d-check.cpp',
               'fused': BASE/'results/glm-flash-q8-batch-0908/q8-check.cpp'}
    inputs = {str(p):sha256(p) for p in [Path(__file__),cpu,*sources.values()]}
    fixtures = {}
    for kind,path in sources.items():
        source = path.read_text()
        assert source.count('    ggml_backend_load_all();') == 1
        source = source.replace('    ggml_backend_load_all();','')
        if kind == 'attention':
            source = source.replace('    setenv("GGML_CPU_X16_CHUNK_MAX", "16", 1);\n','')
            source = source.replace('{1, 2, 3, 4, 5, 9}', '{1, 2, 3, 4, 5, 9, 64}')
            marker = '        uint64_t digest = 14695981039346656037ULL;'
            value = 'candidate'
        else:
            source = source.replace('    const bool q8 = argc > 1 && std::string(argv[1]) == "q8";', '''    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\\n", runtime.dli_fname);
    const bool q8 = argc > 1 && std::string(argv[1]) == "q8";''')
            marker = ': q8 ? std::vector<std::pair<int, int>>{{256, 512}, {512, 256}, {1536, 384}, {2048, 512}, {4096, 512}, {16384, 64}}'
            assert source.count(marker) == 1
            source = source.replace(marker, ': q8 ? std::vector<std::pair<int, int>>{{4096,512},{4096,3072}}')
            source = source.replace('std::vector<int>{1, 2, 3, 4, 5, 9}', 'std::vector<int>{1, 3, 64}')
            marker = '        if (work_sharing && moe == dense_work_sharing) continue;'
            assert source.count(marker) == 1
            source = source.replace(marker, marker+'\n        if (q8 && (moe || !fused)) continue;')
            marker = '        uint64_t checksum = 14695981039346656037ULL;'
            value = 'got.values'
        assert source.count(marker) == 1
        source = source.replace(marker, f'''        FILE * dump = std::fopen(std::getenv("REPACK_TEST_DUMP"), "ab");
        if (!dump || std::fwrite({value}.data(), sizeof(float), {value}.size(), dump) != {value}.size() || std::fclose(dump)) std::abort();
''' + marker)
        fixture = OUT/(kind+'.cpp'); fixture.write_text(source)
        inputs[str(fixture)] = sha256(fixture)
        fixtures[kind] = fixture
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_'))}
    env.update(LD_LIBRARY_PATH=str(cpu.parent)+':'+str(PINNED), GGML_CPU_X16_Q8_0='1',
        GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',
        REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
    result = dict(started=time.time(),passed=False,cpu_sha256=sha256(cpu),input_sha256=inputs,steps=[],runs=[])
    def save(): (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()

    def run(command,label,environment=None):
        manager.validate_current(); guard.assert_idle()
        log = OUT/(label+'.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command,cwd=BASE,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+600
                while process.poll() is None:
                    guard.assert_idle(); assert time.monotonic() < deadline,label
                    time.sleep(.5)
                assert process.returncode == 0,(label,process.returncode)
            finally:
                if process.poll() is None: process.terminate(); process.wait(timeout=10)
        result['steps'].append(dict(label=label,command=command)); save()
        print(json.dumps(dict(completed=label)),flush=True)
        return log.read_text()

    try:
        for kind,fixture in fixtures.items():
            binary = OUT/kind
            flags = ['/usr/bin/c++','-O3','-std=c++17','-march=native','-fopenmp']
            flags += ['-I'+str(ENGINE/p) for p in ('include','ggml/include','ggml/src','ggml/src/ggml-cpu')]
            flags += [str(fixture),'-L'+str(cpu.parent),'-L'+str(PINNED),'-Wl,-rpath,'+env['LD_LIBRARY_PATH'],
                      '-lggml-cpu','-lggml-base','-ldl','-pthread','-o',str(binary)]
            run(flags,'compile-'+kind)
            modes = [(0,False)] if kind == 'attention' else [(t,p) for t in (1,15) for p in (False,True)]
            for threads,padded in modes:
                reference = None
                for chunk in (16,128):
                    label = f'{kind}-t{threads}-p{int(padded)}-chunk{chunk}'
                    output = OUT/(label+'.bin')
                    assert not output.exists()
                    environment = dict(env,GGML_CPU_X16_CHUNK_MAX=str(chunk),REPACK_TEST_DUMP=str(output))
                    if threads: environment['REPACK_TEST_THREADS'] = str(threads)
                    if padded: environment['REPACK_TEST_PADDED'] = '1'
                    command = ['taskset','-c','48-62',str(binary)]+(['q8'] if kind == 'fused' else [])
                    log = run(command,label,environment)
                    marker = 'Attention: 168 cases, 0 failures' if kind == 'attention' else 'Repacking: 6 cases, 0 failures'
                    assert marker in log
                    loaded, = re.findall(r'^CPU_LIBRARY (.+)$',log,re.M)
                    assert Path(loaded).resolve() == cpu.resolve()
                    values = output.read_bytes()
                    if reference is None: reference = values
                    assert values == reference,label
                    result['runs'].append(dict(label=label,kind=kind,chunk=chunk,threads=threads,padded=padded,
                        cases=168 if kind=='attention' else 6,output_bytes=len(values),output_sha256=sha256(output),bit_exact=True,cpu_library=loaded))
                    save()
        assert all(sha256(p)==digest for p,digest in inputs.items())
        manager.validate_current(); guard.assert_idle()
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time(); save()


if __name__ == '__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
