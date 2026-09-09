#!/usr/bin/env python3
"""Compare existing dense chunk settings on actual Flash Q8 matrix shapes."""
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

OUT = BASE/'results/glm-flash-q8-dense-chunks-0908b'
ENGINE = BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE/'validated-chunk16-bin'
SHAPES = [(4096,2048,'attn_q'), (2048,4096,'attn_output'), (4096,512,'attn_kv_a_mqa'),
          (4096,4096,'attn_output'), (4096,38720,'output'), (1536,4096,'attn_q_b'),
          (4096,128,'ssm_f_a'), (512,4096,'ffn_down_shexp'), (4096,3072,'ffn_gate'),
          (4096,1536,'attn_q_a'), (128,2048,'ssm_f_b'), (3072,4096,'ffn_down'), (4096,16,'ssm_beta')]


def main():
    os.umask(0o077)
    m = Manager()
    current = m.validate_current()
    assert current['pooling'] and current['drafts'] == 0 and current['workers'] == 15
    guard = ModelMeasurementGuard(current['pid'], {current['pid']:PORT}, inference_snapshot)
    guard.assert_idle()
    library = Path(current['cpu_library'])
    assert sha256(library) == current['cpu_sha256'] == 'f86a02d1d042bd2ed92b09af28983dd53a001f32d3624901599c26f7eb206191'
    original = BASE/'results/glm-flash-q8-batch-0908/q8-check.cpp'
    source = original.read_text()
    marker = '    ggml_backend_load_all();'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\\n", runtime.dli_fname);''')
    marker = 'native_ms=%.3f packed_ms=%.3f'
    assert source.count(marker) == 1
    source = source.replace(marker, 'native_ms=%.6f packed_ms=%.6f')
    marker = '    ggml_set_name(w, "blk.0.attn_output.weight");'
    assert source.count(marker) == 1
    names = []
    for k,rows,name in SHAPES:
        tensor_name = 'output.weight' if name == 'output' else 'blk.0.'+name+'.weight'
        names.append(f'    if (k == {k} && rows == {rows}) ggml_set_name(w, "{tensor_name}");')
    source = source.replace(marker, '\n'.join(names))
    marker = ': q8 ? std::vector<std::pair<int, int>>{{256, 512}, {512, 256}, {1536, 384}, {2048, 512}, {4096, 512}, {16384, 64}}'
    assert source.count(marker) == 1
    source = source.replace(marker, ': q8 ? std::vector<std::pair<int, int>>{'+','.join('{%d,%d}'%(k,r) for k,r,_ in SHAPES)+'}')
    marker = '        if (work_sharing && moe == dense_work_sharing) continue;'
    assert source.count(marker) == 1
    source = source.replace(marker, marker+'\n        if (q8 && (moe || fused || (tokens != 1 && tokens != 3))) continue;')
    marker = '        uint64_t checksum = 14695981039346656037ULL;'
    assert source.count(marker) == 1
    source = source.replace(marker, '''        if (const char * path = std::getenv("REPACK_TEST_DUMP")) {
            FILE * stream = std::fopen(path, "ab");
            if (!stream || std::fwrite(got.values.data(), sizeof(float), got.values.size(), stream) != got.values.size() || std::fclose(stream)) std::abort();
        }
''' + marker)
    OUT.mkdir(exist_ok=False)
    fixture = OUT/'dense-chunk-check.cpp'
    fixture.write_text(source)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    inputs = {str(p):sha256(p) for p in (Path(__file__), original, fixture, library)}
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_'))}
    environment.update(LD_LIBRARY_PATH=str(library.parent)+':'+str(PINNED), GGML_CPU_X16_Q8_0='1',
        GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096', REPACK_TEST_SMALL_BATCHES='1',
        REPACK_TEST_THREADS='15', REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1',
        REPACK_TEST_DENSE_WORK_SHARING='1', REPACK_TEST_REPEATS='30', REPACK_TEST_TIMING_MEDIAN='1')
    result = dict(started=time.time(),passed=False,cpu_sha256=sha256(library),shapes=SHAPES,
                  input_sha256=inputs,steps=[],runs=[],scope='Component timings and output comparisons only; no full-model or memory-bandwidth result.')
    def save(): (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()

    def run(command,label,env=None):
        m.validate_current(); guard.assert_idle()
        log = OUT/(label+'.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command,cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+600
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline,label
                    time.sleep(.5)
                assert process.returncode == 0,(label,process.returncode)
            finally:
                if process.poll() is None: process.terminate(); process.wait(timeout=10)
        result['steps'].append(dict(label=label,command=command)); save()
        print(json.dumps(dict(completed=label)),flush=True)
        return log.read_text()

    try:
        binary = OUT/'dense-chunk-check'
        command = ['/usr/bin/c++','-O3','-std=c++17','-march=native','-fopenmp']
        command += ['-I'+str(ENGINE/p) for p in ('include','ggml/include','ggml/src','ggml/src/ggml-cpu')]
        command += [str(fixture),'-L'+str(library.parent),'-L'+str(PINNED),'-Wl,-rpath,'+environment['LD_LIBRARY_PATH'],
                    '-lggml-cpu','-lggml-base','-ldl','-pthread','-o',str(binary)]
        run(command,'compile')
        result['binary_sha256'] = sha256(binary)
        reference = None
        for index,chunk in enumerate((16,32,64,128,128,64,32,16)):
            output = OUT/f'run-{index}-chunk{chunk}.bin'
            assert not output.exists()
            env = dict(environment,GGML_CPU_X16_CHUNK_MAX=str(chunk),REPACK_TEST_DUMP=str(output))
            label = f'run-{index}-chunk{chunk}'
            log = run(['taskset','-c','48-62',str(binary),'q8'],label,env)
            cpu_library, = re.findall(r'^CPU_LIBRARY (.+)$',log,re.M)
            assert Path(cpu_library).resolve() == library.resolve()
            assert 'Repacking: 26 cases, 0 failures' in log
            rows = re.findall(r'^PASS type=q8_0 k=(\d+) rows=(\d+) threads=15 tokens=(\d+) moe=0 fused=0 .* packed_ms=([\d.]+)',log,re.M)
            assert len(rows) == 26
            values = output.read_bytes()
            if reference is None: reference = values
            assert values == reference,'Dense chunk outputs differ'
            result['runs'].append(dict(label=label,chunk=chunk,cpu_library=cpu_library,bit_exact=True,output_bytes=len(values),
                output_sha256=sha256(output),graph_ms={f'{k}-{r}-{t}':float(ms) for k,r,t,ms in rows}))
            save()
        assert all(sha256(p)==digest for p,digest in inputs.items())
        m.validate_current(); guard.assert_idle()
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
