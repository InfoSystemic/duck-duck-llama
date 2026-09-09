#!/usr/bin/env python3
"""Compare bit-exact Q8 activation reductions without changing the running model."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from flash_q8_partials_fixture_0908 import SOURCE, SOURCE_SHA
from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE/'results/glm-flash-q8-sums-0908'
ENGINE = BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'

def generate():
    assert sha256(SOURCE) == SOURCE_SHA
    source = SOURCE.read_text()
    start = source.index('void ggml_gemv_q8_0_x16_q8_0(')
    end = source.index('\nvoid ggml_gemv2_q4_K_x16_q8_K(', start)
    original = source[start:end]
    body = original.replace('void ggml_gemv_q8_0_x16_q8_0(', 'template<int MODE> __attribute__((noinline)) static void candidate_q8(', 1)
    dispatch = '    if (nr > 1) return qwen_q8_batch_dispatch(n, s, bs, vx, vy, nr, nc);\n'
    assert body.count(dispatch) == 1
    body = body.replace(dispatch, '    static_assert(MODE >= 0 && MODE <= 2);\n')
    body = body.replace('if (nb > 512)', 'if (nb > 512 && !(MODE == 2 && nc == 16))', 1)
    first = body.index('    for (int b = 0; b < nb; b++) {')
    last = body.index('    for (int g = 0;', first)
    preparation = body[first:last]
    sad = preparation[preparation.index('        yd[b]'):]
    replacement = '''    if constexpr (MODE == 0) {
ORIGINAL
    } else if (MODE != 2 || nc != 16) {
        for (int b = 0; b < nb; b++) {
            ysum[b] = q8_activation_sum(vy8[b].qs);
SCALE
    }
'''.replace('ORIGINAL',preparation.rstrip()).replace('SCALE',sad.rstrip())
    body = body[:first]+replacement+body[last:]
    old = '            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * ysum[b]));'
    new = '''            const int32_t sum = MODE == 2 && nc == 16 ? q8_activation_sum(vy8[b].qs) : ysum[b];
            const float scale = MODE == 2 && nc == 16 ? GGML_CPU_FP16_TO_FP32(vy8[b].d) : yd[b];
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * sum));'''
    assert body.count(old) == 1
    body = body.replace(old,new)
    assert body.count('_mm512_set1_ps(yd[b])') == 1
    body = body.replace('_mm512_set1_ps(yd[b])','_mm512_set1_ps(scale)')
    header = '''#pragma once
#include <immintrin.h>
#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))
static inline int32_t q8_activation_sum(const int8_t * values) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) values);
    const __m256i biased = _mm256_xor_si256(q, _mm256_set1_epi8(char(0x80)));
    const __m256i sums = _mm256_sad_epu8(biased, _mm256_setzero_si256());
    const __m128i pairs = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    return int32_t(_mm_cvtsi128_si64(pairs) + _mm_extract_epi64(pairs, 1)) - 4096;
}
'''
    path = OUT/'q8-sums-candidates.h'
    path.write_text(header+body+'\n#undef GGML_X16_BC32\n')
    (OUT/'kernel.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), body.splitlines(True))))
    return path

def main():
    os.umask(0o077)
    m = Manager(); current = m.validate_current()
    guard = ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot)
    guard.assert_idle()
    cpu = Path(current['cpu_library']); pinned = Path(current['pinned_directory'])
    assert sha256(cpu) == current['cpu_sha256'] == 'f86a02d1d042bd2ed92b09af28983dd53a001f32d3624901599c26f7eb206191'
    assert current['drafts'] == 0 and current['dense_chunk'] == 16
    OUT.mkdir(exist_ok=False)
    header = generate()
    fixture = BASE/'flash-q8-sums-check-0908.cpp'
    for p in (Path(__file__), fixture, BASE/'flash_q8_partials_fixture_0908.py'):
        (OUT/p.name).write_bytes(p.read_bytes())
    inputs = {str(p):sha256(p) for p in (Path(__file__),fixture,header,SOURCE,cpu,BASE/'flash_q8_partials_fixture_0908.py',BASE/'glm_flash_q8_trial.py')}
    result = dict(started=time.time(),passed=False,current_pid=current['pid'],cpu_sha256=sha256(cpu),
                  input_sha256=inputs,steps=[],events=[],
                  scope='One-core kernel timing and bounded exact output checks; no model speed or memory-bandwidth result.')
    def save(): (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(command,label,env=None):
        m.validate_current(); guard.assert_idle()
        log = OUT/(label+'.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command,env=env,cwd=BASE,stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+600
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic()<deadline,label
                    time.sleep(.5)
                assert process.returncode==0,(label,process.returncode)
            finally:
                if process.poll() is None: process.terminate(); process.wait(timeout=10)
        result['steps'].append(dict(label=label,command=command));save()
        print(json.dumps(dict(completed=label)),flush=True)
        return log.read_text()
    save()
    try:
        binary = OUT/'q8-sums-check'
        command = ['/usr/bin/c++','-O3','-std=c++17','-march=native','-I'+str(OUT)]
        command += ['-I'+str(ENGINE/p) for p in ('include','ggml/include','ggml/src','ggml/src/ggml-cpu')]
        command += [str(fixture),str(cpu),str((pinned/'libggml-base.so').resolve()),'-ldl','-pthread','-o',str(binary)]
        run(command,'compile')
        result['binary_sha256']=sha256(binary)
        env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_'))}
        env['LD_LIBRARY_PATH']=str(cpu.parent)+':'+str(pinned)
        output=OUT/'outputs.bin'
        log=run(['taskset','-c','48',str(binary),str(output)],'probe',env)
        events=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
        result['events']=events
        assert Path(events[0]['path']).resolve()==cpu.resolve()
        check=next(x for x in events if x['event']=='correctness')
        assert check['passed'] and check['sum_cases']==18448 and check['matrix_cases']==240
        assert events[-1]==dict(event='done',passed=True)
        rows=[x for x in events if x['event']=='timing']
        assert len(rows)==80 and all(x['samples']==34 for x in rows)
        result['output_bytes']=output.stat().st_size
        assert result['output_bytes']==check['compared_values']*4
        result['output_sha256']=sha256(output)
        result['comparisons']=[]
        for row in rows:
            if row['mode'] not in ('sad','inline16'):continue
            references=[x for x in rows if x['k']==row['k'] and x['nc']==row['nc'] and x['rotating']==row['rotating'] and x['mode'] in ('library','copy')]
            assert len(references)==2 and all(x['hash']==row['hash'] for x in references)
            result['comparisons'].append(dict(k=row['k'],nc=row['nc'],rotating=row['rotating'],mode=row['mode'],
                change_percent={r['mode']:100*(row['median_us']/r['median_us']-1) for r in references}))
        assert all(sha256(p)==digest for p,digest in inputs.items())
        m.validate_current();guard.assert_idle()
        result['passed']=True
        print(json.dumps(dict(passed=True,correctness=check,comparisons=result['comparisons'])),flush=True)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['finished']=time.time();save()

if __name__=='__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
