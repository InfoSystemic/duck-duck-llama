#!/usr/bin/env python3
from pathlib import Path
import shutil
import difflib

root=Path(__file__).resolve().parents[2]
p=root/'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/ops.cpp'
backup=p.with_name(p.name+'.before-goal-hc-post-vector')
assert not backup.exists()
shutil.copy2(p,backup)
s=p.read_text()
start=s.index('static void ggml_compute_forward_dsv4_hc_post_f32(')
insert=s.index('    for (int64_t ir = ir0; ir < ir1; ++ir) {',start)
body=r'''
    static const bool vector_enabled = []() {
        const char * value = std::getenv("GGML_CPU_HC_POST_VECTOR");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (vector_enabled && hc == 4 && nbx0 == sizeof(float) &&
            nbr0 == sizeof(float) && nbd0 == sizeof(float)) {
        int64_t ir = ir0;
        while (ir < ir1) {
            const int64_t row = ir / n_embd;
            const int64_t idst = row % hc;
            const int64_t it = row / hc;
            const int64_t end = std::min(ir1, (row + 1) * n_embd) - row * n_embd;
            int64_t i = ir - row * n_embd;
            const float * xp = (const float *) ((const char *) x->data + it * nbx1);
            const float * rp[4];
            float cv[4];
            for (int j = 0; j < 4; ++j) {
                rp[j] = (const float *) ((const char *) residual->data + j * nbr1 + it * nbr2);
                cv[j] = *(const float *) ((const char *) comb->data + idst * nbc0 + j * nbc1 + it * nbc2);
            }
            const float pv = *(const float *) ((const char *) post->data + idst * nbp0 + it * nbp1);
            float * out = (float *) ((char *) dst->data + idst * nbd1 + it * nbd2);
#if defined(__AVX512F__)
            const __m512 p = _mm512_set1_ps(pv);
            const __m512 c0 = _mm512_set1_ps(cv[0]);
            const __m512 c1 = _mm512_set1_ps(cv[1]);
            const __m512 c2 = _mm512_set1_ps(cv[2]);
            const __m512 c3 = _mm512_set1_ps(cv[3]);
            for (; i + 16 <= end; i += 16) {
                __m512 sum = _mm512_mul_ps(_mm512_loadu_ps(xp + i), p);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(rp[0] + i), c0, sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(rp[1] + i), c1, sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(rp[2] + i), c2, sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(rp[3] + i), c3, sum);
                _mm512_storeu_ps(out + i, sum);
            }
#endif
            for (; i < end; ++i) {
                float sum = xp[i] * pv;
                for (int j = 0; j < 4; ++j) {
                    sum = std::fma(rp[j][i], cv[j], sum);
                }
                out[i] = sum;
            }
            ir = row * n_embd + end;
        }
        return;
    }

'''
s=s[:insert]+body+s[insert:]
p.write_text(s)
patch=''.join(difflib.unified_diff(backup.read_text().splitlines(True),s.splitlines(True),fromfile='a/ggml/src/ggml-cpu/ops.cpp',tofile='b/ggml/src/ggml-cpu/ops.cpp'))
(root/'serving/fleet-0903/glm-hc-post-vector.patch').write_text(patch)
print('Added opt-in HC_POST vector path')
