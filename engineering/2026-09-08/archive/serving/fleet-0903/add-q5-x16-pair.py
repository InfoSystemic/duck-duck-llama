#!/usr/bin/env python3
from pathlib import Path
import shutil
import difflib

root=Path(__file__).resolve().parents[2]
engine=root/'engines/llama.cpp-glm5n-goal-0904'
p=engine/'ggml/src/ggml-cpu/arch/x86/repack.cpp'
backup=p.with_name(p.name+'.before-goal-q5-pair')
assert not backup.exists()
shutil.copy2(p,backup)
s=p.read_text()
body=r'''
static void ggml_gemv_q5_K_x16_pair_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0);
    const int nb = n / QK_K;
    const auto * weights = (const block_q5_K_x16 *) vx;
    const auto * activations = (const block_q8_K *) vy;
    const __m512i m4 = _mm512_set1_epi8(15);
    const __m512i m10 = _mm512_set1_epi8(16);
    for (int g = 0; g < nc / 16; ++g) {
        const block_q5_K_x16 * w = weights + g * nb;
        __m512 sum[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int b = 0; b < nb; ++b) {
            const block_q8_K * a0 = activations + b;
            const block_q8_K * a1 = activations + nb + b;
            __m512i dots[2][8];
            for (int y = 0; y < 2; ++y) for (int sb = 0; sb < 8; ++sb) dots[y][sb] = _mm512_setzero_si512();
            for (int chunk = 0; chunk < 8; ++chunk) {
                const uint8_t * q = w[b].qsh + chunk * 320;
                const __m512i h = _mm512_loadu_si512(q + 256);
#define Q5_PAIR_DOT(sb, weights) { \
                const __m512i v = (weights); \
                dots[0][sb] = _mm512_dpbusd_epi32(dots[0][sb], v, GGML_X16_BC32(a0->qs + 32*(sb) + 4*chunk)); \
                dots[1][sb] = _mm512_dpbusd_epi32(dots[1][sb], v, GGML_X16_BC32(a1->qs + 32*(sb) + 4*chunk)); }
                const __m512i v0 = _mm512_loadu_si512(q);
                Q5_PAIR_DOT(0, _mm512_or_si512(_mm512_and_si512(v0, m4), _mm512_and_si512(_mm512_slli_epi16(h, 4), m10)))
                Q5_PAIR_DOT(1, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v0, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 3), m10)))
                const __m512i v1 = _mm512_loadu_si512(q + 64);
                Q5_PAIR_DOT(2, _mm512_or_si512(_mm512_and_si512(v1, m4), _mm512_and_si512(_mm512_slli_epi16(h, 2), m10)))
                Q5_PAIR_DOT(3, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v1, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 1), m10)))
                const __m512i v2 = _mm512_loadu_si512(q + 128);
                Q5_PAIR_DOT(4, _mm512_or_si512(_mm512_and_si512(v2, m4), _mm512_and_si512(h, m10)))
                Q5_PAIR_DOT(5, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v2, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 1), m10)))
                const __m512i v3 = _mm512_loadu_si512(q + 192);
                Q5_PAIR_DOT(6, _mm512_or_si512(_mm512_and_si512(v3, m4), _mm512_and_si512(_mm512_srli_epi16(h, 2), m10)))
                Q5_PAIR_DOT(7, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v3, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 3), m10)))
#undef Q5_PAIR_DOT
            }
            for (int y = 0; y < 2; ++y) {
                const block_q8_K * a = y == 0 ? a0 : a1;
                __m512i isum = _mm512_setzero_si512(), imin = _mm512_setzero_si512();
                for (int sb = 0; sb < 8; ++sb) {
                    const __m512i scale = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].scales[sb]));
                    const __m512i min = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].mins[sb]));
                    isum = _mm512_add_epi32(isum, _mm512_mullo_epi32(dots[y][sb], scale));
                    imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(min, _mm512_set1_epi32(int(a->bsums[2*sb]) + a->bsums[2*sb+1])));
                }
                const __m512 ad = _mm512_set1_ps(a->d);
                const __m512 d = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d)), ad);
                const __m512 dmin = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].dmin)), ad);
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), d, sum[y]);
                sum[y] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin), dmin, sum[y]);
            }
        }
        _mm512_storeu_ps(s + g*16, sum[0]);
        _mm512_storeu_ps(s + bs + g*16, sum[1]);
    }
#else
    const size_t row_bytes = (n / QK_K) * sizeof(block_q8_K);
    ggml_gemv_q5_K_x16_q8_K_generic(n, s, 0, vx, vy, 1, nc);
    ggml_gemv_q5_K_x16_q8_K_generic(n, s + bs, 0, vx, (const char *) vy + row_bytes, 1, nc);
#endif
}

'''
anchor='void ggml_gemv_q5_K_x16_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {'
assert s.count(anchor)==1
s=s.replace(anchor,body+anchor+'''
    if (nr == 2) {
        ggml_gemv_q5_K_x16_pair_q8_K(n, s, bs, vx, vy, nc);
        return;
    }
''')
p.write_text(s)
changes=[(p,backup)]
p=engine/'ggml/src/ggml-cpu/repack.cpp'
backup=p.with_name(p.name+'.before-goal-q5-pair')
assert not backup.exists()
shutil.copy2(p,backup)
s=p.read_text()
start=s.index('class tensor_traits_x16 :')
start=s.index('    void forward_mul_mat(',start)
old='''                for (int64_t i11 = 0; i11 < ne11; i11++) {
                    const int64_t i = i12 * ne11 + i11;
                    float * out = (float *) ((char *) dst->data + i11 * dst->nb[1] + i12 * dst->nb[2]) + r0;
                    sp.gemv((int) k, out, 0, w, wdata + (size_t) i * row_bytes, 1, (int) (r1 - r0));
                }'''
at=s.index(old,start)
new='''                static const bool pair_enabled = []() {
                    const char * value = getenv("GGML_CPU_X16_Q5_BATCH2");
                    return value != nullptr && atoi(value) != 0;
                }();
                for (int64_t i11 = 0; i11 < ne11;) {
                    const int nr = pair_enabled && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;
                    const int64_t i = i12 * ne11 + i11;
                    float * out = (float *) ((char *) dst->data + i11 * dst->nb[1] + i12 * dst->nb[2]) + r0;
                    sp.gemv((int) k, out, dst->nb[1] / sizeof(float), w, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                    i11 += nr;
                }'''
s=s[:at]+s[at:].replace(old,new,1)
p.write_text(s)
changes.append((p,backup))
patch=''
for p,b in changes:
    rel=str(p.relative_to(engine))
    patch+=''.join(difflib.unified_diff(b.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(root/'serving/fleet-0903/glm-q5-x16-pair.patch').write_text(patch)
print('Added opt-in two-token Q5 x16 kernel')
