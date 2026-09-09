#!/usr/bin/env python3
from pathlib import Path
import shutil
import difflib

root=Path(__file__).resolve().parents[2]
p=root/'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
b=p.with_name(p.name+'.before-goal-q5-bytes')
assert not b.exists()
shutil.copy2(p,b)
s=p.read_text()
body=r'''
struct block_q5_K_x16_bytes {
    ggml_half d[16], dmin[16];
    uint8_t scales[8][16], mins[8][16];
    uint8_t qs[8][8][64];
};
static_assert(sizeof(block_q5_K_x16_bytes) == 4416, "wrong byte-expanded Q5 block size");

static void ggml_x16_pack_q5_K_bytes(const block_q5_K * in, int nb, block_q5_K_x16_bytes * out) {
    std::vector<block_q5_K_x16> compact(nb);
    ggml_x16_pack_q5_K(in, nb, compact.data());
    for (int b = 0; b < nb; ++b) {
        memcpy(out[b].d, compact[b].d, sizeof(out[b].d));
        memcpy(out[b].dmin, compact[b].dmin, sizeof(out[b].dmin));
        memcpy(out[b].scales, compact[b].scales, sizeof(out[b].scales));
        memcpy(out[b].mins, compact[b].mins, sizeof(out[b].mins));
        for (int sb = 0; sb < 8; ++sb) for (int chunk = 0; chunk < 8; ++chunk) {
            const uint8_t * q = compact[b].qsh + chunk * 320;
            for (int i = 0; i < 64; ++i) {
                const uint8_t lo = (q[(sb / 2) * 64 + i] >> (4 * (sb & 1))) & 15;
                out[b].qs[sb][chunk][i] = lo | (((q[256 + i] >> sb) & 1) << 4);
            }
        }
    }
}

static void ggml_gemv_q5_K_x16_bytes_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0 && nr >= 1 && nr <= 2);
    const int nb = n / QK_K;
    const auto * weights = (const block_q5_K_x16_bytes *) vx;
    const auto * activations = (const block_q8_K *) vy;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    for (int g = 0; g < nc / 16; ++g) {
        const block_q5_K_x16_bytes * w = weights + g * nb;
        __m512 sum[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int b = 0; b < nb; ++b) {
            __m512i isum[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
            __m512i imin[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
            for (int sb = 0; sb < 8; ++sb) {
                __m512i dots[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
                for (int chunk = 0; chunk < 8; ++chunk) {
                    const __m512i q = _mm512_loadu_si512(w[b].qs[sb][chunk]);
                    for (int y = 0; y < nr; ++y) {
                        int32_t aq;
                        memcpy(&aq, activations[y * nb + b].qs + sb * 32 + chunk * 4, sizeof(aq));
                        dots[y] = _mm512_dpbusd_epi32(dots[y], q, _mm512_set1_epi32(aq));
                    }
                }
                const __m512i scale = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].scales[sb]));
                const __m512i min = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].mins[sb]));
                for (int y = 0; y < nr; ++y) {
                    const block_q8_K & a = activations[y * nb + b];
                    isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dots[y], scale));
                    imin[y] = _mm512_add_epi32(imin[y], _mm512_mullo_epi32(min, _mm512_set1_epi32(int(a.bsums[2 * sb]) + a.bsums[2 * sb + 1])));
                }
            }
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
            const __m512 dmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].dmin));
            for (int y = 0; y < nr; ++y) {
                const __m512 ad = _mm512_set1_ps(activations[y * nb + b].d);
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[y]), _mm512_mul_ps(d, ad), sum[y]);
                sum[y] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin[y]), _mm512_mul_ps(dmin, ad), sum[y]);
            }
        }
        for (int y = 0; y < nr; ++y) _mm512_storeu_ps(s + y * bs + g * 16, sum[y]);
    }
#else
    for (int g = 0; g < nc / 16; ++g) for (int y = 0; y < nr; ++y) {
        const block_q5_K_x16_bytes * w = weights + g * nb;
        const block_q8_K * a = activations + y * nb;
        for (int row = 0; row < 16; ++row) {
            float sum = 0;
            for (int b = 0; b < nb; ++b) {
                int32_t isum = 0, imin = 0;
                for (int sb = 0; sb < 8; ++sb) {
                    int32_t dot = 0;
                    for (int k = 0; k < 32; ++k) dot += int(w[b].qs[sb][k / 4][row * 4 + k % 4]) * int(a[b].qs[sb * 32 + k]);
                    isum += dot * w[b].scales[sb][row];
                    imin += int(w[b].mins[sb][row]) * (int(a[b].bsums[2 * sb]) + a[b].bsums[2 * sb + 1]);
                }
                sum = std::fma(float(isum), GGML_FP16_TO_FP32(w[b].d[row]) * a[b].d, sum);
                sum = std::fma(-float(imin), GGML_FP16_TO_FP32(w[b].dmin[row]) * a[b].d, sum);
            }
            s[y * bs + g * 16 + row] = sum;
        }
    }
#endif
}

'''
anchor='static void ggml_x16_pack_q6_K('
assert s.count(anchor)==1
s=s.replace(anchor,body+anchor)
anchor='        case GGML_TYPE_Q5_K: ggml_x16_pack_q5_K((const block_q5_K *) rows, nb, (block_q5_K_x16 *) dst); break;'
assert s.count(anchor)==1
s=s.replace(anchor,'''        case GGML_TYPE_Q5_K:
            if (sp.x16_bytes == sizeof(block_q5_K_x16_bytes)) {
                ggml_x16_pack_q5_K_bytes((const block_q5_K *) rows, nb, (block_q5_K_x16_bytes *) dst);
            } else {
                ggml_x16_pack_q5_K((const block_q5_K *) rows, nb, (block_q5_K_x16 *) dst);
            }
            break;''')
anchor='    static const tensor_traits_x16 t_q4_K(spec_q4_K), t_q5_K(spec_q5_K), t_q6_K(spec_q6_K), t_q8_0(spec_q8_0);'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'''
    static const x16_spec spec_q5_bytes = { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, QK_K, sizeof(block_q5_K_x16_bytes), GGML_TYPE_Q8_K, ggml_gemv_q5_K_x16_bytes_q8_K, "x16 q5_K bytes" };
    static const tensor_traits_x16 t_q5_bytes(spec_q5_bytes);
    static const bool q5_bytes = []() {
        const char * value = getenv("GGML_CPU_X16_Q5_BYTES");
        return value != nullptr && atoi(value) != 0;
    }();''')
anchor='    if (cur->type == GGML_TYPE_Q5_K && x16_q5_K && cur->ne[0] % QK_K == 0) return &t_q5_K;'
assert s.count(anchor)==1
s=s.replace(anchor,'    if (cur->type == GGML_TYPE_Q5_K && x16_q5_K && cur->ne[0] % QK_K == 0) return q5_bytes ? &t_q5_bytes : &t_q5_K;')
p.write_text(s)
patch=''.join(difflib.unified_diff(b.read_text().splitlines(True),s.splitlines(True),fromfile='a/ggml/src/ggml-cpu/repack.cpp',tofile='b/ggml/src/ggml-cpu/repack.cpp'))
(root/'serving/fleet-0903/glm-q5-byte-layout.patch').write_text(patch)
print('Added opt-in lossless byte-expanded Q5 layout')
