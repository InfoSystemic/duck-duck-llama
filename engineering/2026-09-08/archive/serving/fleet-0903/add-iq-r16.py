#!/usr/bin/env python3
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
path = root/'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
backup = path.with_name(path.name+'.before-goal-iq-r16')
assert not backup.exists()
shutil.copy2(path, backup)
s = path.read_text()
addition = r'''
// Four codes per output channel let each VNNI lane produce one row sum.
struct block_iq_r16 {
    ggml_half d[16];
    uint8_t scales[(QK_K / 32) * 16];
    uint8_t qs[(QK_K / 16) * 4 * 32];
};
static_assert(sizeof(block_iq_r16) == 2208, "wrong block_iq_r16 size/padding");

template <typename BLOC_TYPE>
static int repack_iq_to_r16(ggml_tensor * t, const void * data, size_t data_size) {
    const int64_t nrows = ggml_nrows(t);
    const int64_t nb = t->ne[0] / QK_K;
    GGML_ASSERT(data_size == nrows * nb * sizeof(BLOC_TYPE));
    if (t->ne[0] % QK_K != 0 || t->ne[1] % 16 != 0) {
        return -1;
    }
    const BLOC_TYPE * src = (const BLOC_TYPE *) data;
    block_iq_r16 * dst = (block_iq_r16 *) t->data;
    parallel_repack_groups(nrows / 16, [&](int64_t group) {
        for (int64_t b = 0; b < nb; ++b) {
            block_iq_r16 & out = dst[group * nb + b];
            memset(&out, 0, sizeof(out));
            for (int half = 0; half < 2; ++half) {
                BLOC_TYPE rows[8];
                for (int r = 0; r < 8; ++r) {
                    rows[r] = src[(group * 16 + half * 8 + r) * nb + b];
                }
                const auto r8 = [&]() {
                    if constexpr (std::is_same_v<BLOC_TYPE, block_iq3_xxs>) {
                        return make_block_iq3_xxs_r8(rows);
                    } else {
                        return make_block_iq2_r8(rows);
                    }
                }();
                for (int r = 0; r < 8; ++r) {
                    const int row = half * 8 + r;
                    out.d[row] = r8.d[r];
                    for (int sb = 0; sb < QK_K / 32; ++sb) {
                        if constexpr (std::is_same_v<BLOC_TYPE, block_iq3_xxs>) {
                            const uint8_t scale = (r8.scales[sb] >> (4 * r)) & 15;
                            out.scales[sb * 16 + row] = scale | (scale << 4);
                        } else {
                            out.scales[sb * 16 + row] = r8.scales[sb * 8 + r];
                        }
                    }
                    for (int sb = 0; sb < QK_K / 16; ++sb) {
                        for (int k = 0; k < 16; k += 2) {
                            const int from = (sb * 2 + r / 4) * 32 + (r % 4) * 8 + k / 2;
                            const int to = (sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2;
                            out.qs[to] = r8.qs[from];
                        }
                    }
                }
            }
        }
    });
    return 0;
}

template <bool IQ3>
static void ggml_gemv_iq_r16_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0);
    const int nb = n / QK_K;
    const auto * weights = (const block_iq_r16 *) vx;
    const auto * activations = (const block_q8_K *) vy;
    constexpr float factor = IQ3 ? 0.25f : 0.125f;
    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const uint8_t * values = IQ3 ? values3 : values2;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) values));
    const __m512i lo_mask = _mm512_set1_epi16(15);
    const __m512i hi_mask = _mm512_set1_epi16(0x0f00);
    const __m512i scale_mask = _mm512_set1_epi32(15);
    const __m512i one = _mm512_set1_epi32(1);
    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a = activations + y * nb;
        for (int x = 0; x < nc / 16; ++x) {
            const block_iq_r16 * w = weights + x * nb;
            __m512 sum = _mm512_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m512i isum = _mm512_setzero_si512();
                for (int sb = 0; sb < QK_K / 16; ++sb) {
                    __m512i dots = _mm512_setzero_si512();
                    for (int chunk = 0; chunk < 4; ++chunk) {
                        const __m512i packed = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                            (const __m256i *) (w[b].qs + (sb * 4 + chunk) * 32)));
                        const __m512i codes = _mm512_or_si512(
                            _mm512_and_si512(packed, lo_mask),
                            _mm512_and_si512(_mm512_slli_epi16(packed, 4), hi_mask));
                        int32_t aq;
                        memcpy(&aq, a[b].qs + sb * 16 + chunk * 4, sizeof(aq));
                        dots = _mm512_dpbusd_epi32(dots, _mm512_shuffle_epi8(lut, codes), _mm512_set1_epi32(aq));
                    }
                    dots = _mm512_sub_epi32(dots, _mm512_set1_epi32(64 * a[b].bsums[sb]));
                    __m512i scales = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                        (const __m128i *) (w[b].scales + (sb / 2) * 16)));
                    if (sb & 1) {
                        scales = _mm512_srli_epi32(scales, 4);
                    }
                    scales = _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(scales, scale_mask), 1), one);
                    isum = _mm512_add_epi32(isum, _mm512_mullo_epi32(dots, scales));
                }
                const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
                const __m512 scale = _mm512_mul_ps(d, _mm512_set1_ps(a[b].d * factor));
                sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), scale, sum);
            }
            _mm512_storeu_ps(s + y * bs + x * 16, sum);
        }
    }
#else
    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a = activations + y * nb;
        for (int x = 0; x < nc / 16; ++x) {
            const block_iq_r16 * w = weights + x * nb;
            for (int row = 0; row < 16; ++row) {
                float sum = 0.0f;
                for (int b = 0; b < nb; ++b) {
                    int32_t isum = 0;
                    for (int sb = 0; sb < QK_K / 16; ++sb) {
                        int32_t dots = 0;
                        for (int k = 0; k < 16; ++k) {
                            const uint8_t packed = w[b].qs[(sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2];
                            const uint8_t code = (packed >> (4 * (k % 2))) & 15;
                            dots += (int(values[code]) - 64) * int(a[b].qs[sb * 16 + k]);
                        }
                        const int scale = (w[b].scales[(sb / 2) * 16 + row] >> (4 * (sb % 2))) & 15;
                        isum += dots * (2 * scale + 1);
                    }
                    sum = std::fma(float(isum), GGML_FP16_TO_FP32(w[b].d[row]) * (a[b].d * factor), sum);
                }
                s[y * bs + x * 16 + row] = sum;
            }
        }
    }
#endif
}

'''
anchor = 'static void qK_r8_get_scale_min('
assert s.count(anchor) == 1
s = s.replace(anchor, addition+anchor)
for name in ['iq2_xs', 'iq2_xxs', 'iq3_xxs']:
    anchor = f'template <> int repack<block_{name}, 8, 8>'
    addition = f'''template <> int repack<block_{name}, 8, 16>(ggml_tensor * t, const void * data, size_t data_size) {{
    return repack_iq_to_r16<block_{name}>(t, data, data_size);
}}

'''
    assert s.count(anchor) == 1
    s = s.replace(anchor, addition+anchor)
    for function in ['gemv', 'gemm']:
        anchor = f'{function}<block_{name}, 8, 8, GGML_TYPE_Q8_K>'
        idx = s.index(anchor)
        idx = s.rindex('template <>', 0, idx)
        iq3 = 'true' if name == 'iq3_xxs' else 'false'
        addition = f'''template <> void {function}<block_{name}, 8, 16, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {{
    ggml_gemv_iq_r16_q8_K<{iq3}>(n, s, bs, vx, vy, nr, nc);
}}

'''
        s = s[:idx]+addition+s[idx:]
s = s.replace('INTER_SIZE == 8 && NB_COLS == 8;\n    static constexpr bool expanded_iq3_xxs', 'INTER_SIZE == 8 && (NB_COLS == 8 || NB_COLS == 16);\n    static constexpr bool expanded_iq3_xxs')
s = s.replace('std::is_same_v<BLOC_TYPE, block_iq3_xxs> && INTER_SIZE == 8 && NB_COLS == 8;', 'std::is_same_v<BLOC_TYPE, block_iq3_xxs> && INTER_SIZE == 8 && (NB_COLS == 8 || NB_COLS == 16);')
s = s.replace('if constexpr (expanded_iq2_xs) {\n            GGML_ASSERT', 'if constexpr (expanded_iq && NB_COLS == 16) {\n            return (t->ne[0] / QK_K) * sizeof(block_iq_r16);\n        } else if constexpr (expanded_iq2_xs) {\n            GGML_ASSERT')
for name in ['iq2_xs', 'iq2_xxs', 'iq3_xxs']:
    anchor = f'    static const ggml::cpu::repack::tensor_traits<block_{name}, 8, 8, GGML_TYPE_Q8_K> {name}_r8_q8_K;'
    s = s.replace(anchor, anchor+f'\n    static const ggml::cpu::repack::tensor_traits<block_{name}, 8, 16, GGML_TYPE_Q8_K> {name}_r16_q8_K;')
    anchor = f'            return &{name}_r8_q8_K;'
    s = s.replace(anchor, f'            if (iq_r16_enabled && cur->ne[1] % 16 == 0) {{\n                return &{name}_r16_q8_K;\n            }}\n'+anchor)
anchor = 'static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {'
s = s.replace(anchor, anchor+'''
    static const bool iq_r16_enabled = []() {
        const char * value = getenv("GGML_CPU_IQ_R16_REPACK");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
''')
path.write_text(s)
print('Updated', path)
