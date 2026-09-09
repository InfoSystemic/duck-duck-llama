#!/usr/bin/env python3
"""Stage an opt-in shared-weight IQ kernel in the private GLM fork."""
import difflib
from pathlib import Path

base = Path(__file__).resolve().parent
root = base.parents[1]
path = root / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
old = path.read_text()
assert 'GGML_CPU_IQ_R16_BATCH3' not in old
backup = path.with_name(path.name + '.before-goal-iq-r16-batch3')
assert not backup.exists()
backup.write_text(old)

helper = r'''
static bool iq_r16_batch3_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_BATCH3");
        return value && atoi(value) != 0;
    }();
    return enabled && iq_r16_paired_nibbles();
}

template <bool IQ3, int NR>
static void ggml_gemv_iq_r16_batch_impl(
        int n, float * s, size_t bs, const void * vx, const block_q8_K * const * a, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0 && NR >= 2 && NR <= 3);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const int nb = n / QK_K;
    const auto * weights = (const block_iq_r16 *) vx;
    constexpr float factor = IQ3 ? 0.25f : 0.125f;
    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) (IQ3 ? values3 : values2)));
    const __m512i mask = _mm512_set1_epi8(15);
    for (int x = 0; x < nc / 16; ++x) {
        const block_iq_r16 * w = weights + x * nb;
        __m512 sum[NR];
        for (int y = 0; y < NR; ++y) sum[y] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i isum[NR];
            for (int y = 0; y < NR; ++y) isum[y] = _mm512_setzero_si512();
            for (int sb = 0; sb < QK_K / 16; ++sb) {
                const __m512i p0 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 0) * 64);
                const __m512i p1 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 1) * 64);
                const __m512i q0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, mask));
                const __m512i q1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), mask));
                const __m512i q2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, mask));
                const __m512i q3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), mask));
                __m512i scales = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                    (const __m128i *) (w[b].scales + (sb / 2) * 16)));
                if (sb & 1) scales = _mm512_srli_epi32(scales, 4);
                scales = _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(scales, _mm512_set1_epi32(15)), 1),
                    _mm512_set1_epi32(1));
                for (int y = 0; y < NR; ++y) {
                    int32_t aq[4];
                    memcpy(aq, a[y][b].qs + sb * 16, sizeof(aq));
                    const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q0, _mm512_set1_epi32(aq[0]));
                    const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q1, _mm512_set1_epi32(aq[1]));
                    const __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2, _mm512_set1_epi32(aq[2]));
                    const __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q3, _mm512_set1_epi32(aq[3]));
                    __m512i dots = _mm512_add_epi32(_mm512_add_epi32(d0, d1), _mm512_add_epi32(d2, d3));
                    dots = _mm512_sub_epi32(dots, _mm512_set1_epi32(64 * a[y][b].bsums[sb]));
                    isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dots, scales));
                }
            }
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
            for (int y = 0; y < NR; ++y) {
                const __m512 scale = _mm512_mul_ps(d, _mm512_set1_ps(a[y][b].d * factor));
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[y]), scale, sum[y]);
            }
        }
        for (int y = 0; y < NR; ++y) _mm512_storeu_ps(s + y * bs + x * 16, sum[y]);
    }
#else
    for (int y = 0; y < NR; ++y) {
        ggml_gemv_iq_r16_q8_K<IQ3>(n, s + y * bs, bs, vx, a[y], 1, nc);
    }
#endif
}

template <bool IQ3>
static void ggml_gemv_iq_r16_batch(
        int n, float * s, size_t bs, const void * vx, const block_q8_K * const * a, int nr, int nc) {
    GGML_ASSERT(nr >= 1 && nr <= 3);
    if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3>(n, s, bs, vx, a, nc);
    else if (nr == 2) ggml_gemv_iq_r16_batch_impl<IQ3, 2>(n, s, bs, vx, a, nc);
    else ggml_gemv_iq_r16_q8_K<IQ3>(n, s, bs, vx, a[0], 1, nc);
}

'''
needle = 'static void qK_r8_get_scale_min('
assert old.count(needle) == 1
new = old.replace(needle, helper + needle)

fused = r'''
            if constexpr (expanded_iq && NB_COLS == 16) {
                if (iq_r16_batch3_enabled() && n_rows >= 2) {
                    constexpr int64_t tile_size = 64;
                    alignas(64) float gate_tmp[3][tile_size], up_tmp[3][tile_size];
                    for (int64_t row = 0; row < n_rows; row += 3) {
                        const int nr = std::min<int64_t>(3, n_rows - row);
                        const block_q8_K * activations[3];
                        float * outputs[3];
                        for (int y = 0; y < nr; ++y) {
                            const row_mapping mapping = matrix_rows[expert * n_tokens + row + y];
                            activations[y] = (const block_q8_K *) (wdata +
                                mapping.i2 * src1_plane_size + (mapping.i1 % n_src_rows) * src1_row_size);
                            outputs[y] = (float *) ((char *) dst->data +
                                mapping.i2 * dst->nb[2] + mapping.i1 * dst->nb[1]);
                        }
                        for (int64_t tile_start = row_start; tile_start < row_end; tile_start += tile_size) {
                            const int64_t tile_rows = std::min(tile_size, row_end - tile_start);
                            ggml_gemv_iq_r16_batch<expanded_iq3_xxs>(k, gate_tmp[0], tile_size,
                                gate_cur + physical_row_offset(gate_w, tile_start), activations, nr, tile_rows);
                            ggml_gemv_iq_r16_batch<expanded_iq3_xxs>(k, up_tmp[0], tile_size,
                                up_cur + physical_row_offset(up_w, tile_start), activations, nr, tile_rows);
                            for (int y = 0; y < nr; ++y) {
                                ggml_vec_swiglu_f32(tile_rows, outputs[y] + tile_start, gate_tmp[y], up_tmp[y]);
                            }
                        }
                    }
                    continue;
                }
            }
'''
needle = '            for (int64_t row = 0; row < n_rows; ++row) {\n                const row_mapping mapping = matrix_rows[expert * n_tokens + row];'
assert new.count(needle) == 1
new = new.replace(needle, fused + needle)

ordinary = r'''
            if constexpr (expanded_iq && NB_COLS == 16) {
                if (iq_r16_batch3_enabled() && nr1 >= 2) {
                    constexpr int64_t tile_size = 64;
                    alignas(64) float tmp[3][tile_size];
                    for (int64_t row = 0; row < nr1; row += 3) {
                        const int nr = std::min<int64_t>(3, nr1 - row);
                        const block_q8_K * activations[3];
                        float * outputs[3];
                        for (int y = 0; y < nr; ++y) {
                            const mmid_row_mapping mapping = MMID_MATRIX_ROW(cur_a, row + y);
                            activations[y] = (const block_q8_K *) (wdata +
                                (mapping.i1 % ne11) * nbw1 + mapping.i2 * nbw2);
                            outputs[y] = (float *) ((char *) dst->data + mapping.i1 * nb1 + mapping.i2 * nb2);
                        }
                        for (int64_t tile_start = src0_cur_start; tile_start < src0_cur_end; tile_start += tile_size) {
                            const int64_t tile_rows = std::min(tile_size, src0_cur_end - tile_start);
                            ggml_gemv_iq_r16_batch<expanded_iq3_xxs>(ne00, tmp[0], tile_size,
                                src0_cur + physical_row_offset(src0, tile_start), activations, nr, tile_rows);
                            for (int y = 0; y < nr; ++y) {
                                memcpy(outputs[y] + tile_start, tmp[y], tile_rows * sizeof(float));
                            }
                        }
                    }
                    continue;
                }
            }
'''
needle = '            for (int ir1 = 0; ir1 < nr1; ir1++) {\n                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);'
assert new.count(needle) == 1
new = new.replace(needle, ordinary + needle)
path.write_text(new)
patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
    fromfile='a/ggml/src/ggml-cpu/repack.cpp', tofile='b/ggml/src/ggml-cpu/repack.cpp'))
(base / 'glm-iq-r16-batch3.patch').write_text(patch)
print('Staged IQ shared-weight batching:', len(new.splitlines()) - len(old.splitlines()), 'lines')
