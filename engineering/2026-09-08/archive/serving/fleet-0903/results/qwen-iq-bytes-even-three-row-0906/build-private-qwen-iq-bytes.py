#!/usr/bin/env python3
"""Build exact byte-expanded IQ r16 weights in a private CPU library."""
import importlib.util
from pathlib import Path


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:120]
    return source.replace(old, new)


def transform(source):
    assert 'GGML_CPU_IQ_R16_BYTES' not in source
    source = replace_once(source, '''struct block_iq_r16 {
    ggml_half d[16];
    uint8_t scales[(QK_K / 32) * 16];
    uint8_t qs[(QK_K / 16) * 4 * 32];
};
static_assert(sizeof(block_iq_r16) == 2208, "wrong block_iq_r16 size/padding");
''', '''template <bool BYTES> struct block_iq_r16_layout {
    ggml_half d[16];
    uint8_t scales[(QK_K / 32) * 16];
    uint8_t qs[(QK_K / 16) * 4 * (BYTES ? 64 : 32)];
};
using block_iq_r16 = block_iq_r16_layout<false>;
using block_iq_r16_bytes = block_iq_r16_layout<true>;
static_assert(sizeof(block_iq_r16) == 2208, "wrong block_iq_r16 size/padding");
static_assert(sizeof(block_iq_r16_bytes) == 4256, "wrong block_iq_r16_bytes size/padding");

static bool iq_r16_bytes_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_BYTES");
        return value && atoi(value) != 0;
    }();
    return enabled;
}
''')
    start = source.index('template <typename BLOC_TYPE>\nstatic int repack_iq_to_r16(')
    end = source.index('template <bool IQ3, bool NIBBLE2>', start)
    original = source[start:end]
    changed = original.replace('template <typename BLOC_TYPE>', 'template <typename BLOC_TYPE, bool BYTES>')
    changed = changed.replace('static int repack_iq_to_r16(', 'static int repack_iq_to_r16_impl(')
    changed = changed.replace('block_iq_r16', 'block_iq_r16_layout<BYTES>')
    changed = replace_once(changed, '    const BLOC_TYPE * src = (const BLOC_TYPE *) data;', '''    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const uint8_t * values = std::is_same_v<BLOC_TYPE, block_iq3_xxs> ? values3 : values2;
    const BLOC_TYPE * src = (const BLOC_TYPE *) data;''')
    changed = replace_once(changed, '                            if (iq_r16_paired_nibbles()) {', '''                            if constexpr (BYTES) {
                                const int idx = (sb * 4 + k / 4) * 64 + row * 4 + k % 4;
                                out.qs[idx] = values[r8.qs[from] & 15];
                                out.qs[idx + 1] = values[r8.qs[from] >> 4];
                            } else if (iq_r16_paired_nibbles()) {''')
    changed += '''template <typename BLOC_TYPE>
static int repack_iq_to_r16(ggml_tensor * t, const void * data, size_t data_size) {
    if (iq_r16_bytes_enabled()) {
        static std::atomic<bool> logged { false };
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_INFO("IQ_R16_BYTES block=4256 type=%s\\n", ggml_type_name(t->type));
        }
        return repack_iq_to_r16_impl<BLOC_TYPE, true>(t, data, data_size);
    }
    return repack_iq_to_r16_impl<BLOC_TYPE, false>(t, data, data_size);
}

'''
    source = source[:start] + changed + source[end:]
    start = source.index('template <bool IQ3, bool NIBBLE2>')
    end = source.index('template <bool IQ3>\nstatic void ggml_gemv_iq_r16_q8_K(', start)
    changed = source[start:end].replace('template <bool IQ3, bool NIBBLE2>', 'template <bool IQ3, bool NIBBLE2, bool BYTES = false>')
    changed = changed.replace('block_iq_r16', 'block_iq_r16_layout<BYTES>')
    changed = replace_once(changed, '                    if constexpr (NIBBLE2) {', '''                    if constexpr (BYTES) {
                        int32_t aq[4];
                        memcpy(aq, a[b].qs + sb * 16, sizeof(aq));
                        const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_loadu_si512(w[b].qs + (sb * 4 + 0) * 64), _mm512_set1_epi32(aq[0]));
                        const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_loadu_si512(w[b].qs + (sb * 4 + 1) * 64), _mm512_set1_epi32(aq[1]));
                        const __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_loadu_si512(w[b].qs + (sb * 4 + 2) * 64), _mm512_set1_epi32(aq[2]));
                        const __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_loadu_si512(w[b].qs + (sb * 4 + 3) * 64), _mm512_set1_epi32(aq[3]));
                        dots = _mm512_add_epi32(_mm512_add_epi32(d0, d1), _mm512_add_epi32(d2, d3));
                    } else if constexpr (NIBBLE2) {''')
    changed = replace_once(changed, '''                            const int idx = NIBBLE2 ? (sb * 2 + k / 8) * 64 + row * 4 + k % 4
                                                    : (sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2;
                            const int shift = NIBBLE2 ? 4 * ((k / 4) % 2) : 4 * (k % 2);
                            const uint8_t code = (w[b].qs[idx] >> shift) & 15;
                            dots += (int(values[code]) - 64) * int(a[b].qs[sb * 16 + k]);''', '''                            int value;
                            if constexpr (BYTES) {
                                value = w[b].qs[(sb * 4 + k / 4) * 64 + row * 4 + k % 4];
                            } else {
                                const int idx = NIBBLE2 ? (sb * 2 + k / 8) * 64 + row * 4 + k % 4
                                                        : (sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2;
                                const int shift = NIBBLE2 ? 4 * ((k / 4) % 2) : 4 * (k % 2);
                                value = values[(w[b].qs[idx] >> shift) & 15];
                            }
                            dots += (value - 64) * int(a[b].qs[sb * 16 + k]);''')
    source = source[:start] + changed + source[end:]
    source = replace_once(source, '''    if (iq_r16_paired_nibbles()) {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, true>''', '''    if (iq_r16_bytes_enabled()) {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, true, true>(n, s, bs, vx, vy, nr, nc);
    } else if (iq_r16_paired_nibbles()) {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, true>''')
    start = source.index('template <bool IQ3, int NR>\nstatic void ggml_gemv_iq_r16_batch_impl(')
    end = source.index('template <bool IQ3>\nstatic void ggml_gemv_iq_r16_batch(', start)
    changed = source[start:end].replace('template <bool IQ3, int NR>', 'template <bool IQ3, int NR, bool BYTES = false>')
    changed = changed.replace('block_iq_r16', 'block_iq_r16_layout<BYTES>')
    changed = replace_once(changed, '''                const __m512i p0 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 0) * 64);
                const __m512i p1 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 1) * 64);
                const __m512i q0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, mask));
                const __m512i q1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), mask));
                const __m512i q2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, mask));
                const __m512i q3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), mask));''', '''                __m512i q0, q1, q2, q3;
                if constexpr (BYTES) {
                    q0 = _mm512_loadu_si512(w[b].qs + (sb * 4 + 0) * 64);
                    q1 = _mm512_loadu_si512(w[b].qs + (sb * 4 + 1) * 64);
                    q2 = _mm512_loadu_si512(w[b].qs + (sb * 4 + 2) * 64);
                    q3 = _mm512_loadu_si512(w[b].qs + (sb * 4 + 3) * 64);
                } else {
                    const __m512i p0 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 0) * 64);
                    const __m512i p1 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 1) * 64);
                    q0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, mask));
                    q1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), mask));
                    q2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, mask));
                    q3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), mask));
                }''')
    source = source[:start] + changed + source[end:]
    source = replace_once(source, '''    GGML_ASSERT(nr >= 1 && nr <= 3);
    if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3>''', '''    GGML_ASSERT(nr >= 1 && nr <= 3);
    if (iq_r16_bytes_enabled()) {
        if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3, true>(n, s, bs, vx, a, nc);
        else if (nr == 2) ggml_gemv_iq_r16_batch_impl<IQ3, 2, true>(n, s, bs, vx, a, nc);
        else ggml_gemv_iq_r16_q8_K<IQ3>(n, s, bs, vx, a[0], 1, nc);
        return;
    }
    if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3>''')
    return replace_once(source, '            return (t->ne[0] / QK_K) * sizeof(block_iq_r16);',
                        '            return (t->ne[0] / QK_K) * (iq_r16_bytes_enabled() ? sizeof(block_iq_r16_bytes) : sizeof(block_iq_r16));')


def build(engine, destination, run):
    # Reuse the pinned private compiler/linker path without editing its source.
    helper = Path(__file__).with_name('build-private-qwen-task-rows.py')
    spec = importlib.util.spec_from_file_location('private_iq_bytes_build_base', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.transform = transform

    def named_run(command, cwd, log_name):
        return run(command, cwd, log_name.replace('task-rows-', 'iq-bytes-'))

    info = module.build(engine, destination, named_run)
    destination = Path(destination)
    (destination / 'iq-expert-task-rows.patch').rename(destination / 'iq-r16-bytes.patch')
    return info
