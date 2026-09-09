#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

enum flash_rms_guard_status {
    FLASH_RMS_GUARD_OK,
    FLASH_RMS_GUARD_LENGTH,
    FLASH_RMS_GUARD_ENVIRONMENT,
    FLASH_RMS_GUARD_NONFINITE,
    FLASH_RMS_GUARD_BOUNDARY,
    FLASH_RMS_GUARD_UNSUPPORTED,
    FLASH_RMS_GUARD_STATUS_COUNT
};

static inline flash_rms_guard_status flash_rms_guarded_mean(const float * x, int64_t n, float * mean) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (n < 1024 || n > (int64_t(1) << 24)) return FLASH_RMS_GUARD_LENGTH;
    const unsigned csr = _mm_getcsr();
    if ((csr & _MM_ROUND_MASK) != _MM_ROUND_NEAREST || (csr & 0x1f80) != 0x1f80) return FLASH_RMS_GUARD_ENVIRONMENT;
    __m512d lo = _mm512_setzero_pd(), hi = lo;
    const __m512 infinity = _mm512_set1_ps(INFINITY);
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 value = _mm512_loadu_ps(x + i);
        const __m512 square = _mm512_mul_ps(value, value);
        if (_mm512_cmp_ps_mask(square, infinity, _CMP_LT_OQ) != 0xffff) return FLASH_RMS_GUARD_NONFINITE;
        lo = _mm512_add_pd(lo, _mm512_cvtps_pd(_mm512_castps512_ps256(square)));
        hi = _mm512_add_pd(hi, _mm512_cvtps_pd(_mm512_extractf32x8_ps(square, 1)));
    }
    alignas(64) double lanes[16];
    _mm512_store_pd(lanes, lo); _mm512_store_pd(lanes + 8, hi);
    double sum = 0.0;
    for (double lane : lanes) sum += lane;
    for (; i < n; ++i) {
        const float square = x[i] * x[i];
        if (!std::isfinite(square)) return FLASH_RMS_GUARD_NONFINITE;
        sum += double(square);
    }
    if (sum == 0.0) {
        *mean = 0.0f;
        return FLASH_RMS_GUARD_OK;
    }
    // Both positive summation trees have relative error below gamma_n.
    // This wider interval includes the serial sum and outward-rounded division.
    const double bound = std::nextafter((4.0 * double(n) * DBL_EPSILON) * sum, INFINITY);
    const double lower_sum = std::nextafter(sum - bound, -INFINITY);
    const double upper_sum = std::nextafter(sum + bound, INFINITY);
    const float lower = float(std::nextafter(lower_sum / double(n), -INFINITY));
    const float upper = float(std::nextafter(upper_sum / double(n), INFINITY));
    uint32_t lower_bits, upper_bits;
    std::memcpy(&lower_bits, &lower, sizeof(lower));
    std::memcpy(&upper_bits, &upper, sizeof(upper));
    if (lower_bits != upper_bits) return FLASH_RMS_GUARD_BOUNDARY;
    *mean = lower;
    return FLASH_RMS_GUARD_OK;
#else
    (void) x; (void) n; (void) mean;
    return FLASH_RMS_GUARD_UNSUPPORTED;
#endif
}
