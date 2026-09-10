// SPDX-License-Identifier: MIT
// Reference equations: pinned DeepSeek inference/kernel.py and model.py Engram.
// Native checkpoint codes/scales stay unchanged. No complete model graph here.
#include "deepseek-v41-engram-project-0910.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <vector>

namespace {
struct CsrGuard {
    unsigned saved = _mm_getcsr();
    CsrGuard() { _mm_setcsr(saved & ~(0x8040U | 0x6000U)); }
    ~CsrGuard() { _mm_setcsr(saved); }
};

float from_bf16(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0;
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

bool finite_bf16(const uint16_t * p, uint64_t n) {
    if (!p) return false;
    for (uint64_t i = 0; i < n; ++i) if ((p[i] & 0x7f80U) == 0x7f80U) return false;
    return true;
}

struct Tables {
    alignas(64) std::array<float, 256> fp8{};
    std::array<float, 255> scale{};
    Tables() {
        CsrGuard guard;
        for (unsigned code = 0; code < 256; ++code) {
            unsigned magnitude = code & 127, exp = magnitude >> 3, mant = magnitude & 7;
            float v = magnitude == 127 ? NAN : exp == 0 ? std::ldexp(float(mant), -9) :
                std::ldexp(float(8 + mant), int(exp) - 10);
            fp8[code] = code & 128 ? -v : v;
        }
        for (unsigned code = 0; code < 255; ++code) scale[code] = std::ldexp(1.0f, int(code) - 127);
    }
};

const Tables & tables() { static const Tables t; return t; }

uint8_t fp8_rne(float x) {
    const auto & values = tables().fp8;
    const uint8_t sign = std::signbit(x) ? 128 : 0;
    float value = std::min(std::abs(x), 448.0f);
    unsigned lo = 0, hi = 126;
    while (lo < hi) {
        const unsigned mid = (lo + hi) / 2;
        if (values[mid] < value) lo = mid + 1; else hi = mid;
    }
    if (lo && (value - values[lo - 1] < values[lo] - value ||
            (value - values[lo - 1] == values[lo] - value && (lo & 1U)))) --lo;
    return uint8_t(lo) | sign;
}

void quantize(const uint16_t * input, uint64_t elements, uint8_t * output, uint8_t * scales) {
    const auto & tab = tables();
    for (uint64_t begin = 0; begin < elements; begin += 32) {
        float amax = 1e-4f;
        for (unsigned j = 0; j < 32; ++j) amax = std::max(amax, std::abs(from_bf16(input[begin + j])));
        const float ratio = amax * (1.0f / 448.0f);
        uint32_t bits;
        std::memcpy(&bits, &ratio, sizeof(bits));
        const unsigned code = ((bits >> 23) & 255U) + ((bits & 0x7fffffU) != 0);
        scales[begin / 32] = uint8_t(code);
        const float factor = tab.scale[code];
        for (unsigned j = 0; j < 32; ++j) output[begin + j] = fp8_rne(from_bf16(input[begin + j]) / factor);
    }
}

// An explicit tree makes both CPU modes deterministic and comparable. It is
// not a claim about the undocumented reduction order of GPU tensor cores.
float reduce16(float * v) {
    for (int half = 8; half; half /= 2) for (int j = 0; j < half; ++j) v[j] += v[j + half];
    return v[0];
}

float scalar_dot32(const float * x, const uint8_t * weight) {
    const auto & tab = tables();
    float lanes[16];
    for (int j = 0; j < 16; ++j) lanes[j] = x[j] * tab.fp8[weight[j]] + x[j + 16] * tab.fp8[weight[j + 16]];
    return reduce16(lanes);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
float vector_dot32(const float * x, const uint8_t * weight) {
    const auto & tab = tables();
    __m512i i0 = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(weight)));
    __m512i i1 = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(weight + 16)));
    __m512 a = _mm512_mul_ps(_mm512_loadu_ps(x), _mm512_i32gather_ps(i0, tab.fp8.data(), 4));
    __m512 b = _mm512_mul_ps(_mm512_loadu_ps(x + 16), _mm512_i32gather_ps(i1, tab.fp8.data(), 4));
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, _mm512_add_ps(a, b));
    return reduce16(lanes);
}

struct Project {
    deepseek_v41_project_config c;
    int64_t max_chunk;
    int vectorize, workers;
    std::vector<float> norm, act, raw, gates;
    std::vector<uint8_t> aq, as;
    std::vector<uint16_t> kv, result;
};

bool has_simd() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl");
}

void apply_gate(Project & p, const uint16_t * hidden, const uint8_t * mask, int64_t tokens) {
    const auto & c = p.c;
    const int64_t width = c.hc_mult * c.dim, projected_width = width + c.dim;
    const float dim_inverse = 1.0f / float(c.dim), dot_scale = 1.0f / std::sqrt(float(c.dim));
    for (int64_t t = 0; t < tokens; ++t) for (int64_t h = 0; h < c.hc_mult; ++h) {
        const int64_t offset = t * width + h * c.dim;
        const uint16_t * key = p.kv.data() + t * projected_width + h * c.dim;
        const uint16_t * value = p.kv.data() + t * projected_width + width;
        float hs[16]{}, ks[16]{}, ds[16]{};
        float gate = 0.0f;
        if (!mask || mask[t]) {
            for (int64_t j = 0; j < c.dim; ++j) {
                const float x = from_bf16(hidden[offset + j]), k = from_bf16(key[j]);
                hs[j % 16] += x * x;
                ks[j % 16] += k * k;
                ds[j % 16] += (x * p.norm[h * c.dim + j]) * k;
            }
            const float hvar = reduce16(hs) * dim_inverse + c.norm_eps;
            const float kvar = reduce16(ks) * dim_inverse + c.norm_eps;
            const float rstd = (1.0f / std::sqrt(hvar)) * (1.0f / std::sqrt(kvar));
            const float dot = (reduce16(ds) * rstd) * dot_scale;
            const float z = std::copysign(std::sqrt(std::max(std::abs(dot), 1e-6f)), dot);
            gate = 1.0f / (1.0f + std::exp(-z));
        }
        p.gates[t * c.hc_mult + h] = gate;
        for (int64_t j = 0; j < c.dim; ++j) {
            float result = from_bf16(hidden[offset + j]);
            if (!mask || mask[t]) result += gate * from_bf16(value[j]);
            p.raw[offset + j] = result;
            p.result[offset + j] = mask && !mask[t] ? hidden[offset + j] : to_bf16(result);
        }
    }
}
}  // namespace

extern "C" int deepseek_v41_act_quant(const uint16_t * input, int64_t tokens, int64_t width,
    uint8_t * output, uint64_t output_elements, uint8_t * scales, uint64_t scale_elements) {
    if (tokens < 0 || tokens > 512 || width <= 0 || width > 6144 || width % 32) return -1;
    const uint64_t n = uint64_t(tokens) * uint64_t(width);
    if (output_elements < n || scale_elements < n / 32) return -1;
    if (!n) return 0;
    if (!output || !scales || !finite_bf16(input, n)) return -1;
    CsrGuard guard;
    quantize(input, n, output, scales);
    return 0;
}

extern "C" void * deepseek_v41_project_create(const deepseek_v41_project_config * config,
    int64_t max_chunk, int vectorize, int workers) {
    if (!config || max_chunk < 1 || max_chunk > 512 || (vectorize != 0 && vectorize != 1) ||
            workers < 1 || workers > 60 || (vectorize && !has_simd())) return nullptr;
    const auto & c = *config;
    if (c.input_dim < 32 || c.input_dim > 6144 || c.input_dim % 32 ||
            c.dim < 32 || c.dim > 5120 || c.dim % 32 || c.hc_mult < 1 || c.hc_mult > 4 ||
            !std::isfinite(c.norm_eps) || c.norm_eps <= 0) return nullptr;
    const uint64_t n = uint64_t(c.dim) * uint64_t(c.hc_mult + 1), k = uint64_t(c.input_dim);
    if (!c.weight || !c.scale || c.weight_bytes != n * k || c.scale_bytes != n * k / 1024 ||
            c.norm_elements != uint64_t(c.hc_mult * c.dim) ||
            !finite_bf16(c.q_weight, c.norm_elements) || !finite_bf16(c.k_weight, c.norm_elements)) return nullptr;
    for (uint64_t i = 0; i < c.weight_bytes; ++i) if ((c.weight[i] & 127U) == 127U) return nullptr;
    for (uint64_t i = 0; i < c.scale_bytes; ++i) if (c.scale[i] == 255) return nullptr;
    try {
        CsrGuard guard;
        auto p = std::make_unique<Project>();
        p->c = c; p->max_chunk = max_chunk; p->vectorize = vectorize; p->workers = workers;
        p->norm.resize(c.norm_elements);
        for (uint64_t i = 0; i < c.norm_elements; ++i) {
            p->norm[i] = from_bf16(c.q_weight[i]) * from_bf16(c.k_weight[i]);
            if (!std::isfinite(p->norm[i])) return nullptr;
        }
        p->c.q_weight = p->c.k_weight = nullptr;
        p->aq.resize(max_chunk * k); p->as.resize(max_chunk * k / 32); p->act.resize(max_chunk * k);
        p->kv.resize(max_chunk * n); p->result.resize(max_chunk * c.norm_elements);
        p->raw.resize(p->result.size()); p->gates.resize(max_chunk * c.hc_mult);
        tables();
        return p.release();
    } catch (...) { return nullptr; }
}

extern "C" void deepseek_v41_project_destroy(void * handle) { delete static_cast<Project *>(handle); }

extern "C" int deepseek_v41_project_apply(void * handle, const uint16_t * lookup,
    const uint16_t * hidden, const uint8_t * mask, int64_t tokens,
    uint16_t * output, uint64_t output_elements, uint16_t * projected, uint64_t projected_elements,
    float * raw_output, uint64_t raw_elements, float * gates, uint64_t gate_elements) {
    auto * p = static_cast<Project *>(handle);
    if (!p || tokens < 0 || tokens > p->max_chunk) return -1;
    const auto & c = p->c;
    const uint64_t n = uint64_t(tokens) * uint64_t(c.dim * c.hc_mult);
    const int64_t k = c.input_dim, rows = c.dim * (c.hc_mult + 1), blocks = k / 32;
    const uint64_t pn = uint64_t(tokens) * uint64_t(rows), gn = uint64_t(tokens) * uint64_t(c.hc_mult);
    if (output_elements < n || (projected ? projected_elements < pn : projected_elements != 0) ||
            (raw_output ? raw_elements < n : raw_elements != 0) ||
            (gates ? gate_elements < gn : gate_elements != 0)) return -1;
    if (!tokens) return 0;
    if (!output || !finite_bf16(lookup, uint64_t(tokens) * k) || !finite_bf16(hidden, n)) return -1;
    if (mask) for (int64_t t = 0; t < tokens; ++t) if (mask[t] > 1) return -1;
    CsrGuard guard;
    quantize(lookup, uint64_t(tokens) * k, p->aq.data(), p->as.data());
    const auto & tab = tables();
    for (int64_t i = 0; i < tokens * k; ++i) p->act[i] = tab.fp8[p->aq[i]];
    const auto dot = p->vectorize ? vector_dot32 : scalar_dot32;
    #pragma omp parallel num_threads(p->workers)
    {
        CsrGuard worker_guard;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < tokens * rows; ++i) {
            const int64_t t = i / rows, row = i % rows;
            float total = 0.0f;
            for (int64_t b = 0; b < blocks; ++b) {
                const float partial = dot(p->act.data() + t * k + b * 32, c.weight + row * k + b * 32);
                total += (partial * tab.scale[p->as[t * blocks + b]]) * tab.scale[c.scale[(row / 32) * blocks + b]];
            }
            p->kv[i] = to_bf16(total);
        }
    }
    if (!finite_bf16(p->kv.data(), pn)) return -3;
    apply_gate(*p, hidden, mask, tokens);
    if (!finite_bf16(p->result.data(), n)) return -3;
    for (uint64_t i = 0; i < n; ++i) if (!std::isfinite(p->raw[i])) return -3;
    for (uint64_t i = 0; i < gn; ++i) if (!std::isfinite(p->gates[i])) return -3;
    std::memcpy(output, p->result.data(), n * 2);
    if (projected) std::memcpy(projected, p->kv.data(), pn * 2);
    if (raw_output) std::memcpy(raw_output, p->raw.data(), n * 4);
    if (gates) std::memcpy(gates, p->gates.data(), gn * 4);
    return 0;
}
