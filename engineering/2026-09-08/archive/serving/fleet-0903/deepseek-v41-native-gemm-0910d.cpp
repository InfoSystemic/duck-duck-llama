// SPDX-License-Identifier: MIT
// CPU bring-up kernels for the publisher's unchanged FP8 and packed FP4 weights.
// FP32 partials are scaled per 32 elements, accumulated in K order, then BF16 RNE.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace {
struct CsrGuard {
    unsigned saved = _mm_getcsr();
    CsrGuard() { _mm_setcsr(saved & ~(0x8040U | 0x6000U)); }
    ~CsrGuard() { _mm_setcsr(saved); }
};
struct Tables {
    alignas(64) std::array<float,256> fp8{}, scale{};
    alignas(64) std::array<float,16> fp4{0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    Tables() {
        CsrGuard guard;
        for (unsigned c=0;c<256;++c) {
            unsigned m=c&127,e=m>>3,f=m&7;
            float v=m==127?NAN:e==0?std::ldexp(float(f),-9):std::ldexp(float(8+f),int(e)-10);
            fp8[c]=c&128?-v:v;
            scale[c]=c==255?NAN:std::ldexp(1.f,int(c)-127);
        }
    }
};
const Tables & tables() { static const Tables t; return t; }
inline uint16_t bf16(float f) {
    uint32_t u; std::memcpy(&u,&f,4);
    if ((u&0x7fffffffU)>0x7f800000U) return 0x7fc0;
    return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);
}
inline __m512 unpack4(const uint8_t * w, __m512 lut) {
    const __m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w));
    const __m128i repeated=_mm_unpacklo_epi8(bytes,bytes);
    const __m128i codes=_mm_or_si128(_mm_and_si128(repeated,_mm_set1_epi16(0x000f)),
        _mm_and_si128(_mm_srli_epi16(repeated,4),_mm_set1_epi16(0x0f00)));
    return _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(codes),lut);
}
inline float sum16(__m512 x) {
    // Explicit half reduction, identical to the scalar oracle's 16-lane tree.
    __m256 a=_mm256_add_ps(_mm512_castps512_ps256(x),_mm512_extractf32x8_ps(x,1));
    __m128 b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    b=_mm_add_ps(b,_mm_movehl_ps(b,b));
    b=_mm_add_ss(b,_mm_shuffle_ps(b,b,1));
    return _mm_cvtss_f32(b);
}
// Direct IEEE expansion avoids sixteen scattered LUT loads for each vector.
inline __m512 decode8(const uint8_t * raw) {
    const __m512i codes=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw)));
    const __m512i magnitude=_mm512_and_si512(codes,_mm512_set1_epi32(127));
    __m512i bits=_mm512_add_epi32(_mm512_slli_epi32(magnitude,20),_mm512_set1_epi32(120<<23));
    const __m512 sub=_mm512_mul_ps(_mm512_cvtepi32_ps(magnitude),_mm512_set1_ps(0x1p-9f));
    bits=_mm512_mask_mov_epi32(bits,_mm512_cmplt_epi32_mask(magnitude,_mm512_set1_epi32(8)),_mm512_castps_si512(sub));
    bits=_mm512_mask_mov_epi32(bits,_mm512_cmpeq_epi32_mask(magnitude,_mm512_set1_epi32(127)),_mm512_set1_epi32(0x7fc00000));
    bits=_mm512_or_si512(bits,_mm512_slli_epi32(_mm512_and_si512(codes,_mm512_set1_epi32(128)),24));
    return _mm512_castsi512_ps(bits);
}
template<int Mode> inline float dot32(const float * a,const uint8_t * b,const Tables & t) {
    __m512 w0,w1;
    if constexpr (Mode==4) {
        const __m512 lut=_mm512_load_ps(t.fp4.data());
        w0=unpack4(b,lut);w1=unpack4(b+8,lut);
    } else {
        w0=decode8(b);w1=decode8(b+16);
    }
    return sum16(_mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(a),w0),_mm512_mul_ps(_mm512_loadu_ps(a+16),w1)));
}
template<int Mode> void gemm(const float * a,const float * as,const uint8_t * b,const uint8_t * bs,
        int m,int n,int k,int workers,uint16_t * out) {
    const auto & t=tables();
    const int blocks=k/32, rowbytes=Mode==4?k/2:k;
    #pragma omp parallel num_threads(workers)
    {
        CsrGuard guard;
        #pragma omp for schedule(static)
        for (int row=0;row<n;++row) {
            const auto * weight=b+int64_t(row)*rowbytes;
            const auto * scale=bs+int64_t(Mode==4?row:row/32)*blocks;
            for (int token=0;token<m;++token) {
                const float * act=a+int64_t(token)*k;
                const float * asc=as+int64_t(token)*blocks;
                float result=0;
                for (int j=0;j<blocks;++j) {
                    float part=dot32<Mode>(act+j*32,weight+j*(Mode==4?16:32),t);
                    result+=(part*asc[j])*t.scale[scale[j]];
                }
                out[int64_t(token)*n+row]=bf16(result);
            }
        }
    }
}
// FP8 activations on the 1/64 lattice and FP4 weights on the 1/2 lattice
// have integer products <= 28672*12. Any sum of <=32 products is <2^24,
// so the entire original FP32 reduction is exact in units of 1/128.
// VNNI can therefore change the reduction grouping without changing its result.
inline __m512i unpack4_i16(const uint8_t * weight) {
    const __m128i bytes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(weight));
    __m256i repeated=_mm256_castsi128_si256(_mm_unpacklo_epi8(bytes,bytes));
    repeated=_mm256_inserti128_si256(repeated,_mm_unpackhi_epi8(bytes,bytes),1);
    const __m256i codes=_mm256_or_si256(_mm256_and_si256(repeated,_mm256_set1_epi16(0x000f)),
        _mm256_and_si256(_mm256_srli_epi16(repeated,4),_mm256_set1_epi16(0x0f00)));
    const __m128i lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    return _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(lut),codes));
}
inline bool integer_activation(const float * source,int16_t * output) {
    const __m512 a0=_mm512_mul_ps(_mm512_loadu_ps(source),_mm512_set1_ps(64.f));
    const __m512 a1=_mm512_mul_ps(_mm512_loadu_ps(source+16),_mm512_set1_ps(64.f));
    const __m512i i0=_mm512_cvttps_epi32(a0),i1=_mm512_cvttps_epi32(a1);
    const bool exact=_mm512_cmp_ps_mask(a0,_mm512_cvtepi32_ps(i0),_CMP_EQ_OQ)==0xffff &&
                     _mm512_cmp_ps_mask(a1,_mm512_cvtepi32_ps(i1),_CMP_EQ_OQ)==0xffff;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output),_mm512_cvtsepi32_epi16(i0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+16),_mm512_cvtsepi32_epi16(i1));
    return exact;
}
void gemm4_integer(const float * a,const int16_t * ai,const uint8_t * eligible,const float * as,
        const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    const auto & t=tables();const int blocks=k/32,rowbytes=k/2;
    #pragma omp parallel num_threads(workers)
    {
        CsrGuard guard;
        #pragma omp for schedule(static)
        for (int row=0;row<n;++row) {
            const auto * weight=b+int64_t(row)*rowbytes;
            const auto * scale=bs+int64_t(row)*blocks;
            for (int token=0;token<m;++token) {
                const float * act=a+int64_t(token)*k;
                const int16_t * integers=ai+int64_t(token)*k;
                const uint8_t * fast=eligible+int64_t(token)*blocks;
                const float * asc=as+int64_t(token)*blocks;
                float result=0;
                for (int j=0;j<blocks;++j) {
                    float partial;
                    if (fast[j]) {
                        const __m512i products=_mm512_dpwssd_epi32(_mm512_setzero_si512(),
                            _mm512_loadu_si512(integers+j*32),unpack4_i16(weight+j*16));
                        partial=float(_mm512_reduce_add_epi32(products))*(1.f/128.f);
                    } else partial=dot32<4>(act+j*32,weight+j*16,t);
                    result+=(partial*asc[j])*t.scale[scale[j]];
                }
                out[int64_t(token)*n+row]=bf16(result);
            }
        }
    }
}

}

// Inputs are contiguous, bounded tensors validated by the Python bridge. Caller
// owns every buffer, must keep it live, and must not alias output with inputs.
extern "C" int ds41_gemm(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    if ((mode!=4 && mode!=8)||m<1||m>512||n<1||n>131072||k<32||k>32768||k%32||workers<1||workers>64||
        !a||!as||!b||!bs||!out) return -1;
    try {
        CsrGuard guard;
        const auto & t=tables();
        std::vector<float> act(int64_t(m)*k),scales(int64_t(m)*k/32);
        for (size_t i=0;i<act.size();++i) act[i]=t.fp8[a[i]];
        for (size_t i=0;i<scales.size();++i) scales[i]=t.scale[as[i]];
        if (mode==4) {
            std::vector<int16_t> integers(act.size());
            std::vector<uint8_t> eligible(scales.size());
            for (size_t j=0;j<eligible.size();++j) eligible[j]=integer_activation(act.data()+j*32,integers.data()+j*32);
            gemm4_integer(act.data(),integers.data(),eligible.data(),scales.data(),b,bs,m,n,k,workers,out);
        }
        else gemm<8>(act.data(),scales.data(),b,bs,m,n,k,workers,out);
        return 0;
    } catch (...) { return -2; }
}

// Exposed for exhaustive format validation, without any model-weight side effects.
extern "C" int ds41_decode8(const uint8_t * input,float * output,int count) {
    if (!input||!output||count<0||count>1048576||count%16) return -1;
    CsrGuard guard;
    for (int i=0;i<count;i+=16) _mm512_storeu_ps(output+i,decode8(input+i));
    return 0;
}

// Diagnostic: validate selection of the exact integer path for FP8 activation blocks.
extern "C" int ds41_integer_blocks(const uint8_t * a,int count) {
    if (!a||count<0||count>1048576||count%32) return -1;
    const auto & t=tables();CsrGuard guard;int eligible=0;
    alignas(64) float values[32];alignas(64) int16_t integers[32];
    for (int i=0;i<count;i+=32) {
        for (int j=0;j<32;++j) values[j]=t.fp8[a[i+j]];
        eligible+=integer_activation(values,integers);
    }
    return eligible;
}
