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
template<int Mode, int Tile> void gemm(const float * a,const float * as,const uint8_t * b,const uint8_t * bs,
        int m,int n,int k,int workers,uint16_t * out) {
    const auto & t=tables();
    const int blocks=k/32, rowbytes=Mode==4?k/2:k;
    #pragma omp parallel num_threads(workers)
    {
        CsrGuard guard;
        #pragma omp for schedule(static)
        for (int base=0;base<n;base+=Tile) {
            for (int token=0;token<m;++token) {
                const float * act=a+int64_t(token)*k;
                const float * asc=as+int64_t(token)*blocks;
                float result[Tile]{};
                for (int j=0;j<blocks;++j) {
                    const __m512 a0=_mm512_loadu_ps(act+j*32),a1=_mm512_loadu_ps(act+j*32+16);
                    const float activation_scale=asc[j];
                    const __m512 lut=_mm512_load_ps(t.fp4.data());
                    #pragma GCC unroll 8
                    for (int r=0;r<Tile;++r) {
                        if (base+r>=n) continue;
                        const auto * weight=b+int64_t(base+r)*rowbytes+j*(Mode==4?16:32);
                        __m512 w0,w1;
                        if constexpr (Mode==4) { w0=unpack4(weight,lut);w1=unpack4(weight+8,lut); }
                        else { w0=decode8(weight);w1=decode8(weight+16); }
                        const float part=sum16(_mm512_add_ps(_mm512_mul_ps(a0,w0),_mm512_mul_ps(a1,w1)));
                        const int64_t scale_row=Mode==4?base+r:(base+r)/32;
                        result[r]+=(part*activation_scale)*t.scale[bs[scale_row*blocks+j]];
                    }
                }
                #pragma GCC unroll 8
                for (int r=0;r<Tile;++r) if (base+r<n) out[int64_t(token)*n+base+r]=bf16(result[r]);
            }
        }
    }
}
}

// Inputs are contiguous, bounded tensors validated by the Python bridge. Caller
// owns every buffer, must keep it live, and must not alias output with inputs.
template<int Tile> int run_gemm(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    if ((mode!=4 && mode!=8)||m<1||m>512||n<1||n>131072||k<32||k>32768||k%32||workers<1||workers>64||
        !a||!as||!b||!bs||!out) return -1;
    try {
        CsrGuard guard;
        const auto & t=tables();
        std::vector<float> act(int64_t(m)*k),scales(int64_t(m)*k/32);
        for (size_t i=0;i<act.size();++i) act[i]=t.fp8[a[i]];
        for (size_t i=0;i<scales.size();++i) scales[i]=t.scale[as[i]];
        if (mode==4) gemm<4,Tile>(act.data(),scales.data(),b,bs,m,n,k,workers,out);
        else gemm<8,Tile>(act.data(),scales.data(),b,bs,m,n,k,workers,out);
        return 0;
    } catch (...) { return -2; }
}

// Independent output rows share activation loads; each row keeps the original reduction tree.
extern "C" int ds41_gemm(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    return run_gemm<4>(mode,a,as,b,bs,m,n,k,workers,out);
}
extern "C" int ds41_gemm_t1(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    return run_gemm<1>(mode,a,as,b,bs,m,n,k,workers,out);
}
extern "C" int ds41_gemm_t2(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    return run_gemm<2>(mode,a,as,b,bs,m,n,k,workers,out);
}
extern "C" int ds41_gemm_t4(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    return run_gemm<4>(mode,a,as,b,bs,m,n,k,workers,out);
}
extern "C" int ds41_gemm_t8(int mode,const uint8_t * a,const uint8_t * as,
    const uint8_t * b,const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    return run_gemm<8>(mode,a,as,b,bs,m,n,k,workers,out);
}

// Exposed for exhaustive format validation, without any model-weight side effects.
extern "C" int ds41_decode8(const uint8_t * input,float * output,int count) {
    if (!input||!output||count<0||count>1048576||count%16) return -1;
    CsrGuard guard;
    for (int i=0;i<count;i+=16) _mm512_storeu_ps(output+i,decode8(input+i));
    return 0;
}
