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
    alignas(64) std::array<uint16_t,128> fp8_half{};
    alignas(64) std::array<uint16_t,32> fp4_half{};
    Tables() {
        CsrGuard guard;
        for (unsigned c=0;c<256;++c) {
            unsigned m=c&127,e=m>>3,f=m&7;
            float v=m==127?NAN:e==0?std::ldexp(float(f),-9):std::ldexp(float(8+f),int(e)-10);
            fp8[c]=c&128?-v:v;
            scale[c]=c==255?NAN:std::ldexp(1.f,int(c)-127);
            if (c<128) fp8_half[c]=_cvtss_sh(v,0);
        }
        for (int c=0;c<32;++c) fp4_half[c]=_cvtss_sh(fp4[c&15],0);
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
// All unsigned FP8 values are exactly representable as FP16. Four register
// tables cover the 128 magnitudes; sign is reattached after the word lookup.
// FP16 is only an exact decoding intermediate, never a weight conversion cache.
struct RegisterTables {
    __m512i f80,f81,f82,f83,f4;
    explicit RegisterTables(const Tables & t):
        f80(_mm512_load_si512(t.fp8_half.data())),
        f81(_mm512_load_si512(t.fp8_half.data()+32)),
        f82(_mm512_load_si512(t.fp8_half.data()+64)),
        f83(_mm512_load_si512(t.fp8_half.data()+96)),
        f4(_mm512_load_si512(t.fp4_half.data())) {}
};
inline __m512i decode8half32(const uint8_t * w,const RegisterTables & t) {
    const __m512i codes=_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w)));
    const __m512i lo=_mm512_permutex2var_epi16(t.f80,codes,t.f81);
    const __m512i hi=_mm512_permutex2var_epi16(t.f82,codes,t.f83);
    const __m512i magnitude=_mm512_mask_blend_epi16(_mm512_test_epi16_mask(codes,_mm512_set1_epi16(64)),lo,hi);
    const __m512i sign=_mm512_slli_epi16(_mm512_and_si512(codes,_mm512_set1_epi16(128)),8);
    return _mm512_or_si512(magnitude,sign);
}
inline __m512i decode4half32(const uint8_t * w,const RegisterTables & t) {
    const __m256i bytes=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
    const __m256i codes=_mm256_or_si256(_mm256_and_si256(bytes,_mm256_set1_epi16(15)),
        _mm256_and_si256(_mm256_slli_epi16(bytes,4),_mm256_set1_epi16(0x0f00)));
    return _mm512_permutexvar_epi16(_mm512_cvtepu8_epi16(codes),t.f4);
}
template<int Mode> inline float dot32(__m512 a0,__m512 a1,const uint8_t * b,const RegisterTables & t) {
    const __m512i half=Mode==4?decode4half32(b,t):decode8half32(b,t);
    const __m512 w0=_mm512_cvtph_ps(_mm512_castsi512_si256(half));
    const __m512 w1=_mm512_cvtph_ps(_mm512_extracti64x4_epi64(half,1));
    return sum16(_mm512_add_ps(_mm512_mul_ps(a0,w0),_mm512_mul_ps(a1,w1)));
}
template<int Mode,int Tile> inline void rows(const float * a,const float * as,const uint8_t * b,const uint8_t * bs,
        int base,int m,int n,int k,uint16_t * out,const Tables & t,const RegisterTables & rt) {
    const int blocks=k/32,rowbytes=Mode==4?k/2:k;
    for (int token=0;token<m;++token) {
        const float * act=a+int64_t(token)*k;
        const float * asc=as+int64_t(token)*blocks;
        float result[Tile]{};
        for (int j=0;j<blocks;++j) {
            const __m512 a0=_mm512_loadu_ps(act+j*32),a1=_mm512_loadu_ps(act+j*32+16);
            const float activation_scale=asc[j];
            const float fp8_scale=Mode==8?t.scale[bs[int64_t(base/32)*blocks+j]]:0;
            #pragma GCC unroll 8
            for (int r=0;r<Tile;++r) {
                const auto * weight=b+int64_t(base+r)*rowbytes+j*(Mode==4?16:32);
                const float part=dot32<Mode>(a0,a1,weight,rt);
                const float weight_scale=Mode==4?t.scale[bs[int64_t(base+r)*blocks+j]]:fp8_scale;
                result[r]+=(part*activation_scale)*weight_scale;
            }
        }
        #pragma GCC unroll 8
        for (int r=0;r<Tile;++r) out[int64_t(token)*n+base+r]=bf16(result[r]);
    }
}
template<int Mode,int Tile> void gemm(const float * a,const float * as,const uint8_t * b,const uint8_t * bs,
        int m,int n,int k,int workers,uint16_t * out) {
    const auto & t=tables();
    const int groups=n/Tile;
    #pragma omp parallel num_threads(workers)
    {
        CsrGuard guard;
        const RegisterTables rt(t);
        #pragma omp for schedule(static)
        for (int group=0;group<groups;++group) rows<Mode,Tile>(a,as,b,bs,group*Tile,m,n,k,out,t,rt);
        #pragma omp for schedule(static)
        for (int row=groups*Tile;row<n;++row) rows<Mode,1>(a,as,b,bs,row,m,n,k,out,t,rt);
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

// Tiles are divisors of the FP8 scale's 32-row sharing group.
#define EXPORT_GEMM(Name,Tile) \
extern "C" int Name(int mode,const uint8_t * a,const uint8_t * as,const uint8_t * b,const uint8_t * bs, \
    int m,int n,int k,int workers,uint16_t * out) { return run_gemm<Tile>(mode,a,as,b,bs,m,n,k,workers,out); }
EXPORT_GEMM(ds41_gemm,4)
EXPORT_GEMM(ds41_gemm_t1,1)
EXPORT_GEMM(ds41_gemm_t2,2)
EXPORT_GEMM(ds41_gemm_t4,4)
EXPORT_GEMM(ds41_gemm_t8,8)

// Exposed for exhaustive format validation, without any model-weight side effects.
extern "C" int ds41_decode8(const uint8_t * input,float * output,int count) {
    if (!input||!output||count<0||count>1048576||count%32) return -1;
    CsrGuard guard;
    const RegisterTables rt(tables());
    for (int i=0;i<count;i+=32) {
        const __m512i half=decode8half32(input+i,rt);
        _mm512_storeu_ps(output+i,_mm512_cvtph_ps(_mm512_castsi512_si256(half)));
        _mm512_storeu_ps(output+i+16,_mm512_cvtph_ps(_mm512_extracti64x4_epi64(half,1)));
    }
    return 0;
}
