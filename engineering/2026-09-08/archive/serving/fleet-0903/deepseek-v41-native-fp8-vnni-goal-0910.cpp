// SPDX-License-Identifier: MIT
// Experimental FP8 integer block dots. Native FP8 bytes are only reordered.
// Each exact q=value*512 is represented as signed hi=q>>8 and unsigned lo=q&255.
// Block dots reconstruct int64 before one FP32 rounding, unlike the baseline tree.
#include "deepseek-v41-native-gemm-0910b.cpp"

namespace {
struct FP8WordTables {
    alignas(64) std::array<uint16_t,128> lower{};
    alignas(64) std::array<uint16_t,32> upper{};
    FP8WordTables() {
        CsrGuard guard;
        const auto & t=tables();
        for (int code=0;code<128;++code)
            lower[code]=code==127?0:uint16_t(uint32_t(t.fp8[code]*512.f));
        for (int group=0;group<32;++group)
            upper[group]=uint16_t((uint32_t(t.fp8[group*4]*512.f)>>8)&0xff00);
    }
};
const FP8WordTables & word_tables() { static const FP8WordTables t;return t; }
struct FP8WordRegisters {
    __m512i t0,t1,t2,t3,top;
    explicit FP8WordRegisters(const FP8WordTables & t):
        t0(_mm512_load_si512(t.lower.data())),t1(_mm512_load_si512(t.lower.data()+32)),
        t2(_mm512_load_si512(t.lower.data()+64)),t3(_mm512_load_si512(t.lower.data()+96)),
        top(_mm512_load_si512(t.upper.data())) {}
};
inline void decode_fp8_hi_lo(const uint8_t * raw,const FP8WordRegisters & t,__m512i & high,__m512i & low) {
    const __m512i code=_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw)));
    const __m512i low_table=_mm512_permutex2var_epi16(t.t0,code,t.t1);
    const __m512i high_table=_mm512_permutex2var_epi16(t.t2,code,t.t3);
    const __m512i word=_mm512_mask_blend_epi16(_mm512_test_epi16_mask(code,_mm512_set1_epi16(64)),low_table,high_table);
    const __m512i high_magnitude=_mm512_or_si512(_mm512_srli_epi16(word,8),
        _mm512_permutexvar_epi16(_mm512_srli_epi16(code,2),t.top));
    const __m512i low_magnitude=_mm512_and_si512(word,_mm512_set1_epi16(255));
    const __mmask32 negative=_mm512_test_epi16_mask(code,_mm512_set1_epi16(128));
    const __mmask32 has_low=_mm512_cmpneq_epi16_mask(low_magnitude,_mm512_setzero_si512());
    high=_mm512_mask_sub_epi16(high_magnitude,negative,_mm512_setzero_si512(),high_magnitude);
    high=_mm512_mask_sub_epi16(high,negative&has_low,high,_mm512_set1_epi16(1));
    low=_mm512_and_si512(_mm512_mask_sub_epi16(low_magnitude,negative,_mm512_setzero_si512(),low_magnitude),_mm512_set1_epi16(255));
}
inline __m256 exact_half(__m256i hh,__m256i cross,__m256i ll) {
    const __m512i total=_mm512_add_epi64(_mm512_slli_epi64(_mm512_cvtepi32_epi64(hh),16),
        _mm512_add_epi64(_mm512_slli_epi64(_mm512_cvtepi32_epi64(cross),8),_mm512_cvtepi32_epi64(ll)));
    // Maximum magnitude is 32*(448*512)^2=1,683,627,180,032, within int64.
    return _mm512_cvtepi64_ps(total);
}
inline __m512 exact_block(__m512i hh,__m512i cross,__m512i ll) {
    const __m256 a=exact_half(_mm512_castsi512_si256(hh),_mm512_castsi512_si256(cross),_mm512_castsi512_si256(ll));
    const __m256 b=exact_half(_mm512_extracti64x4_epi64(hh,1),_mm512_extracti64x4_epi64(cross,1),_mm512_extracti64x4_epi64(ll,1));
    return _mm512_mul_ps(_mm512_insertf32x8(_mm512_castps256_ps512(a),b,1),_mm512_set1_ps(0x1p-18f));
}
inline bool valid_fp8_shape(int n,int k,int workers) {
    return n>=1&&n<=131072&&k>=32&&k<=32768&&k%32==0&&workers>=1&&workers<=64;
}
}

// packed[tile16][block32][chunk2][row16][byte2]; original byte count plus tail
// padding. nan_masks[tile16][block32] is a 16-bit per-row NaN-presence bitmap.
extern "C" int ds41_fp8_vnni_pack(const uint8_t * source,int n,int k,int workers,uint8_t * packed,uint16_t * nan_masks) {
    if (!source||!packed||!nan_masks||!valid_fp8_shape(n,k,workers)) return -1;
    const int tiles=(n+15)/16,blocks=k/32;
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int tile=0;tile<tiles;++tile) {
        for (int block=0;block<blocks;++block) {
            uint8_t * target=packed+(int64_t(tile)*blocks+block)*512;
            uint16_t nan=0;
            for (int row=0;row<16;++row) {
                for (int chunk=0;chunk<16;++chunk) {
                    for (int byte=0;byte<2;++byte) {
                        const uint8_t code=tile*16+row<n?source[int64_t(tile*16+row)*k+block*32+chunk*2+byte]:0;
                        target[chunk*32+row*2+byte]=code;
                        if ((code&127)==127) nan|=uint16_t(1U<<row);
                    }
                }
            }
            nan_masks[int64_t(tile)*blocks+block]=nan;
        }
    }
    return 0;
}

extern "C" int ds41_fp8_vnni_decode32(const uint8_t * raw,int16_t * high,uint16_t * low) {
    if (!raw||!high||!low) return -1;
    CsrGuard guard;
    const FP8WordRegisters t(word_tables());
    __m512i h,l;
    decode_fp8_hi_lo(raw,t,h,l);
    _mm512_storeu_si512(high,h);_mm512_storeu_si512(low,l);
    return 0;
}

extern "C" int ds41_fp8_vnni_gemm(const uint8_t * a,const uint8_t * as,const uint8_t * packed,
        const uint8_t * bs,const uint16_t * nan_masks,int m,int n,int k,int workers,uint16_t * out) {
    if (!a||!as||!packed||!bs||!nan_masks||!out||m<1||m>512||!valid_fp8_shape(n,k,workers)) return -1;
    try {
        CsrGuard guard;
        const auto & t=tables();
        const auto & wt=word_tables();
        const int tiles=(n+15)/16,blocks=k/32;
        std::vector<uint32_t> high(int64_t(m)*k/2),low(int64_t(m)*k/2);
        std::vector<uint8_t> activation_nan(int64_t(m)*blocks,0);
        std::vector<float> scale(int64_t(m)*blocks);
        for (int64_t pair=0;pair<int64_t(m)*k/2;++pair) {
            uint32_t h=0,l=0;
            for (int lane=0;lane<2;++lane) {
                const uint8_t code=a[pair*2+lane];
                const bool nan=(code&127)==127;
                const int32_t q=nan?0:int32_t(t.fp8[code]*512.f);
                if (nan) activation_nan[pair/16]=1;
                h|=uint32_t(uint16_t(q>>8))<<(lane*16);
                l|=(uint32_t(q)&255)<<(lane*16);
            }
            high[pair]=h;low[pair]=l;
        }
        for (int64_t i=0;i<int64_t(m)*blocks;++i) scale[i]=t.scale[as[i]];
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const FP8WordRegisters rt(wt);
            #pragma omp for schedule(static)
            for (int tile=0;tile<tiles;++tile) {
                for (int token=0;token<m;++token) {
                    __m512 result=_mm512_setzero_ps();
                    for (int block=0;block<blocks;++block) {
                        __m512i hh=_mm512_setzero_si512(),cross=hh,ll=hh;
                        const uint8_t * weight=packed+(int64_t(tile)*blocks+block)*512;
                        const int64_t ab=int64_t(token)*blocks+block;
                        for (int chunk=0;chunk<16;++chunk) {
                            __m512i wh,wl;
                            decode_fp8_hi_lo(weight+chunk*32,rt,wh,wl);
                            const __m512i ah=_mm512_set1_epi32(high[ab*16+chunk]);
                            const __m512i al=_mm512_set1_epi32(low[ab*16+chunk]);
                            hh=_mm512_dpwssd_epi32(hh,ah,wh);
                            cross=_mm512_dpwssd_epi32(cross,ah,wl);
                            cross=_mm512_dpwssd_epi32(cross,al,wh);
                            ll=_mm512_dpwssd_epi32(ll,al,wl);
                        }
                        __m512 partial=exact_block(hh,cross,ll);
                        const __mmask16 nan=activation_nan[ab]?0xffff:nan_masks[int64_t(tile)*blocks+block];
                        partial=_mm512_mask_mov_ps(partial,nan,_mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000)));
                        const float weight_scale=t.scale[bs[int64_t(tile/2)*blocks+block]];
                        result=_mm512_add_ps(result,_mm512_mul_ps(_mm512_mul_ps(partial,_mm512_set1_ps(scale[ab])),_mm512_set1_ps(weight_scale)));
                    }
                    alignas(64) float values[16];
                    _mm512_store_ps(values,result);
                    for (int row=tile*16;row<std::min(n,tile*16+16);++row) out[int64_t(token)*n+row]=bf16(values[row-tile*16]);
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}
