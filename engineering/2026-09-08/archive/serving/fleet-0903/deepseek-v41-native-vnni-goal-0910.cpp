// SPDX-License-Identifier: MIT
// Experimental lossless FP4 nibble repacking and integer FP8*FP4 block dots.
// Input values, block scales and output BF16 rounding are unchanged. Integer
// block summation rounds once to FP32 and may differ from the original FP32 tree.
#include "deepseek-v41-native-gemm-0910b.cpp"

namespace {
constexpr int8_t fp4_integer[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
inline bool valid_shape(int n,int k,int workers) {
    return n>=1&&n<=131072&&k>=32&&k<=32768&&k%32==0&&workers>=1&&workers<=64;
}
inline __m512i unpack4_integer64(const uint8_t * packed,__m512i lut) {
    const __m512i bytes=_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed)));
    const __m512i codes=_mm512_or_si512(_mm512_and_si512(bytes,_mm512_set1_epi16(15)),
        _mm512_and_si512(_mm512_slli_epi16(bytes,4),_mm512_set1_epi16(0x0f00)));
    return _mm512_shuffle_epi8(lut,codes);
}
inline __m512 decode_scale16(const uint8_t * raw) {
    const __m512i code=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw)));
    __m512i bits=_mm512_slli_epi32(code,23);
    bits=_mm512_mask_mov_epi32(bits,_mm512_cmpeq_epi32_mask(code,_mm512_setzero_si512()),_mm512_set1_epi32(0x00400000));
    bits=_mm512_mask_mov_epi32(bits,_mm512_cmpeq_epi32_mask(code,_mm512_set1_epi32(255)),_mm512_set1_epi32(0x7fc00000));
    return _mm512_castsi512_ps(bits);
}
}

// packed[tile][block32][chunk4][row16][byte2], with zero padding only for tail rows.
// sums[tile][block32][row16] holds sum(FP4*2) in [-384,384]. Scale bytes use the
// same tile/block/row layout. Caller owns and bounds all input/output buffers.
extern "C" int ds41_vnni_pack4(const uint8_t * b,const uint8_t * bs,int n,int k,int workers,
        uint8_t * packed,int16_t * sums,uint8_t * scales) {
    if (!b||!bs||!packed||!sums||!scales||!valid_shape(n,k,workers)) return -1;
    const int blocks=k/32,tiles=(n+15)/16;
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int tile=0;tile<tiles;++tile) {
        for (int block=0;block<blocks;++block) {
            const int64_t meta=(int64_t(tile)*blocks+block)*16;
            uint8_t * target=packed+(int64_t(tile)*blocks+block)*256;
            for (int r=0;r<16;++r) {
                const int row=tile*16+r;
                int sum=0;
                for (int chunk=0;chunk<8;++chunk) {
                    for (int byte=0;byte<2;++byte) {
                        const uint8_t value=row<n?b[int64_t(row)*(k/2)+block*16+chunk*2+byte]:0;
                        target[chunk*32+r*2+byte]=value;
                        sum+=fp4_integer[value&15]+fp4_integer[value>>4];
                    }
                }
                sums[meta+r]=int16_t(sum);
                scales[meta+r]=row<n?bs[int64_t(row)*blocks+block]:127;
            }
        }
    }
    return 0;
}

extern "C" int ds41_vnni_gemm4(const uint8_t * a,const uint8_t * as,const uint8_t * packed,
        const int16_t * sums,const uint8_t * scales,int m,int n,int k,int workers,uint16_t * out) {
    if (!a||!as||!packed||!sums||!scales||!out||m<1||m>512||!valid_shape(n,k,workers)) return -1;
    try {
        CsrGuard guard;
        const auto & t=tables();
        const int blocks=k/32,tiles=(n+15)/16;
        const int64_t words=int64_t(m)*k/4;
        std::vector<uint32_t> low(words),middle(words),top(words);
        std::vector<uint8_t> nan_block(int64_t(m)*blocks,0);
        std::vector<float> activation_scale(int64_t(m)*blocks);
        for (int64_t word=0;word<words;++word) {
            uint32_t lo=0,mid=0,hi=0;
            for (int lane=0;lane<4;++lane) {
                const uint8_t code=a[word*4+lane];
                const bool nan=(code&127)==127;
                const int32_t q=nan?0:int32_t(t.fp8[code]*512.f);
                if (nan) nan_block[word/8]=1;
                const uint32_t u=uint32_t(q);
                lo|=(u&255)<<(lane*8);
                mid|=((u>>8)&255)<<(lane*8);
                hi|=uint32_t((q>>16)+4)<<(lane*8);
            }
            low[word]=lo;middle[word]=mid;top[word]=hi;
        }
        for (int64_t i=0;i<int64_t(m)*blocks;++i) activation_scale[i]=t.scale[as[i]];
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const __m512i lut=_mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(fp4_integer)));
            #pragma omp for schedule(static)
            for (int tile=0;tile<tiles;++tile) {
                for (int token=0;token<m;++token) {
                    __m512 result=_mm512_setzero_ps();
                    for (int block=0;block<blocks;++block) {
                        const int64_t ab=int64_t(token)*blocks+block;
                        const uint8_t * weight=packed+(int64_t(tile)*blocks+block)*256;
                        __m512i lo=_mm512_setzero_si512(),mid=lo,hi=lo;
                        for (int chunk=0;chunk<8;++chunk) {
                            const __m512i w=unpack4_integer64(weight+chunk*32,lut);
                            const int64_t aw=ab*8+chunk;
                            lo=_mm512_dpbusd_epi32(lo,_mm512_set1_epi32(low[aw]),w);
                            mid=_mm512_dpbusd_epi32(mid,_mm512_set1_epi32(middle[aw]),w);
                            hi=_mm512_dpbusd_epi32(hi,_mm512_set1_epi32(top[aw]),w);
                        }
                        const int64_t meta=(int64_t(tile)*blocks+block)*16;
                        const __m512i sw=_mm512_cvtepi16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sums+meta)));
                        hi=_mm512_sub_epi32(hi,_mm512_slli_epi32(sw,2));
                        const __m512i total=_mm512_add_epi32(lo,_mm512_add_epi32(_mm512_slli_epi32(mid,8),_mm512_slli_epi32(hi,16)));
                        __m512 partial=_mm512_mul_ps(_mm512_cvtepi32_ps(total),_mm512_set1_ps(0x1p-10f));
                        if (nan_block[ab]) partial=_mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
                        const __m512 scaled=_mm512_mul_ps(_mm512_mul_ps(partial,_mm512_set1_ps(activation_scale[ab])),decode_scale16(scales+meta));
                        result=_mm512_add_ps(result,scaled);
                    }
                    alignas(64) float values[16];
                    _mm512_store_ps(values,result);
                    for (int row=tile*16;row<std::min(n,tile*16+16);++row)
                        out[int64_t(token)*n+row]=bf16(values[row-tile*16]);
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}
