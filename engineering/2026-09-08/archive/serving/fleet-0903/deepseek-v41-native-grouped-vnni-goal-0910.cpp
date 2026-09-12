// SPDX-License-Identifier: MIT
// Experimental grouped mixed FP4-VNNI / FP8-baseline decode projections.
// Includes the immutable standalone candidate and preserves its arithmetic.
#include "deepseek-v41-native-vnni-goal-0910.cpp"
#include <omp.h>

struct DS41GroupedVnniDescriptor {
    int mode,n,k,reserved;
    const uint8_t * a;
    const uint8_t * as;
    const uint8_t * b;   // FP4: packed tile bytes. FP8: native row-major bytes.
    const uint8_t * bs;  // FP4: packed scale bytes. FP8: native scale bytes.
    const int16_t * sums; // FP4 block sums; null for FP8.
    uint16_t * out;
};

namespace {
struct GroupedVnniActivation {
    const uint8_t * raw;
    const uint8_t * raw_scale;
    int k;
    bool needs4=false,needs8=false;
    std::vector<float> decoded,scale;
    std::vector<uint32_t> low,middle,top;
    std::vector<uint8_t> nan_block;
    explicit GroupedVnniActivation(const DS41GroupedVnniDescriptor & d):raw(d.a),raw_scale(d.as),k(d.k) {}
    void prepare(const Tables & t) {
        scale.resize(k/32);
        for (int block=0;block<k/32;++block) scale[block]=t.scale[raw_scale[block]];
        if (needs8) {
            decoded.resize(k);
            for (int i=0;i<k;++i) decoded[i]=t.fp8[raw[i]];
        }
        if (!needs4) return;
        low.resize(k/4);middle.resize(k/4);top.resize(k/4);nan_block.resize(k/32,0);
        for (int word=0;word<k/4;++word) {
            uint32_t lo=0,mid=0,hi=0;
            for (int lane=0;lane<4;++lane) {
                const uint8_t code=raw[word*4+lane];
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
    }
};
inline void grouped_vnni_tile(const GroupedVnniActivation & a,const DS41GroupedVnniDescriptor & d,int tile,__m512i lut) {
    const int blocks=d.k/32;
    __m512 result=_mm512_setzero_ps();
    for (int block=0;block<blocks;++block) {
        const uint8_t * weight=d.b+(int64_t(tile)*blocks+block)*256;
        __m512i lo=_mm512_setzero_si512(),mid=lo,hi=lo;
        for (int chunk=0;chunk<8;++chunk) {
            const __m512i w=unpack4_integer64(weight+chunk*32,lut);
            const int word=block*8+chunk;
            lo=_mm512_dpbusd_epi32(lo,_mm512_set1_epi32(a.low[word]),w);
            mid=_mm512_dpbusd_epi32(mid,_mm512_set1_epi32(a.middle[word]),w);
            hi=_mm512_dpbusd_epi32(hi,_mm512_set1_epi32(a.top[word]),w);
        }
        const int64_t meta=(int64_t(tile)*blocks+block)*16;
        const __m512i sw=_mm512_cvtepi16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(d.sums+meta)));
        hi=_mm512_sub_epi32(hi,_mm512_slli_epi32(sw,2));
        const __m512i total=_mm512_add_epi32(lo,_mm512_add_epi32(_mm512_slli_epi32(mid,8),_mm512_slli_epi32(hi,16)));
        __m512 partial=_mm512_mul_ps(_mm512_cvtepi32_ps(total),_mm512_set1_ps(0x1p-10f));
        if (a.nan_block[block]) partial=_mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
        result=_mm512_add_ps(result,_mm512_mul_ps(_mm512_mul_ps(partial,_mm512_set1_ps(a.scale[block])),decode_scale16(d.bs+meta)));
    }
    alignas(64) float values[16];
    _mm512_store_ps(values,result);
    for (int row=tile*16;row<std::min(d.n,tile*16+16);++row) d.out[row]=bf16(values[row-tile*16]);
}
inline void grouped_baseline8_row(const GroupedVnniActivation & a,const DS41GroupedVnniDescriptor & d,int row,const Tables & t) {
    const int blocks=d.k/32;
    const auto * weight=d.b+int64_t(row)*d.k;
    const auto * scale=d.bs+int64_t(row/32)*blocks;
    float result=0;
    for (int j=0;j<blocks;++j) {
        const float part=dot32<8>(a.decoded.data()+j*32,weight+j*32,t);
        result+=(part*a.scale[j])*t.scale[scale[j]];
    }
    d.out[row]=bf16(result);
}
}

// Decode-only tasks; packed/native buffer ownership stays with the caller.
// One region covers all tasks. Every thread handles its contiguous row fraction
// in every task (whole row16 tiles for FP4), with no inter-task barriers.
extern "C" int ds41_grouped_vnni(const DS41GroupedVnniDescriptor * tasks,int count,int workers) {
    if (!tasks||count<1||count>64||workers<1||workers>64) return -1;
    for (int i=0;i<count;++i) {
        const auto & d=tasks[i];
        if ((d.mode!=4&&d.mode!=8)||d.reserved||!valid_shape(d.n,d.k,workers)||
            !d.a||!d.as||!d.b||!d.bs||!d.out||(d.mode==4&&!d.sums)) return -1;
    }
    try {
        CsrGuard guard;
        const auto & t=tables();
        std::vector<GroupedVnniActivation> activations;
        activations.reserve(count);
        std::vector<int> index(count);
        for (int i=0;i<count;++i) {
            const auto & d=tasks[i];
            int found=-1;
            for (int j=0;j<int(activations.size());++j) {
                const auto & a=activations[j];
                if (a.raw==d.a&&a.raw_scale==d.as&&a.k==d.k) { found=j;break; }
            }
            if (found<0) { found=int(activations.size());activations.emplace_back(d); }
            index[i]=found;
            activations[found].needs4|=d.mode==4;
            activations[found].needs8|=d.mode==8;
        }
        for (auto & a:activations) a.prepare(t);
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const int thread=omp_get_thread_num(),threads=omp_get_num_threads();
            const __m512i lut=_mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(fp4_integer)));
            for (int task=0;task<count;++task) {
                const auto & d=tasks[task];
                const auto & a=activations[index[task]];
                if (d.mode==4) {
                    const int tiles=(d.n+15)/16;
                    const int begin=tiles*thread/threads,end=tiles*(thread+1)/threads;
                    for (int tile=begin;tile<end;++tile) grouped_vnni_tile(a,d,tile,lut);
                } else {
                    const int begin=d.n*thread/threads,end=d.n*(thread+1)/threads;
                    for (int row=begin;row<end;++row) grouped_baseline8_row(a,d,row,t);
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}
