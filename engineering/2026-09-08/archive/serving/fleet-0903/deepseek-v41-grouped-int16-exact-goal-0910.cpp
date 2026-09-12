// SPDX-License-Identifier: MIT
// Exact FP4 grouped dot products when FP8 activations lie on the 1/64 lattice.
// Otherwise preserve the original FP32 dot32 tree for that individual block.
#include "deepseek-v41-native-gemm-0910b.cpp"
#include <omp.h>

struct DS41Exact16Descriptor {
    int mode,n,k,reserved;
    const uint8_t * a;
    const uint8_t * as;
    const uint8_t * b;  // Native source is retained for ineligible-block fallback.
    const uint8_t * bs;
    const uint8_t * packed;
    const uint8_t * packed_scale;
    uint16_t * out;
};

namespace {
inline bool exact16_shape(int n,int k,int workers) {
    return n>=1&&n<=131072&&k>=32&&k<=32768&&k%32==0&&workers>=1&&workers<=64;
}
constexpr int8_t exact16_weights[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
struct Exact16Codes {
    std::array<int16_t,256> q{};
    std::array<uint8_t,256> eligible{};
    Exact16Codes() {
        CsrGuard guard;
        for (int code=0;code<256;++code) {
            const float x=tables().fp8[code]*64.f;
            if (std::isfinite(x)&&x==std::trunc(x)) {
                q[code]=int16_t(x);
                eligible[code]=1;
            }
        }
    }
};
const Exact16Codes & exact16_codes() { static const Exact16Codes c;return c; }
inline __m512i exact16_pair_weights(const uint8_t * bytes,__m256i lut) {
    const __m256i repeated=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes)));
    const __m256i codes=_mm256_or_si256(_mm256_and_si256(repeated,_mm256_set1_epi16(15)),
        _mm256_and_si256(_mm256_slli_epi16(repeated,4),_mm256_set1_epi16(0x0f00)));
    return _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(lut,codes));
}
inline __m512 exact16_scales(const uint8_t * bytes) {
    const __m512i codes=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes)));
    __m512i bits=_mm512_slli_epi32(codes,23);
    bits=_mm512_mask_mov_epi32(bits,_mm512_cmpeq_epi32_mask(codes,_mm512_setzero_si512()),_mm512_set1_epi32(0x00400000));
    bits=_mm512_mask_mov_epi32(bits,_mm512_cmpeq_epi32_mask(codes,_mm512_set1_epi32(255)),_mm512_set1_epi32(0x7fc00000));
    return _mm512_castsi512_ps(bits);
}
struct Exact16Activation {
    const uint8_t * raw;
    const uint8_t * raw_scale;
    int k;
    bool needs4=false;
    std::vector<float> decoded,scales;
    std::vector<uint32_t> pairs;
    std::vector<uint8_t> eligible;
    explicit Exact16Activation(const DS41Exact16Descriptor & d):raw(d.a),raw_scale(d.as),k(d.k) {}
    void prepare() {
        const auto & t=tables();const auto & c=exact16_codes();
        decoded.resize(k);scales.resize(k/32);
        for (int i=0;i<k;++i) decoded[i]=t.fp8[raw[i]];
        for (int block=0;block<k/32;++block) scales[block]=t.scale[raw_scale[block]];
        if (!needs4) return;
        pairs.resize(k/2);eligible.resize(k/32,1);
        for (int i=0;i<k;i+=2) {
            pairs[i/2]=uint32_t(uint16_t(c.q[raw[i]]))|(uint32_t(uint16_t(c.q[raw[i+1]]))<<16);
            eligible[i/32]&=c.eligible[raw[i]]&c.eligible[raw[i+1]];
        }
    }
};
inline void exact16_tile(const Exact16Activation & a,const DS41Exact16Descriptor & d,int tile,__m256i lut) {
    const int blocks=d.k/32;
    const auto & t=tables();
    __m512 result=_mm512_setzero_ps();
    for (int block=0;block<blocks;++block) {
        __m512 partial;
        if (a.eligible[block]) {
            const uint8_t * weight=d.packed+(int64_t(tile)*blocks+block)*256;
            __m512i sum=_mm512_setzero_si512();
            for (int chunk=0;chunk<8;++chunk) {
                // Each pair broadcast covers the same two K positions in all
                // 16 rows: two VPDPWSSD operations consume four K positions.
                sum=_mm512_dpwssd_epi32(sum,_mm512_set1_epi32(a.pairs[block*16+chunk*2]),
                    exact16_pair_weights(weight+chunk*32,lut));
                sum=_mm512_dpwssd_epi32(sum,_mm512_set1_epi32(a.pairs[block*16+chunk*2+1]),
                    exact16_pair_weights(weight+chunk*32+16,lut));
            }
            // |a*64|<=28672, |w*2|<=12, and 32*28672*12=11010048<2^24.
            // Every product and every subset sum in the baseline FP32 tree is
            // exactly representable in units of 1/128. This reordering is exact.
            partial=_mm512_mul_ps(_mm512_cvtepi32_ps(sum),_mm512_set1_ps(0x1p-7f));
        } else {
            alignas(64) float values[16]{};
            for (int row=tile*16;row<std::min(d.n,tile*16+16);++row)
                values[row-tile*16]=dot32<4>(a.decoded.data()+block*32,d.b+int64_t(row)*(d.k/2)+block*16,t);
            partial=_mm512_load_ps(values);
        }
        const int64_t meta=(int64_t(tile)*blocks+block)*16;
        result=_mm512_add_ps(result,_mm512_mul_ps(_mm512_mul_ps(partial,_mm512_set1_ps(a.scales[block])),
                                               exact16_scales(d.packed_scale+meta)));
    }
    alignas(64) float values[16];_mm512_store_ps(values,result);
    for (int row=tile*16;row<std::min(d.n,tile*16+16);++row) d.out[row]=bf16(values[row-tile*16]);
}
inline void exact16_baseline8(const Exact16Activation & a,const DS41Exact16Descriptor & d,int row) {
    const auto & t=tables();const int blocks=d.k/32;
    const uint8_t * weight=d.b+int64_t(row)*d.k;
    const uint8_t * scale=d.bs+int64_t(row/32)*blocks;
    float result=0;
    for (int block=0;block<blocks;++block) {
        const float part=dot32<8>(a.decoded.data()+block*32,weight+block*32,t);
        result+=(part*a.scales[block])*t.scale[scale[block]];
    }
    d.out[row]=bf16(result);
}
}

// Lossless packed[tile16][block32][chunk4][pair2][row16]. No sums metadata.
extern "C" int ds41_exact16_pack4(const uint8_t * b,const uint8_t * bs,int n,int k,int workers,
                                 uint8_t * packed,uint8_t * scales) {
    if (!b||!bs||!packed||!scales||!exact16_shape(n,k,workers)) return -1;
    const int tiles=(n+15)/16,blocks=k/32;
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int tile=0;tile<tiles;++tile) {
        for (int block=0;block<blocks;++block) {
            uint8_t * target=packed+(int64_t(tile)*blocks+block)*256;
            for (int row=0;row<16;++row) {
                const int source=tile*16+row;
                for (int chunk=0;chunk<8;++chunk)
                    for (int pair=0;pair<2;++pair)
                        target[chunk*32+pair*16+row]=source<n?b[int64_t(source)*(k/2)+block*16+chunk*2+pair]:0;
                scales[(int64_t(tile)*blocks+block)*16+row]=source<n?bs[int64_t(source)*blocks+block]:127;
            }
        }
    }
    return 0;
}

extern "C" int ds41_exact16_grouped(const DS41Exact16Descriptor * tasks,int count,int workers,uint64_t * stats) {
    if (!tasks||count<1||count>64||workers<1||workers>64||!stats) return -1;
    for (int i=0;i<count;++i) {
        const auto & d=tasks[i];
        if ((d.mode!=4&&d.mode!=8)||d.reserved||!exact16_shape(d.n,d.k,workers)||
            !d.a||!d.as||!d.b||!d.bs||!d.out||(d.mode==4&&(!d.packed||!d.packed_scale))) return -1;
    }
    try {
        CsrGuard guard;stats[0]=stats[1]=0;
        std::vector<Exact16Activation> activations;activations.reserve(count);
        std::vector<int> index(count);
        for (int i=0;i<count;++i) {
            const auto & d=tasks[i];int found=-1;
            for (int j=0;j<int(activations.size());++j)
                if (activations[j].raw==d.a&&activations[j].raw_scale==d.as&&activations[j].k==d.k) {found=j;break;}
            if (found<0) {found=int(activations.size());activations.emplace_back(d);}
            index[i]=found;activations[found].needs4|=d.mode==4;
        }
        for (auto & a:activations) {
            a.prepare();
            for (uint8_t fast:a.eligible) ++stats[fast?0:1];
        }
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const int thread=omp_get_thread_num(),threads=omp_get_num_threads();
            const __m256i lut=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(exact16_weights)));
            for (int task=0;task<count;++task) {
                const auto & d=tasks[task];const auto & a=activations[index[task]];
                if (d.mode==4) {
                    const int tiles=(d.n+15)/16;
                    for (int tile=tiles*thread/threads;tile<tiles*(thread+1)/threads;++tile) exact16_tile(a,d,tile,lut);
                } else {
                    for (int row=d.n*thread/threads;row<d.n*(thread+1)/threads;++row) exact16_baseline8(a,d,row);
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}
