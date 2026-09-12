// SPDX-License-Identifier: MIT
// Exact FP4 row16 VNNI path. Eligible activations lie on a 1/64 lattice;
// integer sums then fit exactly in FP32 throughout the published dot32 tree.
// Other blocks explicitly reconstruct that original tree, never approximate it.
#include "deepseek-v41-native-grouped-vnni-goal-0910.cpp"

namespace {
struct LatticeActivation {
    const uint8_t * raw;
    const uint8_t * raw_scale;
    int k;
    std::vector<float> decoded,scale;
    std::vector<uint32_t> even,odd;
    std::vector<uint8_t> eligible;
    LatticeActivation(const DS41GroupedVnniDescriptor & d,const Tables & t):
        raw(d.a),raw_scale(d.as),k(d.k),decoded(k),scale(k/32),even(k/4),odd(k/4),eligible(k/32,1) {
        for (int i=0;i<k;++i) decoded[i]=t.fp8[raw[i]];
        for (int i=0;i<k/32;++i) scale[i]=t.scale[raw_scale[i]];
        for (int word=0;word<k/4;++word) {
            uint16_t lanes[4];
            for (int lane=0;lane<4;++lane) {
                const float f=decoded[word*4+lane]*64.f;
                if (!std::isfinite(f)||f<-32768.f||f>32767.f||f!=std::trunc(f)) {
                    eligible[word/8]=0;lanes[lane]=0;
                } else lanes[lane]=uint16_t(int16_t(f));
            }
            even[word]=uint32_t(lanes[0])|(uint32_t(lanes[2])<<16);
            odd[word]=uint32_t(lanes[1])|(uint32_t(lanes[3])<<16);
        }
    }
};

inline void lattice_tile(const LatticeActivation & a,const DS41GroupedVnniDescriptor & d,
        int tile,const Tables & t,__m512i lut) {
    const int blocks=d.k/32;
    __m512 result=_mm512_setzero_ps();
    for (int block=0;block<blocks;++block) {
        const auto * weights=d.b+(int64_t(tile)*blocks+block)*256;
        __m512 partial;
        if (a.eligible[block]) {
            __m512i sum=_mm512_setzero_si512();
            for (int chunk=0;chunk<8;++chunk) {
                // Each row supplies two packed bytes for four weights. The
                // low/high nibbles select (w0,w2)/(w1,w3) respectively.
                const __m512i bytes=_mm512_cvtepu8_epi16(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weights+chunk*32)));
                const __m512i lo=_mm512_permutexvar_epi16(_mm512_and_si512(bytes,_mm512_set1_epi16(15)),lut);
                const __m512i hi=_mm512_permutexvar_epi16(_mm512_and_si512(_mm512_srli_epi16(bytes,4),_mm512_set1_epi16(15)),lut);
                const int word=block*8+chunk;
                sum=_mm512_dpwssd_epi32(sum,_mm512_set1_epi32(a.even[word]),lo);
                sum=_mm512_dpwssd_epi32(sum,_mm512_set1_epi32(a.odd[word]),hi);
            }
            // At most 32 * 28672 * 12 = 11,010,048 in any partial sum.
            partial=_mm512_mul_ps(_mm512_cvtepi32_ps(sum),_mm512_set1_ps(0x1p-7f));
        } else {
            alignas(64) float values[16];
            for (int r=0;r<16;++r) {
                uint8_t original[16];
                for (int chunk=0;chunk<8;++chunk) {
                    original[chunk*2]=weights[chunk*32+r*2];
                    original[chunk*2+1]=weights[chunk*32+r*2+1];
                }
                values[r]=dot32<4>(a.decoded.data()+block*32,original,t);
            }
            partial=_mm512_load_ps(values);
        }
        const int64_t meta=(int64_t(tile)*blocks+block)*16;
        const __m512 scaled=_mm512_mul_ps(_mm512_mul_ps(partial,_mm512_set1_ps(a.scale[block])),
            decode_scale16(d.bs+meta));
        result=_mm512_add_ps(result,scaled);
    }
    alignas(64) float values[16];_mm512_store_ps(values,result);
    for (int row=tile*16;row<std::min(d.n,tile*16+16);++row) d.out[row]=bf16(values[row-tile*16]);
}

inline void lattice_baseline8_row(const LatticeActivation & a,const DS41GroupedVnniDescriptor & d,
        int row,const Tables & t) {
    const int blocks=d.k/32;
    float sum=0;
    for (int block=0;block<blocks;++block) {
        const float partial=dot32<8>(a.decoded.data()+block*32,d.b+int64_t(row)*d.k+block*32,t);
        sum+=(partial*a.scale[block])*t.scale[d.bs[int64_t(row/32)*blocks+block]];
    }
    d.out[row]=bf16(sum);
}
}

extern "C" int ds41_grouped_lattice16(const DS41GroupedVnniDescriptor * tasks,int count,int workers) {
    if (!tasks||count<1||count>64||workers<1||workers>64) return -1;
    for (int i=0;i<count;++i) {
        const auto & d=tasks[i];
        if ((d.mode!=4&&d.mode!=8)||d.reserved||!valid_shape(d.n,d.k,workers)||
            !d.a||!d.as||!d.b||!d.bs||!d.out||(d.mode==4&&!d.sums)) return -1;
    }
    try {
        CsrGuard guard;const auto & t=tables();
        std::vector<LatticeActivation> activations;activations.reserve(count);
        std::vector<int> index(count);
        for (int i=0;i<count;++i) {
            const auto & d=tasks[i];int found=-1;
            for (int j=0;j<int(activations.size());++j) {
                const auto & a=activations[j];
                if (a.raw==d.a&&a.raw_scale==d.as&&a.k==d.k) { found=j;break; }
            }
            if (found<0) { found=int(activations.size());activations.emplace_back(d,t); }
            index[i]=found;
        }
        alignas(64) const int16_t table[32]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const int thread=omp_get_thread_num(),threads=omp_get_num_threads();
            const __m512i lut=_mm512_load_si512(table);
            for (int task=0;task<count;++task) {
                const auto & d=tasks[task];const auto & a=activations[index[task]];
                if (d.mode==4) {
                    const int tiles=(d.n+15)/16;
                    for (int tile=tiles*thread/threads;tile<tiles*(thread+1)/threads;++tile)
                        lattice_tile(a,d,tile,t,lut);
                } else {
                    for (int row=d.n*thread/threads;row<d.n*(thread+1)/threads;++row)
                        lattice_baseline8_row(a,d,row,t);
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}

extern "C" int ds41_lattice16_blocks(const uint8_t * a,int k) {
    if (!a||k<32||k>32768||k%32) return -1;
    const auto & t=tables();int count=0;
    for (int block=0;block<k/32;++block) {
        bool eligible=true;
        for (int i=0;i<32;++i) {
            const float f=t.fp8[a[block*32+i]]*64.f;
            eligible=eligible&&std::isfinite(f)&&f>=-32768.f&&f<=32767.f&&f==std::trunc(f);
        }
        count+=eligible;
    }
    return count;
}
