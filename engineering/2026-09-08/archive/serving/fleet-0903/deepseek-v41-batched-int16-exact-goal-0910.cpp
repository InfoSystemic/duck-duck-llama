// SPDX-License-Identifier: MIT
// Exact FP4 token batches reuse each packed weight decode across m<=8 tokens.
// The immutable grouped exact16 implementation supplies packing and fallbacks.
#include "deepseek-v41-grouped-int16-exact-goal-0910.cpp"

namespace {
template<int M,bool Selective> inline void batch_exact16_products(__m512i (&sum)[M],
        const std::vector<Exact16Activation> & activations,const uint8_t * weight,
        int block,unsigned eligible,__m256i lut) {
    for (int chunk=0;chunk<8;++chunk) {
        const __m512i w0=exact16_pair_weights(weight+chunk*32,lut);
        const __m512i w1=exact16_pair_weights(weight+chunk*32+16,lut);
        #pragma GCC unroll 8
        for (int token=0;token<M;++token) {
            if constexpr (Selective) { if (!(eligible&(1U<<token))) continue; }
            const auto * pairs=activations[token].pairs.data()+block*16+chunk*2;
            sum[token]=_mm512_dpwssd_epi32(sum[token],_mm512_set1_epi32(pairs[0]),w0);
            sum[token]=_mm512_dpwssd_epi32(sum[token],_mm512_set1_epi32(pairs[1]),w1);
        }
    }
}
template<int M> inline void batch_exact16_tile(const std::vector<Exact16Activation> & activations,
        const DS41Exact16Descriptor & d,int tile,__m256i lut) {
    const int blocks=d.k/32;
    const auto & t=tables();
    __m512 result[M];
    #pragma GCC unroll 8
    for (int token=0;token<M;++token) result[token]=_mm512_setzero_ps();
    for (int block=0;block<blocks;++block) {
        __m512i sums[M];
        unsigned eligible=0;
        #pragma GCC unroll 8
        for (int token=0;token<M;++token) {
            sums[token]=_mm512_setzero_si512();
            if (activations[token].eligible[block]) eligible|=1U<<token;
        }
        const uint8_t * weight=d.packed+(int64_t(tile)*blocks+block)*256;
        if (eligible==((1U<<M)-1)) batch_exact16_products<M,false>(sums,activations,weight,block,eligible,lut);
        else if (eligible) batch_exact16_products<M,true>(sums,activations,weight,block,eligible,lut);
        const int64_t meta=(int64_t(tile)*blocks+block)*16;
        const __m512 weight_scale=exact16_scales(d.packed_scale+meta);
        #pragma GCC unroll 8
        for (int token=0;token<M;++token) {
            __m512 partial;
            if (eligible&(1U<<token)) partial=_mm512_mul_ps(_mm512_cvtepi32_ps(sums[token]),_mm512_set1_ps(0x1p-7f));
            else {
                alignas(64) float values[16]{};
                for (int row=tile*16;row<std::min(d.n,tile*16+16);++row)
                    values[row-tile*16]=dot32<4>(activations[token].decoded.data()+block*32,
                        d.b+int64_t(row)*(d.k/2)+block*16,t);
                partial=_mm512_load_ps(values);
            }
            result[token]=_mm512_add_ps(result[token],_mm512_mul_ps(
                _mm512_mul_ps(partial,_mm512_set1_ps(activations[token].scales[block])),weight_scale));
        }
    }
    #pragma GCC unroll 8
    for (int token=0;token<M;++token) {
        alignas(64) float values[16];_mm512_store_ps(values,result[token]);
        for (int row=tile*16;row<std::min(d.n,tile*16+16);++row)
            d.out[int64_t(token)*d.n+row]=bf16(values[row-tile*16]);
    }
}
template<int M> void batch_exact16_parallel(const std::vector<Exact16Activation> & activations,
        const DS41Exact16Descriptor & d,int workers) {
    const int tiles=(d.n+15)/16;
    #pragma omp parallel num_threads(workers)
    {
        CsrGuard thread_guard;
        const __m256i lut=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(exact16_weights)));
        #pragma omp for schedule(static)
        for (int tile=0;tile<tiles;++tile) batch_exact16_tile<M>(activations,d,tile,lut);
    }
}
}

extern "C" int ds41_exact16_batched(const uint8_t * a,const uint8_t * as,const uint8_t * b,const uint8_t * bs,
        const uint8_t * packed,const uint8_t * packed_scale,int m,int n,int k,int workers,uint16_t * out,uint64_t * stats) {
    if (!a||!as||!b||!bs||!packed||!packed_scale||!out||!stats||m<1||m>8||!exact16_shape(n,k,workers)) return -1;
    try {
        CsrGuard guard;stats[0]=stats[1]=0;
        std::vector<Exact16Activation> activations;activations.reserve(m);
        const DS41Exact16Descriptor d={4,n,k,0,a,as,b,bs,packed,packed_scale,out};
        for (int token=0;token<m;++token) {
            DS41Exact16Descriptor token_descriptor=d;
            token_descriptor.a=a+int64_t(token)*k;
            token_descriptor.as=as+int64_t(token)*(k/32);
            activations.emplace_back(token_descriptor);
            activations.back().needs4=true;
            activations.back().prepare();
            for (uint8_t fast:activations.back().eligible) ++stats[fast?0:1];
        }
        switch (m) {
            case 1:batch_exact16_parallel<1>(activations,d,workers);break;
            case 2:batch_exact16_parallel<2>(activations,d,workers);break;
            case 3:batch_exact16_parallel<3>(activations,d,workers);break;
            case 4:batch_exact16_parallel<4>(activations,d,workers);break;
            case 5:batch_exact16_parallel<5>(activations,d,workers);break;
            case 6:batch_exact16_parallel<6>(activations,d,workers);break;
            case 7:batch_exact16_parallel<7>(activations,d,workers);break;
            case 8:batch_exact16_parallel<8>(activations,d,workers);break;
        }
        return 0;
    } catch (...) { return -2; }
}
