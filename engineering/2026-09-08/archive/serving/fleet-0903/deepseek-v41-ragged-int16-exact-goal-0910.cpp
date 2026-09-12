// SPDX-License-Identifier: MIT
// Mixed exact FP4 / baseline FP8 ragged token batches in one parallel region.
// Frozen standalone batched helpers supply per-M weight-unpack reuse.
#include "deepseek-v41-batched-int16-exact-goal-0910.cpp"

struct DS41Exact16RaggedDescriptor {
    int mode,m,n,k;
    const uint8_t * a;
    const uint8_t * as;
    const uint8_t * b;
    const uint8_t * bs;
    const uint8_t * packed;
    const uint8_t * packed_scale;
    uint16_t * out;
};
static_assert(sizeof(DS41Exact16RaggedDescriptor)==72,"Unexpected ragged descriptor ABI");

namespace {
struct RaggedExact16Activation {
    const uint8_t * a;
    const uint8_t * as;
    int m,k;
    bool needs4=false;
    std::vector<Exact16Activation> tokens;
    explicit RaggedExact16Activation(const DS41Exact16RaggedDescriptor & d):a(d.a),as(d.as),m(d.m),k(d.k) {}
    void prepare() {
        tokens.reserve(m);
        for (int token=0;token<m;++token) {
            const DS41Exact16Descriptor d={4,1,k,0,a+int64_t(token)*k,as+int64_t(token)*(k/32),nullptr,nullptr,nullptr,nullptr,nullptr};
            tokens.emplace_back(d);tokens.back().needs4=needs4;tokens.back().prepare();
        }
    }
};
template<int M> inline void ragged_exact16_tiles(const RaggedExact16Activation & a,const DS41Exact16Descriptor & d,
        int begin,int end,__m256i lut) {
    for (int tile=begin;tile<end;++tile) batch_exact16_tile<M>(a.tokens,d,tile,lut);
}
}

// All outputs are caller-owned contiguous [m,n] BF16 buffers. N and K are common
// to a call; individual M is 1..8. FP4 needs original bytes for exact fallback plus
// the existing pair-major nibble/scale pack. FP8 ignores packed/packed_scale.
extern "C" int ds41_exact16_ragged(const DS41Exact16RaggedDescriptor * tasks,int count,int workers,uint64_t * stats) {
    if (!tasks||!stats||count<1||count>128||workers<1||workers>64) return -1;
    const int n=tasks[0].n,k=tasks[0].k;
    if (!exact16_shape(n,k,workers)) return -1;
    for (int task=0;task<count;++task) {
        const auto & d=tasks[task];
        if ((d.mode!=4&&d.mode!=8)||d.m<1||d.m>8||d.n!=n||d.k!=k||
            !d.a||!d.as||!d.b||!d.bs||!d.out||(d.mode==4&&(!d.packed||!d.packed_scale))) return -1;
    }
    try {
        CsrGuard guard;stats[0]=stats[1]=0;
        std::vector<RaggedExact16Activation> activations;activations.reserve(count);
        std::vector<int> index(count);
        for (int task=0;task<count;++task) {
            const auto & d=tasks[task];int found=-1;
            for (int candidate=0;candidate<int(activations.size());++candidate) {
                const auto & a=activations[candidate];
                if (a.a==d.a&&a.as==d.as&&a.m==d.m&&a.k==d.k) {found=candidate;break;}
            }
            if (found<0) {found=int(activations.size());activations.emplace_back(d);}
            index[task]=found;activations[found].needs4|=d.mode==4;
        }
        for (auto & group:activations) {
            group.prepare();
            for (const auto & token:group.tokens)
                for (uint8_t fast:token.eligible) ++stats[fast?0:1];
        }
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const int thread=omp_get_thread_num(),threads=omp_get_num_threads();
            const __m256i lut=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(exact16_weights)));
            for (int task=0;task<count;++task) {
                const auto & input=tasks[task];const auto & a=activations[index[task]];
                DS41Exact16Descriptor d={input.mode,n,k,0,input.a,input.as,input.b,input.bs,input.packed,input.packed_scale,input.out};
                if (input.mode==4) {
                    const int tiles=(n+15)/16,begin=tiles*thread/threads,end=tiles*(thread+1)/threads;
                    switch (input.m) {
                        case 1:ragged_exact16_tiles<1>(a,d,begin,end,lut);break;
                        case 2:ragged_exact16_tiles<2>(a,d,begin,end,lut);break;
                        case 3:ragged_exact16_tiles<3>(a,d,begin,end,lut);break;
                        case 4:ragged_exact16_tiles<4>(a,d,begin,end,lut);break;
                        case 5:ragged_exact16_tiles<5>(a,d,begin,end,lut);break;
                        case 6:ragged_exact16_tiles<6>(a,d,begin,end,lut);break;
                        case 7:ragged_exact16_tiles<7>(a,d,begin,end,lut);break;
                        case 8:ragged_exact16_tiles<8>(a,d,begin,end,lut);break;
                    }
                } else {
                    const int begin=n*thread/threads,end=n*(thread+1)/threads;
                    for (int token=0;token<input.m;++token) {
                        d.out=input.out+int64_t(token)*n;
                        for (int row=begin;row<end;++row) exact16_baseline8(a.tokens[token],d,row);
                    }
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}
