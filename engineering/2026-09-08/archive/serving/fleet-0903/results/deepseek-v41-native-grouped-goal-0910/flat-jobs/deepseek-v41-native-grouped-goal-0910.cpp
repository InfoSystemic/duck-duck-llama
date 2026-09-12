// SPDX-License-Identifier: MIT
// Mixed FP4/FP8 decode projections share one parallel region. Native checkpoint
// buffers are read unchanged. The included exact 32-element reduction is reused.
#include "deepseek-v41-native-gemm-goal-0910.cpp"

struct DS41GemmDescriptor {
    int mode, n, k, reserved;
    const uint8_t * a;
    const uint8_t * as;
    const uint8_t * b;
    const uint8_t * bs;
    uint16_t * out;
};

namespace {
struct DecodedActivation {
    const uint8_t * raw;
    const uint8_t * scale_raw;
    int k;
    std::vector<float> a,scale;
    DecodedActivation(const DS41GemmDescriptor & d,const Tables & t):
        raw(d.a),scale_raw(d.as),k(d.k),a(d.k),scale(d.k/32) {
        for (int i=0;i<k;++i) a[i]=t.fp8[raw[i]];
        for (int i=0;i<k/32;++i) scale[i]=t.scale[scale_raw[i]];
    }
};
struct GroupedJob { int task,base,end; };
// Baseline b decoding is the default. Define DS41_GROUPED_USE_LUT=1 to compare
// register-LUT decoding independently of the grouped scheduling change.
#ifndef DS41_GROUPED_USE_LUT
#define DS41_GROUPED_USE_LUT 0
#endif
template<int Mode> inline void grouped_row(const DecodedActivation & act,const DS41GemmDescriptor & d,
        int row,const Tables & t,const RegisterTables & rt) {
#if DS41_GROUPED_USE_LUT
    rows<Mode,1>(act.a.data(),act.scale.data(),d.b,d.bs,row,1,d.n,d.k,d.out,t,rt);
#else
    const int blocks=d.k/32;
    const auto * weight=d.b+int64_t(row)*(Mode==4?d.k/2:d.k);
    const auto * scale=d.bs+int64_t(Mode==4?row:row/32)*blocks;
    float result=0;
    const __m512 lut=_mm512_load_ps(t.fp4.data());
    for (int j=0;j<blocks;++j) {
        const auto * raw=weight+j*(Mode==4?16:32);
        __m512 w0,w1;
        if constexpr (Mode==4) { w0=unpack4(raw,lut);w1=unpack4(raw+8,lut); }
        else { w0=decode8(raw);w1=decode8(raw+16); }
        const float part=sum16(_mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(act.a.data()+j*32),w0),
            _mm512_mul_ps(_mm512_loadu_ps(act.a.data()+j*32+16),w1)));
        result+=(part*act.scale[j])*t.scale[scale[j]];
    }
    d.out[row]=bf16(result);
#endif
}
}

// ABI is a bounded array of 56-byte descriptors on the x86-64 target. Each task
// is one decoded token. Caller keeps all buffers/descriptors live and validates
// their extents; output buffers must not alias any input or another output.
extern "C" int ds41_gemm_grouped(const DS41GemmDescriptor * tasks,int count,int workers) {
    if (!tasks||count<1||count>64||workers<1||workers>64) return -1;
    for (int i=0;i<count;++i) {
        const auto & d=tasks[i];
        if ((d.mode!=4&&d.mode!=8)||d.n<1||d.n>131072||d.k<32||d.k>32768||d.k%32||d.reserved||
            !d.a||!d.as||!d.b||!d.bs||!d.out) return -1;
    }
    try {
        CsrGuard guard;
        const auto & t=tables();
        std::vector<DecodedActivation> activations;
        activations.reserve(count);
        std::vector<int> activation_index(count);
        std::vector<GroupedJob> jobs;
        for (int i=0;i<count;++i) {
            const auto & d=tasks[i];
            int found=-1;
            for (int j=0;j<int(activations.size());++j) {
                const auto & act=activations[j];
                if (act.raw==d.a&&act.scale_raw==d.as&&act.k==d.k) { found=j;break; }
            }
            if (found<0) { found=int(activations.size());activations.emplace_back(d,t); }
            activation_index[i]=found;
            for (int row=0;row<d.n;row+=32) jobs.push_back({i,row,std::min(row+32,d.n)});
        }
        #pragma omp parallel num_threads(workers)
        {
            CsrGuard thread_guard;
            const RegisterTables rt(t);
            #pragma omp for schedule(static)
            for (int job=0;job<int(jobs.size());++job) {
                const auto & j=jobs[job];
                const auto & d=tasks[j.task];
                const auto & act=activations[activation_index[j.task]];
                if (d.mode==4) {
                    for (int row=j.base;row<j.end;++row)
                        grouped_row<4>(act,d,row,t,rt);
                } else {
                    for (int row=j.base;row<j.end;++row)
                        grouped_row<8>(act,d,row,t,rt);
                }
            }
        }
        return 0;
    } catch (...) { return -2; }
}
