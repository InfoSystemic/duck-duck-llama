// SPDX-License-Identifier: MIT
// Lossless E4M3-to-BF16 expansion amortized across repeated inference calls.
// Keeps native E8M0 scales, the original dot32 FP32 tree, K-order accumulation,
// and BF16 output rounding. Expanded weights use twice the source storage.
#include "deepseek-v41-native-gemm-0910b.cpp"

namespace {
inline __m512 expand_bf16(const uint16_t * raw) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(raw))),16));
}
template<int Tile> void expanded_gemm(const float * a,const float * as,const uint16_t * b,
        const uint8_t * bs,int m,int n,int k,int workers,uint16_t * out) {
    const auto & t=tables();
    const int blocks=k/32;
    #pragma omp parallel num_threads(workers)
    {
        CsrGuard guard;
        #pragma omp for schedule(static)
        for (int row=0;row<n;row+=Tile) {
            for (int token=0;token<m;++token) {
                const float * act=a+int64_t(token)*k;
                const float * asc=as+int64_t(token)*blocks;
                float result[Tile]={};
                for (int block=0;block<blocks;++block) {
                    const __m512 a0=_mm512_loadu_ps(act+block*32);
                    const __m512 a1=_mm512_loadu_ps(act+block*32+16);
                    for (int r=0;r<Tile&&row+r<n;++r) {
                        const auto * w=b+int64_t(row+r)*k+block*32;
                        const float partial=sum16(_mm512_add_ps(_mm512_mul_ps(a0,expand_bf16(w)),
                            _mm512_mul_ps(a1,expand_bf16(w+16))));
                        result[r]+=(partial*asc[block])*t.scale[bs[int64_t((row+r)/32)*blocks+block]];
                    }
                }
                for (int r=0;r<Tile&&row+r<n;++r) out[int64_t(token)*n+row+r]=bf16(result[r]);
            }
        }
    }
}
}

extern "C" int ds41_pack_fp8_bf16(const uint8_t * input,uint16_t * output,int64_t count,int workers) {
    if (!input||!output||count<1||count>(int64_t(131072)*32768)||workers<1||workers>64) return -1;
    const auto & t=tables();
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int64_t i=0;i<count;++i) {
        // Shift the FP32 representation directly: all finite E4M3 values are
        // exactly representable in BF16; preserve sign even for zero and NaN.
        uint32_t bits;std::memcpy(&bits,&t.fp8[input[i]],4);output[i]=uint16_t(bits>>16);
    }
    return 0;
}

extern "C" int ds41_gemm_fp8_bf16(const uint8_t * a,const uint8_t * as,const uint16_t * b,
        const uint8_t * bs,int m,int n,int k,int workers,int tile,uint16_t * out) {
    if (!a||!as||!b||!bs||!out||m<1||m>512||n<1||n>131072||k<32||k>32768||k%32||
            workers<1||workers>64||(tile!=1&&tile!=2&&tile!=4)) return -1;
    try {
        CsrGuard guard;const auto & t=tables();
        std::vector<float> act(int64_t(m)*k),scales(int64_t(m)*k/32);
        for (size_t i=0;i<act.size();++i) act[i]=t.fp8[a[i]];
        for (size_t i=0;i<scales.size();++i) scales[i]=t.scale[as[i]];
        if (tile==1) expanded_gemm<1>(act.data(),scales.data(),b,bs,m,n,k,workers,out);
        else if (tile==2) expanded_gemm<2>(act.data(),scales.data(),b,bs,m,n,k,workers,out);
        else expanded_gemm<4>(act.data(),scales.data(),b,bs,m,n,k,workers,out);
        return 0;
    } catch (...) { return -2; }
}
