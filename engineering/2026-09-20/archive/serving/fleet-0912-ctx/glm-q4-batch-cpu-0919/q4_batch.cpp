#include "repack.h"
#include <immintrin.h>
#include <cassert>
#include <cstring>

// Exact production per-token integer terms and per-block floating FMA order.
// Pointer arrays allow nonadjacent routed tokens and output rows.
template<int NR>
static inline void q4_batch_impl(int n, float * const *out, const void *vx,
                                const void * const *in, int nc) {
    const int nb=n/QK_K;
    const auto *weights=static_cast<const block_q4_K_x16 *>(vx);
    const __m512i mask=_mm512_set1_epi8(15);
    for(int g=0;g<nc/16;++g) {
        const auto *bp=weights+size_t(g)*nb;
        __m512 acc[NR];
        #pragma GCC unroll 3
        for(int r=0;r<NR;++r)acc[r]=_mm512_setzero_ps();
        for(int b=0;b<nb;++b) {
            const block_q8_K *a[NR];
            __m512i dot[NR],mins[NR];
            #pragma GCC unroll 3
            for(int r=0;r<NR;++r) {
                a[r]=static_cast<const block_q8_K *>(in[r])+b;
                dot[r]=mins[r]=_mm512_setzero_si512();
            }
            for(int j=0;j<4;++j) {
                __m512i lo0[NR],lo1[NR],hi0[NR],hi1[NR];
                #pragma GCC unroll 3
                for(int r=0;r<NR;++r)lo0[r]=lo1[r]=hi0[r]=hi1[r]=_mm512_setzero_si512();
                #pragma GCC unroll 8
                for(int i=0;i<8;++i) {
                    const __m512i w=_mm512_loadu_si512(bp[b].qs+j*512+i*64);
                    const __m512i wl=_mm512_and_si512(w,mask);
                    const __m512i wh=_mm512_and_si512(_mm512_srli_epi16(w,4),mask);
                    #pragma GCC unroll 3
                    for(int r=0;r<NR;++r) {
                        int32_t ql,qh;
                        std::memcpy(&ql,a[r]->qs+j*64+i*4,4);
                        std::memcpy(&qh,a[r]->qs+j*64+32+i*4,4);
                        if(i%2==0) {
                            lo0[r]=_mm512_dpbusd_epi32(lo0[r],wl,_mm512_set1_epi32(ql));
                            hi0[r]=_mm512_dpbusd_epi32(hi0[r],wh,_mm512_set1_epi32(qh));
                        } else {
                            lo1[r]=_mm512_dpbusd_epi32(lo1[r],wl,_mm512_set1_epi32(ql));
                            hi1[r]=_mm512_dpbusd_epi32(hi1[r],wh,_mm512_set1_epi32(qh));
                        }
                    }
                }
                const __m512i sl=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)bp[b].scales[2*j]));
                const __m512i sh=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)bp[b].scales[2*j+1]));
                #pragma GCC unroll 3
                for(int r=0;r<NR;++r) {
                    dot[r]=_mm512_add_epi32(dot[r],_mm512_mullo_epi32(_mm512_add_epi32(lo0[r],lo1[r]),sl));
                    dot[r]=_mm512_add_epi32(dot[r],_mm512_mullo_epi32(_mm512_add_epi32(hi0[r],hi1[r]),sh));
                }
                const __m512i ml=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)bp[b].mins[2*j]));
                const __m512i mh=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)bp[b].mins[2*j+1]));
                #pragma GCC unroll 3
                for(int r=0;r<NR;++r) {
                    const int32_t ql=int32_t(a[r]->bsums[4*j])+a[r]->bsums[4*j+1];
                    const int32_t qh=int32_t(a[r]->bsums[4*j+2])+a[r]->bsums[4*j+3];
                    mins[r]=_mm512_add_epi32(mins[r],_mm512_mullo_epi32(ml,_mm512_set1_epi32(ql)));
                    mins[r]=_mm512_add_epi32(mins[r],_mm512_mullo_epi32(mh,_mm512_set1_epi32(qh)));
                }
            }
            const __m512 d=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)bp[b].d));
            const __m512 dm=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)bp[b].dmin));
            #pragma GCC unroll 3
            for(int r=0;r<NR;++r) {
                const __m512 ad=_mm512_set1_ps(a[r]->d);
                acc[r]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(dot[r]),_mm512_mul_ps(d,ad),acc[r]);
                acc[r]=_mm512_fnmadd_ps(_mm512_cvtepi32_ps(mins[r]),_mm512_mul_ps(dm,ad),acc[r]);
            }
        }
        #pragma GCC unroll 3
        for(int r=0;r<NR;++r)_mm512_storeu_ps(out[r]+g*16,acc[r]);
    }
}
extern "C" __attribute__((noinline)) void ggml_gemv_q4_K_x16_batch2(int n,float * const *out,const void *vx,const void * const *in,int nc) {
    q4_batch_impl<2>(n,out,vx,in,nc);
}
extern "C" __attribute__((noinline)) void ggml_gemv_q4_K_x16_batch3(int n,float * const *out,const void *vx,const void * const *in,int nc) {
    q4_batch_impl<3>(n,out,vx,in,nc);
}
