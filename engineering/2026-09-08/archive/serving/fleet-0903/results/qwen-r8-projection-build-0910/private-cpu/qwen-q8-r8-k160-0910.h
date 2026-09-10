// Qwen Q8 expert down projection: prepare five activation blocks once per call.
// Keep the R8 weight layout, integer dot products, and ordered FP32 FMA chain.
#pragma once

static std::atomic<uint64_t> qwen_q8_r8_k160_calls{0};
extern "C" uint64_t ggml_cpu_q8_r8_k160_calls(int);
extern "C" uint64_t ggml_cpu_q8_r8_k160_calls(int reset) {
    return reset ? qwen_q8_r8_k160_calls.exchange(0,std::memory_order_relaxed)
                 : qwen_q8_r8_k160_calls.load(std::memory_order_relaxed);
}

static void qwen_q8_r8_k160_prepared(float * GGML_RESTRICT output,
        const void * GGML_RESTRICT vx,const void * GGML_RESTRICT vy,int nc) {
    GGML_ASSERT(nc>0 && nc%8==0);
    static const bool audit=[] {
        const char * value=std::getenv("GGML_CPU_Q8_R8_K160_AUDIT");
        return value && std::strcmp(value,"1")==0;
    }();
    if (audit) qwen_q8_r8_k160_calls.fetch_add(1,std::memory_order_relaxed);
    const auto * weights=static_cast<const block_q8_0x8 *>(vx);
    const auto * activation=static_cast<const block_q8_0 *>(vy);
    __m128i aq[5][2];
    __m256i correction[5];
    __m256 activation_scale[5];
    for (int b=0;b<5;++b) {
        aq[b][0]=_mm_loadu_si128(reinterpret_cast<const __m128i *>(activation[b].qs));
        aq[b][1]=_mm_loadu_si128(reinterpret_cast<const __m128i *>(activation[b].qs+16));
        const int sum=q8_0_sum_i8x16(aq[b][0])+q8_0_sum_i8x16(aq[b][1]);
        correction[b]=_mm256_set1_epi32(128*sum);
        activation_scale[b]=_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(activation[b].d));
    }
    for (int group=0;group<nc/8;++group) {
        __m256 result=_mm256_setzero_ps();
        const block_q8_0x8 * w=weights+group*5;
        for (int b=0;b<5;++b) {
            const __m512i w0=_mm512_loadu_si512(static_cast<const void *>(w[b].qs));
            const __m512i w1=_mm512_loadu_si512(static_cast<const void *>(w[b].qs+64));
            const __m512i w2=_mm512_loadu_si512(static_cast<const void *>(w[b].qs+128));
            const __m512i w3=_mm512_loadu_si512(static_cast<const void *>(w[b].qs+192));
            const __m128i lo=_mm_add_epi32(q8_0_vnni_dot_4rows(w0,aq[b][0]),q8_0_vnni_dot_4rows(w2,aq[b][1]));
            const __m128i hi=_mm_add_epi32(q8_0_vnni_dot_4rows(w1,aq[b][0]),q8_0_vnni_dot_4rows(w3,aq[b][1]));
            const __m256i dot=_mm256_sub_epi32(_mm256_set_m128i(hi,lo),correction[b]);
            const __m256 scale=_mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].d))),activation_scale[b]);
            result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),scale,result);
        }
        _mm256_storeu_ps(output+group*8,result);
    }
}
