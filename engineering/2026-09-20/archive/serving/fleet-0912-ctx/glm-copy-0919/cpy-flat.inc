// All devices keep their own copy. Split packed rows by bytes instead of ne[1].
// No new synchronization: the existing graph worker barrier joins each copy.
static std::atomic<uint64_t> glm_cpy_flat_calls{0};
extern "C" uint64_t ggml_cpu_cpy_flat_count(void);
extern "C" uint64_t ggml_cpu_cpy_flat_count(void) {
    return glm_cpy_flat_calls.load(std::memory_order_relaxed);
}

static bool glm_cpy_flat_span(const ggml_tensor * t, size_t * span) {
    const size_t es=ggml_type_size(t->type);
    if(t->nb[0]!=es || t->ne[0]<=0 || (uint64_t)t->ne[0]>SIZE_MAX/es) return false;
    size_t n=(size_t)t->ne[0]*es;
    for(int dim=1;dim<4;++dim) {
        if(t->ne[dim]<=0) return false;
        if(t->ne[dim]==1) continue;
        // Reject permuted/overlapping rows. Padding is fine.
        if(t->nb[dim]<n || (uint64_t)(t->ne[dim]-1)>(SIZE_MAX-n)/t->nb[dim]) return false;
        n+=(size_t)(t->ne[dim]-1)*t->nb[dim];
    }
    *span=n;
    return true;
}

static bool glm_cpy_try_flat(const ggml_compute_params * params, ggml_tensor * dst) {
    static const bool enabled=[] { const char * v=std::getenv("GGML_CPU_CPY_FLAT"); return v && std::atoi(v)==1; }();
    if(!enabled || params->use_ref || !std::strstr(dst->name,"cache_s_l")) return false;
    const ggml_tensor * src=dst->src[0];
    if(!src || src->type!=dst->type || ggml_is_quantized(src->type) || !ggml_are_same_shape(src,dst)) return false;
    const size_t es=ggml_type_size(src->type);
    if(es==0 || 64%es!=0) return false;
    size_t src_span=0,dst_span=0;
    if(!glm_cpy_flat_span(src,&src_span) || !glm_cpy_flat_span(dst,&dst_span)) return false;
    const uintptr_t s=(uintptr_t)src->data,d=(uintptr_t)dst->data;
    if(!s || !d || (d<=s ? s-d<dst_span : d-s<src_span)) return false;
    const uint64_t elements=ggml_nelements(src);
    if(elements>SIZE_MAX/es) return false;
    const size_t total=(size_t)elements*es;
    if(total<65536) return false;
    // Split the logical byte stream at 64-byte boundaries. Each worker may
    // visit partial rows, preserving destination padding and source strides.
    const size_t blocks=total/64+(total%64!=0);
    const auto boundary = [&](int thread) -> size_t {
        if(thread==params->nth) return total;
        const size_t nth=(size_t)params->nth,t=(size_t)thread;
        // Quotient/remainder avoids blocks*thread overflowing for huge spans.
        const size_t block=(blocks/nth)*t+(size_t)(((uint64_t)(blocks%nth)*t)/nth);
        return block*64;
    };
    const size_t b0=boundary(params->ith),b1=boundary(params->ith+1);
    const size_t row_bytes=(size_t)src->ne[0]*es;
    size_t b=b0;
    while(b<b1) {
        const size_t row=b/row_bytes,off=b%row_bytes;
        const size_t count=std::min(row_bytes-off,b1-b);
        const size_t i1=row%(size_t)src->ne[1];
        const size_t rest=row/(size_t)src->ne[1];
        const size_t i2=rest%(size_t)src->ne[2],i3=rest/(size_t)src->ne[2];
        std::memcpy((char *)dst->data+i1*dst->nb[1]+i2*dst->nb[2]+i3*dst->nb[3]+off,
            (const char *)src->data+i1*src->nb[1]+i2*src->nb[2]+i3*src->nb[3]+off,count);
        b+=count;
    }
    if(params->ith==0) {
        const uint64_t call=glm_cpy_flat_calls.fetch_add(1,std::memory_order_relaxed);
        static const bool probe=[] { const char * v=std::getenv("GGML_CPU_CPY_FLAT_PROBE"); return v && std::atoi(v)==1; }();
        if(probe && call<8) fprintf(stderr,"CPY_FLAT name='%s' src=%p dst=%p bytes=%zu threads=%d\n",dst->name,src->data,dst->data,total,params->nth);
    }
    return true;
}
