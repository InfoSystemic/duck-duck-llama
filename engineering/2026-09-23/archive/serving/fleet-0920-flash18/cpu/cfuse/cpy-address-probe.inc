// Diagnostic only. Does not change the copy, synchronization, or tensor contents.
static void glm_pool_cpy_address_probe(const ggml_compute_params * params, const ggml_tensor * dst) {
    static const bool enabled = [] { const char * v = std::getenv("GGML_CPU_CPY_ADDRESS_PROBE"); return v && std::atoi(v) == 1; }();
    static std::atomic<int> remaining{8};
    if (!enabled || params->ith != 0 || !std::strstr(dst->name, "cache_s_l0 ")) return;
    if (remaining.fetch_sub(1, std::memory_order_relaxed) <= 0) return;
    const ggml_tensor * src = dst->src[0];
    fprintf(stderr, "CPY_ADDRESS name='%s' threadpool=%p src=%p dst=%p shape=[%lld,%lld,%lld,%lld] src_nb=[%zu,%zu,%zu,%zu] dst_nb=[%zu,%zu,%zu,%zu]\n",
        dst->name, (void *) params->threadpool, src->data, dst->data,
        (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
}
