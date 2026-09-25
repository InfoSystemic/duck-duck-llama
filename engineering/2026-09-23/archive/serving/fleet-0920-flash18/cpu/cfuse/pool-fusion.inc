// No model/meta graph mutation: fuse only the exact GLM kpool subgraph.
static int ggml_cpu_try_glm_pool_fusion(
        const struct ggml_cgraph * graph,
        const int start,
        const struct ggml_compute_params * params) {
    #define POOL_REJECT() do { if (params->ith == 0 && ggml_cpu_glm_pool_probe && strstr(graph->nodes[start]->name, "indexer_pool_members") && atomic_fetch_sub_explicit(&ggml_cpu_glm_pool_probe_left, 1, memory_order_relaxed) > 0) fprintf(stderr, "GLM_POOL_REJECT line=%d name=%s\n", __LINE__, graph->nodes[start]->name); return 0; } while (0)
    enum { N = 14 };
    static const enum ggml_op ops[N] = {
        GGML_OP_GET_ROWS, GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_CONT,
        GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_CONT,
        GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE,
        GGML_OP_ADD, GGML_OP_SOFT_MAX, GGML_OP_MUL, GGML_OP_SUM_ROWS,
    };
    if (start + N > graph->n_nodes) POOL_REJECT();
    struct ggml_tensor ** n = graph->nodes + start;
    for (int i = 0; i < N; ++i) {
        if (n[i]->op != ops[i] || n[i]->type != GGML_TYPE_F32) POOL_REJECT();
    }
    const struct ggml_tensor * kg = n[0]->src[0];
    const struct ggml_tensor * cells = n[0]->src[1];
    const struct ggml_tensor * ape = n[7]->src[0];
    const int64_t d = n[13]->ne[1], pools = n[13]->ne[2], streams = n[13]->ne[3];
    if (d <= 0 || pools <= 0 || streams <= 0 || !kg || !cells || !ape ||
            kg->type != GGML_TYPE_F16 || cells->type != GGML_TYPE_I32 || ape->type != GGML_TYPE_F32 ||
            kg->ne[0] != 2*d || kg->ne[2] != streams || kg->ne[3] != 1 || kg->nb[0] != sizeof(ggml_fp16_t) ||
            cells->ne[0] != 4*pools || cells->ne[1] != streams || cells->ne[2] != 1 || cells->ne[3] != 1 ||
            cells->nb[0] != sizeof(int32_t) || ape->ne[0] != d || ape->ne[1] != 4 ||
            ape->ne[2] != 1 || ape->ne[3] != 1 || ape->nb[0] != sizeof(float) ||
            n[0]->ne[0] != 2*d || n[0]->ne[1] != 4*pools || n[0]->ne[2] != streams || n[0]->ne[3] != 1 ||
            !ggml_is_contiguous(n[0]) || n[0]->view_src || n[13]->ne[0] != 1) POOL_REJECT();

    // Verify the actual strides and views, not names or expected model metadata.
    for (int branch = 0; branch < 2; ++branch) {
        const int v = branch ? 4 : 1;
        if (n[v]->src[0] != n[0] || n[v]->view_src != n[0] ||
                n[v]->view_offs != (size_t) branch*d*sizeof(float) ||
                n[v]->ne[0] != d || n[v]->ne[1] != 4 || n[v]->ne[2] != pools || n[v]->ne[3] != streams ||
                n[v]->nb[0] != sizeof(float) || n[v]->nb[1] != n[0]->nb[1] ||
                n[v]->nb[2] != 4*n[0]->nb[1] || n[v]->nb[3] != n[0]->nb[2] ||
                n[v+1]->src[0] != n[v] || n[v+2]->src[0] != n[v+1] ||
                n[v+1]->ne[0] != 4 || n[v+1]->ne[1] != d || n[v+1]->ne[2] != pools || n[v+1]->ne[3] != streams ||
                n[v+1]->nb[0] != n[v]->nb[1] || n[v+1]->nb[1] != n[v]->nb[0] ||
                n[v+1]->nb[2] != n[v]->nb[2] || n[v+1]->nb[3] != n[v]->nb[3] ||
                !ggml_are_same_shape(n[v+1], n[v+2]) || !ggml_is_contiguous(n[v+2])) POOL_REJECT();
    }
    if (n[7]->ne[0] != 4 || n[7]->ne[1] != d || n[7]->ne[2] != 1 || n[7]->ne[3] != 1 ||
            n[7]->nb[0] != ape->nb[1] || n[7]->nb[1] != ape->nb[0] ||
            n[8]->src[0] != n[7] || !ggml_are_same_shape(n[7], n[8]) || !ggml_is_contiguous(n[8]) ||
            n[9]->src[0] != n[8] || !ggml_are_same_shape(n[8], n[9]) || !ggml_is_contiguous(n[9]) ||
            n[10]->src[0] != n[6] || n[10]->src[1] != n[9] ||
            n[11]->src[0] != n[10] || n[11]->src[1] || n[11]->src[2] ||
            !((n[12]->src[0] == n[3] && n[12]->src[1] == n[11]) ||
              (n[12]->src[1] == n[3] && n[12]->src[0] == n[11])) || n[13]->src[0] != n[12]) POOL_REJECT();
    for (int i = 10; i <= 12; ++i) {
        if (!ggml_are_same_shape(n[3], n[i]) || !ggml_is_contiguous(n[i]) || n[i]->view_src) POOL_REJECT();
    }
    if (!ggml_is_contiguous(n[13]) || n[13]->view_src) POOL_REJECT();
    float scale, max_bias;
    memcpy(&scale, (const float *) n[11]->op_params, sizeof(float));
    memcpy(&max_bias, (const float *) n[11]->op_params + 1, sizeof(float));
    if (scale != 1.0f || max_bias != 0.0f) POOL_REJECT();

    const int output = start + N - 1;
    if (!ggml_can_fuse_subgraph(graph, start, N, ops, &output, 1)) POOL_REJECT();
    // Allocation happened before fusion; final storage may reuse intermediate
    // storage, but must never overwrite the live cache, cells, or position weights.
    // 2026-09-20: an output placed over `cells` no longer rejects. Once the LAST DSA layer's GET_ROWS has consumed
    // pool_cells the allocator may hand its storage to that layer's output, which is exactly what happened at depth:
    // layer 43 alone ran the unfused chain (+51 ms per graph at 128K). The kernels now snapshot the cell indices
    // they need and barrier before writing, so only the KV cache and the weight remain hazards.
    if (ggml_cpu_pool_overlaps(n[13], kg) || ggml_cpu_pool_overlaps(n[13], ape)) POOL_REJECT();
    ggml_compute_forward_glm_pool_fused(params, n[0], ape, n[11], n[12], n[13]);
    #undef POOL_REJECT
    return N - 1;
}
