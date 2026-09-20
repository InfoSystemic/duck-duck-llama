#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

struct Case { int d, pools, streams, slots, pad, kind; bool side_output, reverse_mul, gallocr; };
static uint32_t rng = 0x612afe43;
static uint32_t random_bits() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float random_float() { return (int(random_bits()%8193) - 4096) / 1024.f; }
static void fail(const char * message) { std::fprintf(stderr, "%s\n", message); std::exit(1); }

int main(int argc, char ** argv) {
    if (argc != 4) fail("usage: test_pool OUTPUT EXPECT_FUSION(0/1) THREADS");
    const int threads = std::atoi(argv[3]);
    const bool expect_fusion = std::atoi(argv[2]);
    FILE * output = std::fopen(argv[1], "wb");
    if (!output) fail("open output");
    auto count = reinterpret_cast<uint64_t (*)(void)>(dlsym(RTLD_DEFAULT, "ggml_cpu_glm_pool_fused_count"));
    Dl_info cpu_info{}, base_info{};
    if(!dladdr((void *)ggml_backend_cpu_init,&cpu_info) || !dladdr((void *)ggml_init,&base_info)) fail("dladdr");
    std::fprintf(stderr,"CPU_LIBRARY %s\nBASE_LIBRARY %s\n",cpu_info.dli_fname,base_info.dli_fname);
    auto backend = ggml_backend_cpu_init();
    if (!backend) fail("CPU backend init");
    ggml_backend_cpu_set_n_threads(backend, threads);
    const std::vector<Case> cases = {
        {128,66,1,4,0,0,false,false,false},
        {128,67,2,4,8,0,false,false,false},
        {129,17,3,4,5,0,false,true,false},
        {7,3,2,4,3,1,false,false,false},
        {1,1,1,4,0,2,false,false,false},
        {128,65,1,4,0,2,false,false,false},
        {128,19,2,4,1,3,false,false,false},
        {128,67,1,4,0,4,false,false,false},
        {128,66,1,4,0,0,true,false,false},
        {128,66,1,3,0,0,false,false,false},
        {128,66,1,4,0,5,false,false,false},
        {128,66,1,4,0,0,false,false,true},
        {128,67,2,4,8,0,false,false,true},
        {129,17,3,4,5,0,false,true,true},
        {128,66,1,4,0,0,true,false,true},
        {128,1026,1,4,0,0,false,false,true},
        {128,17,2,4,7,6,false,false,false},
        {128,17,2,4,7,7,false,false,true},
        {128,17,2,4,7,8,false,false,true},
        {128,17,2,4,7,9,false,false,true},
        {128,17,2,4,7,10,false,false,false},
    };
    uint64_t total_calls = 0;
    for (size_t ci=0; ci<cases.size(); ++ci) {
        const Case c = cases[ci];
        const int cells_count = std::max(19, c.pools*c.slots + 7);
        ggml_init_params ip{1024*1024, nullptr, true};
        auto ctx = ggml_init(ip), weights = ggml_init(ip), inputs = ggml_init(ip);
        // APE uses a real WEIGHTS buffer, as required by can_fuse_subgraph's
        // external-view-source guard. Dynamic cache/cells use a separate buffer.
        auto ape = ggml_new_tensor_2d(weights, GGML_TYPE_F32, c.d+(c.kind==6?c.pad:0), c.slots);
        if(c.kind==6) ape=ggml_view_2d(weights,ape,c.d,c.slots,ape->nb[1],0);
        auto wb = ggml_backend_alloc_ctx_tensors(weights, backend);
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto kg_storage = ggml_new_tensor_3d(inputs, c.kind==7?GGML_TYPE_F32:GGML_TYPE_F16, 2*c.d+c.pad, cells_count, c.streams);
        auto kg = ggml_view_3d(inputs, kg_storage, 2*c.d, cells_count, c.streams,
                               kg_storage->nb[1], kg_storage->nb[2], 0);
        auto cell_storage = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, c.slots*c.pools+c.pad, c.streams);
        auto cells = ggml_view_2d(inputs, cell_storage, c.slots*c.pools, c.streams, cell_storage->nb[1], 0);
        auto ib = ggml_backend_alloc_ctx_tensors(inputs, backend);
        auto members = ggml_get_rows(ctx, kg, cells);
        ggml_set_name(members, "indexer_pool_members-test");
        auto mk = ggml_view_4d(ctx, members, c.d,c.slots,c.pools,c.streams,
                              members->nb[1],members->nb[1]*c.slots,members->nb[2],0);
        auto mg = ggml_view_4d(ctx, members, c.d,c.slots,c.pools,c.streams,
                              members->nb[1],members->nb[1]*c.slots,members->nb[2],c.d*sizeof(float));
        auto keys = ggml_cont(ctx, ggml_permute(ctx,mk,1,0,2,3));
        auto gates = ggml_cont(ctx, ggml_permute(ctx,mg,1,0,2,3));
        auto ac = ggml_cont(ctx, ggml_transpose(ctx,ape));
        auto ape_broadcast=ggml_reshape_4d(ctx,ac,c.slots,c.d,1,1);
        gates = c.kind==9 ? ggml_add_inplace(ctx,gates,ape_broadcast) : ggml_add(ctx,gates,ape_broadcast);
        auto probs = c.kind == 5 ? ggml_soft_max_ext(ctx,gates,nullptr,.5f,0.f) : ggml_soft_max(ctx,gates);
        auto product = ggml_mul(ctx,keys,probs);
        auto sum = ggml_sum_rows(ctx,product);
        auto final = ggml_reshape_4d(ctx,sum,c.d,c.pools,1,c.streams);
        ggml_set_output(final);
        if (c.side_output) ggml_set_output(members);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,final);
        if (c.reverse_mul) std::swap(product->src[0], product->src[1]);
        ggml_tensor * side = nullptr;
        if(c.kind==8) { side=ggml_sum(ctx,members); ggml_set_output(side); ggml_build_forward_expand(graph,side); }
        ggml_backend_buffer_t cb = nullptr;
        ggml_gallocr_t ga = nullptr;
        if (c.gallocr) {
            ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            if (!ggml_gallocr_alloc_graph(ga,graph)) fail("graph allocator");
        } else cb = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if(c.kind==10) { sum->data=kg->data; final->data=kg->data; }
        if (ci == 0) {
            for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
                auto n=ggml_graph_node(graph,i);
                std::fprintf(stderr,"node %d %s [%lld,%lld,%lld,%lld] flags=%d\n",i,ggml_op_name(n->op),
                    (long long)n->ne[0],(long long)n->ne[1],(long long)n->ne[2],(long long)n->ne[3],n->flags);
            }
        }
        const uint64_t before = count ? count() : 0;
        for (int repeat=0;repeat<2;++repeat) {
            for (int s=0;s<c.streams;++s) {
                for(int cell=0;cell<cells_count;++cell) {
                    auto p=(ggml_fp16_t *)((char *)kg->data+cell*kg->nb[1]+s*kg->nb[2]);
                    for(int ch=0;ch<2*c.d;++ch) {
                        float f=random_float();
                        if(c.kind==1) f=ch<c.d ? ((ch&1) ? -0.f : 0.f) : 0.f;
                        if(c.kind==2) {
                            const uint16_t halves[]={0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff,0x0400,0x8400};
                            p[ch]=halves[(ch+cell+s+repeat)%10];
                            continue;
                        }
                        if(c.kind==3 && ch>=c.d) f=(cell%4-2)*120.f;
                        if(c.kind==7) ((float *)p)[ch]=f;
                        else p[ch]=ggml_fp32_to_fp16(f);
                    }
                }
                for(int p=0;p<c.slots*c.pools;++p) {
                    int32_t cell=random_bits()%cells_count;
                    if(c.kind==4) cell=0; // invalid/resident-missing pools use cell zero.
                    *(int32_t *)((char *)cells->data+p*cells->nb[0]+s*cells->nb[1])=cell;
                }
            }
            for(int m=0;m<c.slots;++m) for(int ch=0;ch<c.d;++ch)
                *(float *)((char *)ape->data+ch*ape->nb[0]+m*ape->nb[1])=c.kind==1 ? 0.f : random_float()/8.f;
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) fail("graph compute failed");
            const size_t bytes=ggml_nbytes(final);
            if (std::fwrite(final->data,1,bytes,output)!=bytes) fail("write output");
            if(side && std::fwrite(side->data,1,ggml_nbytes(side),output)!=ggml_nbytes(side)) fail("write side consumer");
            if (c.side_output && std::fwrite(members->data,1,ggml_nbytes(members),output)!=ggml_nbytes(members)) fail("write side output");
        }
        const uint64_t delta=(count ? count() : 0)-before;
        const bool eligible=threads>1 && c.slots==4 && !c.side_output && c.kind<5;
        const uint64_t expected=expect_fusion && eligible ? 2 : 0;
        if(delta!=expected) {
            std::fprintf(stderr,"case %zu expected %llu fusions, got %llu\n",ci,(unsigned long long)expected,(unsigned long long)delta);
            return 2;
        }
        total_calls+=delta;
        std::fprintf(stderr,"case %zu PASS d=%d p=%d streams=%d slots=%d pad=%d kind=%d allocator=%s calls=%llu\n",ci,c.d,c.pools,c.streams,c.slots,c.pad,c.kind,c.gallocr?"graph":"ctx",(unsigned long long)delta);
        if(ga) ggml_gallocr_free(ga);
        if(cb) ggml_backend_buffer_free(cb);
        ggml_backend_buffer_free(ib); ggml_backend_buffer_free(wb);
        ggml_free(ctx); ggml_free(inputs); ggml_free(weights);
    }
    ggml_backend_free(backend);
    std::fclose(output);
    std::printf("PASS %zu cases, two executions each, %llu fused calls\n",cases.size(),(unsigned long long)total_calls);
}
