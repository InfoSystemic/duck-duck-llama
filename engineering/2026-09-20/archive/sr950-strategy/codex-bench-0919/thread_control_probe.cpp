#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <array>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <dlfcn.h>
#include <vector>

struct Observed {
    std::array<int,16> seen{};
    std::array<int,16> nth{};
    std::array<int,16> cpu{};
};
static void observe(ggml_tensor *dst, const ggml_tensor *, int ith, int nth, void *data) {
    auto &o=*static_cast<Observed *>(data);
    assert(ith>=0 && ith<16 && nth<=16);
    o.seen[ith]++;
    o.nth[ith]=nth;
    o.cpu[ith]=sched_getcpu();
    static_cast<float *>(dst->data)[ith]=float(nth);
}
int main() {
    const char *path=std::getenv("GGML_CPU_NUMA_THREADS_FILE");
    assert(path);
    ggml_backend_register(ggml_backend_cpu_reg());
    Dl_info info{};
    assert(dladdr(reinterpret_cast<void *>(&ggml_backend_cpu_reg),&info));
    std::cout << "{\"cpu_library\":\"" << info.dli_fname << "\"}\n";
    std::array<ggml_backend_t,4> backends{};
    for(int node=0;node<4;node++) {
        std::string name="CPU-NUMA"+std::to_string(node);
        auto device=ggml_backend_dev_by_name(name.c_str());
        assert(device);
        backends[node]=ggml_backend_dev_init(device,nullptr);
        assert(backends[node]);
    }
    for(int requested:{8,10,12,14,15,16,15}) {
        {std::ofstream control(path); control<<requested<<'\n'; assert(control.good());}
        for(int node=0;node<4;node++) {
            auto ctx=ggml_init({4*1024*1024,nullptr,true});
            assert(ctx);
            Observed o;
            auto input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16384);
            auto output=ggml_map_custom1(ctx,input,observe,GGML_N_TASKS_MAX,&o);
            ggml_set_name(output,"thread_control_observer");
            auto graph=ggml_new_graph_custom(ctx,16,false);
            ggml_build_forward_expand(graph,output);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backends[node]);
            assert(buffer);
            assert(ggml_backend_graph_compute(backends[node],graph)==GGML_STATUS_SUCCESS);
            std::cout<<"{\"node\":"<<node<<",\"requested\":"<<requested<<",\"observed\":[";
            for(int i=0;i<16;i++) {
                assert(o.seen[i]==(i<requested ? 1:0));
                if(i<requested) {
                    assert(o.nth[i]==requested);
                    assert(o.cpu[i]>=node*16 && o.cpu[i]<(node+1)*16);
                    if(i)std::cout<<',';
                    std::cout<<"{\"ith\":"<<i<<",\"nth\":"<<o.nth[i]<<",\"cpu\":"<<o.cpu[i]<<'}';
                }
            }
            std::cout<<"],\"passed\":true}\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    for(auto backend:backends)ggml_backend_free(backend);
}
