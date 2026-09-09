#include "ops.h"
#include "ggml-cpu-impl.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sched.h>
#include <sys/stat.h>

extern "C" void ggml_compute_forward_cpy(const ggml_compute_params * params, ggml_tensor * dst) {
    using forward_fn = void (*)(const ggml_compute_params *, ggml_tensor *);
    static const auto forward = reinterpret_cast<forward_fn>(dlsym(RTLD_NEXT, "ggml_compute_forward_cpy"));
    static const char * arm = std::getenv("GGML_CPU_CPY_STRIDE_ARM_FILE");
    static std::atomic<unsigned> count{0};
    if (!forward) std::abort();
    const ggml_tensor * src = dst->src[0];
    if (params->ith == 0 && arm && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src->ne[0] >= 262144) {
        struct stat status{};
        if (stat(arm, &status) != 0) {
            count.store(0, std::memory_order_relaxed);
        } else if (count.fetch_add(1, std::memory_order_relaxed) < 128) {
            Dl_info info{};
            if (!dladdr(reinterpret_cast<void *>(forward), &info)) std::abort();
            std::fprintf(stderr,
                "CPY_STRIDE_TRACE {\"cpu\":%d,\"ith\":%d,\"nth\":%d,"
                "\"src_ne\":[%lld,%lld,%lld,%lld],\"dst_ne\":[%lld,%lld,%lld,%lld],"
                "\"src_nb\":[%zu,%zu,%zu,%zu],\"dst_nb\":[%zu,%zu,%zu,%zu],"
                "\"src_contiguous\":%s,\"dst_contiguous\":%s,\"same_shape\":%s,\"forward_library\":\"%s\"}\n",
                sched_getcpu(), params->ith, params->nth,
                (long long)src->ne[0], (long long)src->ne[1], (long long)src->ne[2], (long long)src->ne[3],
                (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3],
                src->nb[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                ggml_is_contiguous(src) ? "true" : "false", ggml_is_contiguous(dst) ? "true" : "false",
                ggml_are_same_shape(src, dst) ? "true" : "false", info.dli_fname);
        }
    }
    forward(params, dst);
}
