#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "repack.h"
#include "ggml-quants.h"
#include "qwen-q8-hc-ordered-k-0909.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sched.h>
#include <vector>

static FILE * output_dump;
static int test_threads = 15;

static bool check(int k, int nc, int nr, bool padded, ggml_backend_t backend) {
    auto wc = ggml_init({ggml_tensor_overhead()*4, nullptr, true});
    auto w = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0, k, nc);
    ggml_set_name(w, "blk.0.hc_attn_down.weight");
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, ggml_backend_cpu_repack_buffer_type());
    if (!wb || !w->extra) std::abort();
    std::vector<block_q8_0> weight(size_t(k/32)*nc);
    for (size_t b = 0; b < weight.size(); ++b) {
        weight[b].d = ggml_fp32_to_fp16(float(1+b%11)/8192.0f);
        for (int j = 0; j < 32; ++j) weight[b].qs[j] = int((b*71+j*43+13)%255)-127;
    }
    ggml_backend_tensor_set(w, weight.data(), 0, ggml_nbytes(w));
    std::vector<unsigned char> saved_weight(ggml_nbytes(w));
    std::memcpy(saved_weight.data(), w->data, saved_weight.size());
    auto ctx = ggml_init({16U*1024U*1024U, nullptr, false});
    const int xs = k+(padded ? 32 : 0), ys = nc+16, nb = k/32;
    auto storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, xs, nr);
    auto x = padded ? ggml_view_2d(ctx, storage, k, nr, storage->nb[1], 0) : storage;
    auto y = ggml_mul_mat(ctx, w, x);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    bool exact = true, quant_exact = true, guards = true, inputs = true;
    size_t mismatches = 0;
    for (int sample = 0; sample < 3; ++sample) {
        std::vector<float> original(size_t(xs)*nr, 12345.0f);
        std::vector<float> tight(size_t(k)*nr);
        for (int r = 0; r < nr; ++r) for (int j = 0; j < k; ++j) {
            const size_t i = size_t(r)*k+j;
            float value = sample == 2 ? 0.0f : float(int((i*53+sample*71+17)%1021)-510)/128.0f;
            if (sample == 1 && j%32 == 0) value = (r%2 ? -127.0f : 127.0f)/64.0f;
            original[size_t(r)*xs+j] = tight[i] = value;
        }
        std::memcpy(storage->data, original.data(), original.size()*sizeof(float));
        std::vector<block_q8_0> aq(size_t(nb)*nr);
        for (int r = 0; r < nr; ++r) {
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(tight.data()+r*k, aq.data()+r*nb, k);
        }
        for (int r = 0; r+4 <= nr; r += 4) {
            std::vector<block_q8_0x4> packed(nb);
            ggml_quantize_mat_q8_0_4x8(tight.data()+r*k, packed.data(), k);
            for (int b = 0; b < nb; ++b) for (int row = 0; row < 4; ++row) {
                const auto & a = aq[(r+row)*nb+b];
                quant_exact &= a.d == packed[b].d[row];
                for (int j = 0; j < 32; ++j) {
                    quant_exact &= a.qs[j] == packed[b].qs[(j/8)*32+row*8+j%8];
                }
            }
        }
        using count_fn = uint64_t (*)();
        const auto counter = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_hc_ordered_k_count"));
        const char * flag = std::getenv("GGML_CPU_QWEN_HC_ORDERED_K");
        const bool selected = counter && flag && !std::strcmp(flag,"1") && k == 10240 && test_threads > nc/8;
        const uint64_t before_count = counter ? counter() : 0;
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
        if (counter && counter()-before_count != uint64_t(selected)) std::abort();
        if (std::fwrite(y->data, 1, ggml_nbytes(y), output_dump) != ggml_nbytes(y)) std::abort();
        std::vector<qwen_q8_hc_partial> partial(size_t(nb)*(nc/8)*nr+2);
        std::memset(partial.data(), 0xA5, partial.size()*sizeof(partial[0]));
        const auto before = partial.front(), after = partial.back();
        for (int ith = 0; ith < 15; ++ith) {
            qwen_q8_hc_prepare(k, nc, nr, static_cast<const block_q8_0x8 *>(w->data), aq.data(),
                               ith*nb/15, (ith+1)*nb/15, partial.data()+1);
        }
        std::vector<float> output(size_t(ys)*nr+16, 23456.0f);
        for (int ith = 0; ith < 15; ++ith) {
            qwen_q8_hc_finish(k, nc, nr, partial.data()+1, output.data()+8, ys, ith, 15);
        }
        guards &= !std::memcmp(&before, &partial.front(), sizeof(before));
        guards &= !std::memcmp(&after, &partial.back(), sizeof(after));
        for (int r = 0; r < nr; ++r) {
            const auto actual = reinterpret_cast<const float *>(static_cast<const char *>(y->data)+r*y->nb[1]);
            for (int j = 0; j < nc; ++j) {
                if (std::memcmp(actual+j, output.data()+8+r*ys+j, sizeof(float))) ++mismatches;
            }
            for (int j = nc; j < ys; ++j) guards &= output[8+r*ys+j] == 23456.0f;
        }
        for (int j = 0; j < 8; ++j) guards &= output[j] == 23456.0f && output[output.size()-1-j] == 23456.0f;
        inputs &= !std::memcmp(storage->data, original.data(), original.size()*sizeof(float));
        inputs &= !std::memcmp(w->data, saved_weight.data(), saved_weight.size());
    }
    exact = mismatches == 0;
    const bool good = exact && quant_exact && guards && inputs;
    std::printf("HC_PROOF {\"k\":%d,\"nc\":%d,\"nr\":%d,\"padded\":%s,\"exact\":%s,\"quant_exact\":%s,\"guards\":%s,\"inputs\":%s,\"mismatches\":%zu}\n",
                k,nc,nr,padded?"true":"false",exact?"true":"false",quant_exact?"true":"false",
                guards?"true":"false",inputs?"true":"false",mismatches);
    std::fflush(stdout);
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_free(ctx);
    return good;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    test_threads = std::atoi(argv[2]);
    if (test_threads != 1 && test_threads != 4 && test_threads != 15) return 2;
    output_dump = std::fopen(argv[1], "wb");
    if (!output_dump) return 2;
    const auto audit_count = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_hc_ordered_k_count"));
    std::printf("HC_COUNTER_PRESENT %d\n", audit_count != nullptr);
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\n", runtime.dli_fname);
    auto pp = ggml_threadpool_params_default(test_threads);
    cpu_set_t available;
    if (sched_getaffinity(0, sizeof(available), &available) || CPU_COUNT(&available) != 15) std::abort();
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) pp.cpumask[i] = i < test_threads && CPU_ISSET(i, &available);
    pp.strict_cpu = true;
    auto pool = ggml_threadpool_new(&pp);
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, test_threads);
    ggml_backend_cpu_set_threadpool(backend, pool);
    int cases = 0, failures = 0;
    for (int k : {4096,10240,32768}) for (int nc : {64,96}) for (int nr = 1; nr <= 8; ++nr) for (bool padded : {false,true}) {
        ++cases;
        failures += !check(k,nc,nr,padded,backend);
    }
    ggml_backend_free(backend);
    ggml_threadpool_free(pool);
    std::printf("HC_SUMMARY {\"cases\":%d,\"failures\":%d}\n", cases, failures);
    if (std::fclose(output_dump)) ++failures;
    std::printf("HC_TOTAL_CALLS %llu\n", (unsigned long long)(audit_count ? audit_count() : 0));
    return failures ? 1 : 0;
}
