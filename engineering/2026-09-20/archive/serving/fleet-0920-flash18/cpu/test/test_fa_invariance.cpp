// test_fa_invariance.cpp -- a query's attention output must not depend on the other queries of its batch.
// Runs query 0 alone, then inside 3-query batches whose other two queries (and their masks) differ, and compares bits.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static uint32_t rng = 0x1234abcd;
static uint32_t rb() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float rf() { return (int(rb() % 20001) - 10000) / 10000.f; }
struct G { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * q, * k, * mask, * out; ggml_backend_buffer_t buf; };
static G make(ggml_backend_t be, int N, int H, int n_kv) {
    G g; ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.q = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 512, N, H);
    g.k = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, 512, n_kv, 1);
    g.mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, n_kv, N);
    ggml_tensor * v = ggml_view_3d(g.ctx, g.k, 512, n_kv, 1, g.k->nb[1], g.k->nb[2], 0);
    g.out = ggml_flash_attn_ext(g.ctx, g.q, g.k, v, g.mask, 1.0f/16.0f, 0.0f, 0.0f);
    g.gf = ggml_new_graph(g.ctx); ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}
int main(int argc, char ** argv) {
    const int n_kv = argc > 1 ? atoi(argv[1]) : 4096, n_sel = argc > 2 ? atoi(argv[2]) : 2051, H = 16, threads = argc > 3 ? atoi(argv[3]) : 15;
    auto be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, threads);
    std::vector<float> kd((size_t) 512*n_kv); for (auto & x : kd) x = rf()*0.5f;
    std::vector<ggml_fp16_t> k16(kd.size()); ggml_fp32_to_fp16_row(kd.data(), k16.data(), (int64_t) kd.size());
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY), zero = ggml_fp32_to_fp16(0.0f);
    auto rand_mask = [&](std::vector<ggml_fp16_t> & m, int row, int n_vis) {
        for (int c = 0; c < n_kv; ++c) m[(size_t) row*n_kv + c] = ninf;
        for (int s = 0; s < n_vis; ) { int c = rb() % n_kv; if (m[(size_t) row*n_kv + c] == ninf) { m[(size_t) row*n_kv + c] = zero; ++s; } }
    };
    std::vector<float> q0((size_t) 512*H); for (auto & x : q0) x = rf();
    std::vector<ggml_fp16_t> m0((size_t) n_kv); rand_mask(m0, 0, n_sel);
    // reference: query 0 alone
    G g1 = make(be, 1, H, n_kv);
    ggml_backend_tensor_set(g1.k, k16.data(), 0, ggml_nbytes(g1.k));
    ggml_backend_tensor_set(g1.q, q0.data(), 0, ggml_nbytes(g1.q));   // layout [512, N=1, H]
    ggml_backend_tensor_set(g1.mask, m0.data(), 0, ggml_nbytes(g1.mask));
    ggml_backend_graph_compute(be, g1.gf);
    std::vector<float> ref((size_t) 512*H); ggml_backend_tensor_get(g1.out, ref.data(), 0, ggml_nbytes(g1.out)); // dst [512, H, N=1]
    int bad = 0;
    G g3 = make(be, 3, H, n_kv);
    ggml_backend_tensor_set(g3.k, k16.data(), 0, ggml_nbytes(g3.k));
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<float> q3((size_t) 512*3*H);
        for (int h = 0; h < H; ++h) for (int n = 0; n < 3; ++n) for (int d = 0; d < 512; ++d)   // q layout [512, N, H]
            q3[((size_t) h*3 + n)*512 + d] = n == 0 ? q0[(size_t) h*512 + d] : rf();
        std::vector<ggml_fp16_t> m3((size_t) n_kv*3);
        memcpy(m3.data(), m0.data(), n_kv*sizeof(ggml_fp16_t));
        rand_mask(m3, 1, trial % 2 ? n_sel : 17 + trial); rand_mask(m3, 2, trial % 3 ? n_sel/2 : n_kv - 3);
        ggml_backend_tensor_set(g3.q, q3.data(), 0, ggml_nbytes(g3.q));
        ggml_backend_tensor_set(g3.mask, m3.data(), 0, ggml_nbytes(g3.mask));
        ggml_backend_graph_compute(be, g3.gf);
        std::vector<float> o((size_t) 512*H*3); ggml_backend_tensor_get(g3.out, o.data(), 0, ggml_nbytes(g3.out)); // dst [512, H, N]: rows n*H + h
        const bool same = memcmp(o.data(), ref.data(), ref.size()*sizeof(float)) == 0;
        double md = 0; for (size_t i = 0; i < ref.size(); ++i) md = std::fmax(md, std::fabs((double) o[i] - ref[i]));
        printf("trial %d: query 0 inside a 3-query batch vs alone: %s (max abs diff %.3e)\n", trial, same ? "BIT-IDENTICAL" : "DIFFERENT", md);
        bad += !same;
    }
    printf("%s\n", bad ? "NOT INVARIANT" : "INVARIANT");
    return bad != 0;
}
