// fa-bench: time ggml's CPU FLASH_ATTN_EXT at MiMo-V2.6-Pro's per-NUMA-node decode shapes and check it against a
// float64 reference. Layouts match llama.cpp's KV cache under tensor parallelism (one node holds 32 Q heads and
// 2 KV heads): K/V cells hold both KV heads contiguously and are viewed permuted, Q is [DK, rows, heads] permuted,
// the mask is causal over the last `rows` cells, output is [DV, heads, rows].
//
// usage: fa-bench <n_kv> <rows> [threads 15] [iters 5] [n_head 32] [n_head_kv 2] [DK 192] [DV 128] [sinks 0|1] [window 0]
// env:   whatever GGML_CPU_* knobs the kernel under test reads (e.g. GGML_CPU_FA_GQA=1)
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cstdint>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <n_kv> <rows> [threads] [iters] [n_head] [n_head_kv] [DK] [DV] [sinks]\n", argv[0]);
        return 1;
    }
    const int n_kv      = atoi(argv[1]);
    const int rows      = atoi(argv[2]);
    const int n_threads = argc > 3 ? atoi(argv[3]) : 15;
    const int iters     = argc > 4 ? atoi(argv[4]) : 5;
    const int n_head    = argc > 5 ? atoi(argv[5]) : 32;
    const int n_head_kv = argc > 6 ? atoi(argv[6]) : 2;
    const int DK        = argc > 7 ? atoi(argv[7]) : 192;
    const int DV        = argc > 8 ? atoi(argv[8]) : 128;
    const int use_sinks = argc > 9 ? atoi(argv[9]) : 0;
    const int window    = argc > 10 ? atoi(argv[10]) : 0;   // >0: row sees only the last `window` cells (SWA)

    const size_t mem = (size_t) n_kv * n_head_kv * (DK + DV) * 2      // K, V cache
                     + (size_t) rows * n_head * (DK + DV) * 4 * 2       // q, out
                     + (size_t) n_kv * rows * 2 + 256 * 1024 * 1024;    // mask + graph/work slack
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // caches: one cell = all KV heads of this node, as in llama_kv_cache
    ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, DK * n_head_kv, n_kv);
    ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, DV * n_head_kv, n_kv);
    ggml_tensor * qc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DK, n_head, rows);
    ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, rows);
    ggml_tensor * sk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> kf((size_t) DK * n_head_kv * n_kv), vf((size_t) DV * n_head_kv * n_kv), qf((size_t) DK * n_head * rows);
    // K gets a per-head offset so scores are not all near zero (a flat softmax hides accumulation errors)
    for (size_t i = 0; i < kf.size(); i++) kf[i] = nd(rng) * 0.5f + ((i / DK) % 7 == 0 ? 0.3f : 0.0f);
    for (auto & x : vf) x = nd(rng);
    for (auto & x : qf) x = nd(rng) * 0.5f;
    for (size_t i = 0; i < kf.size(); i++) ((ggml_fp16_t *) kc->data)[i] = ggml_fp32_to_fp16(kf[i]);
    for (size_t i = 0; i < vf.size(); i++) ((ggml_fp16_t *) vc->data)[i] = ggml_fp32_to_fp16(vf[i]);
    // reference must see exactly the f16-rounded values the kernel sees
    for (size_t i = 0; i < kf.size(); i++) kf[i] = ggml_fp16_to_fp32(((ggml_fp16_t *) kc->data)[i]);
    for (size_t i = 0; i < vf.size(); i++) vf[i] = ggml_fp16_to_fp32(((ggml_fp16_t *) vc->data)[i]);
    memcpy(qc->data, qf.data(), qf.size() * 4);
    for (int r = 0; r < rows; r++) {
        const int last = n_kv - rows + r;  // causal: row r is the token at cell `last`
        for (int c = 0; c < n_kv; c++) {
            const bool vis = c <= last && (window <= 0 || last - c < window);
            ((ggml_fp16_t *) mk->data)[(size_t) r * n_kv + c] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
        }
    }
    std::vector<float> sinkv(n_head);
    for (int h = 0; h < n_head; h++) { sinkv[h] = 0.5f * nd(rng); ((float *) sk->data)[h] = sinkv[h]; }

    ggml_tensor * k = ggml_view_3d(ctx, kc, DK, n_head_kv, n_kv, DK * 2, kc->nb[1], 0);
    ggml_tensor * v = ggml_view_3d(ctx, vc, DV, n_head_kv, n_kv, DV * 2, vc->nb[1], 0);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [DK, n_kv, n_head_kv]
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * q = ggml_permute(ctx, qc, 0, 2, 1, 3);  // [DK, rows, n_head]
    const float scale = 1.0f / sqrtf((float) DK);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mk, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (use_sinks) {
        ggml_flash_attn_ext_add_sinks(out, sk);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // plan once and reuse the work buffer (ggml_graph_compute_with_ctx would allocate a new one per call)
    ggml_cplan cplan = ggml_graph_plan(gf, n_threads, nullptr);
    std::vector<uint8_t> work(cplan.work_size + 64);
    cplan.work_data = work.data();
    ggml_graph_compute(gf, &cplan);   // warmup
    std::vector<double> ts;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_graph_compute(gf, &cplan);
        auto t1 = std::chrono::high_resolution_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    const double ms = ts[0];

    // float64 reference over every (row, head)
    const int gqa = n_head / n_head_kv;
    double err2 = 0, ref2 = 0, max_rel_row = 0;
    const float * o = (const float *) out->data;   // [DV, n_head, rows]
    std::vector<double> acc(DV);
    for (int r = 0; r < rows; r++) {
        const int last = n_kv - rows + r;
        for (int h = 0; h < n_head; h++) {
            const int hk = h / gqa;
            const float * qp = qf.data() + ((size_t) r * n_head + h) * DK;
            double M = use_sinks ? sinkv[h] : -INFINITY, S = use_sinks ? 1.0 : 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int c = (window > 0 ? std::max(0, last - window + 1) : 0); c <= last; c++) {
                const float * kp = kf.data() + ((size_t) c * n_head_kv + hk) * DK;
                double s = 0; for (int d = 0; d < DK; d++) s += (double) qp[d] * kp[d];
                s *= scale;
                const float * vp = vf.data() + ((size_t) c * n_head_kv + hk) * DV;
                if (s > M) { double ms_ = std::exp(M - s); for (auto & a : acc) a *= ms_; S = S * ms_ + 1.0; M = s;
                             for (int d = 0; d < DV; d++) acc[d] += vp[d]; }
                else { double w = std::exp(s - M); S += w; for (int d = 0; d < DV; d++) acc[d] += w * vp[d]; }
            }
            double e_row = 0, r_row = 0;
            for (int d = 0; d < DV; d++) {
                const double ref = acc[d] / S;
                const double got = o[((size_t) r * n_head + h) * DV + d];
                e_row += (got - ref) * (got - ref); r_row += ref * ref;
            }
            err2 += e_row; ref2 += r_row;
            max_rel_row = std::max(max_rel_row, std::sqrt(e_row / std::max(r_row, 1e-300)));
        }
    }
    const double kv_gb   = (double) n_kv * n_head_kv * (DK + DV) * 2 / 1e9;
    const double gflop   = 2.0 * n_kv * rows * n_head * (DK + DV) / 1e9;
    uint64_t h = 1469598103934665603ull;   // FNV-1a over the output bytes: equal hash == bit-identical output
    for (size_t i = 0; i < ggml_nbytes(out); i++) { h ^= ((const unsigned char *) out->data)[i]; h *= 1099511628211ull; }
    printf("hash %016llx  ", (unsigned long long) h);
    printf("n_kv %6d rows %d threads %d: %8.3f ms (median %8.3f)  KV %.3f GB -> %6.1f GB/s  %7.1f GFLOP/s  "
           "rel.err %.3e (worst row %.3e)\n",
           n_kv, rows, n_threads, ms, ts[ts.size() / 2], kv_gb, kv_gb / (ms / 1e3), gflop / (ms / 1e3),
           std::sqrt(err2 / ref2), max_rel_row);
    ggml_free(ctx);
    return 0;
}
