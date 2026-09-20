// fa_mqa_check.cpp -- CPU flash attention for an MQA cache (one 512-wide K/V head, V aliases K, f16 mask row per query)
// against a float64 reference: relative RMS error and op time. This is the reproducer for the FP16 V accumulation error.
//
// Build against any ggml build tree (upstream works):
//   c++ -std=c++17 -O2 -I<llama.cpp>/ggml/include fa_mqa_check.cpp -L<build>/bin -lggml-cpu -lggml-base -ldl -o fa_mqa_check
// Run (one socket, 15 workers):   printf '\0\0\0\0' > /dev/shm/ctl.u32
//   LD_LIBRARY_PATH=<build>/bin numactl --cpunodebind=1 --membind=1 taskset -c 16-30 ./fa_mqa_check /dev/shm/ctl.u32 4096 2051 3 16 15
// Arguments: CONTROL n_kv n_visible n_queries n_heads threads [reps]. CONTROL is the 4-byte GGML_F18_CONTROL_FILE: the program
// writes 0 then 1 into it, so a library with patches/glm5next-fa-mqa-cellsplit.patch runs its stock path ("mode 0") and the new
// kernel ("mode 1") in one process. A stock library ignores the file and both lines measure the stock kernel.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

static uint32_t rng = 0x9e3779b9;
static uint32_t rb() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float rf() { return (int(rb() % 20001) - 10000) / 10000.f; }

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: test_fa CONTROL n_kv n_sel N H threads [reps]\n"); return 1; }
    const char * control = argv[1];
    const int n_kv = atoi(argv[2]), n_sel = atoi(argv[3]), N = atoi(argv[4]), H = atoi(argv[5]), threads = atoi(argv[6]);
    const int reps = argc > 7 ? atoi(argv[7]) : 200;
    const int D = 512;

    int fd = open(control, O_RDWR); if (fd < 0) { perror("control"); return 1; }
    uint32_t * ctl = (uint32_t *) mmap(nullptr, 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N, H);
    ggml_tensor * k    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, N);
    ggml_tensor * v    = ggml_view_3d(ctx, k, D, n_kv, 1, k->nb[1], k->nb[2], 0);
    const float scale = 1.0f/sqrtf(256.0f);
    ggml_tensor * out  = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    std::vector<float> qd((size_t) D*N*H); for (auto & x : qd) x = rf();
    std::vector<float> kd((size_t) D*n_kv); for (auto & x : kd) x = rf()*0.5f;
    std::vector<ggml_fp16_t> k16(kd.size()); ggml_fp32_to_fp16_row(kd.data(), k16.data(), (int64_t) kd.size());
    ggml_fp16_to_fp32_row(k16.data(), kd.data(), (int64_t) kd.size()); // reference sees what the cache holds
    // selection: n_sel random cells per query, slightly different per query, plus a causal tail
    std::vector<ggml_fp16_t> md((size_t) n_kv*N);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY), zero = ggml_fp32_to_fp16(0.0f);
    std::vector<int> perm(n_kv); for (int i = 0; i < n_kv; ++i) perm[i] = i;
    for (int i = n_kv - 1; i > 0; --i) std::swap(perm[i], perm[rb() % (i + 1)]);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < n_kv; ++c) md[(size_t) n*n_kv + c] = ninf;
        for (int s = 0; s < n_sel; ++s) md[(size_t) n*n_kv + perm[(s + 3*n) % n_kv]] = zero;
    }
    ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k16.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(mask, md.data(), 0, ggml_nbytes(mask));

    // float64 reference: out[n][h] = softmax(q.k*scale + mask) . k
    std::vector<double> ref((size_t) D*N*H);
    for (int n = 0; n < N; ++n) for (int h = 0; h < H; ++h) {
        const float * qr = qd.data() + ((size_t) h*N + n)*D; // q layout [D, N, H]
        std::vector<double> s(n_kv, -INFINITY); double mx = -INFINITY;
        for (int c = 0; c < n_kv; ++c) {
            if (md[(size_t) n*n_kv + c] == ninf) continue;
            double a = 0; const float * kr = kd.data() + (size_t) c*D;
            for (int d = 0; d < D; ++d) a += (double) qr[d]*kr[d];
            s[c] = a*scale; mx = std::max(mx, s[c]);
        }
        double S = 0; std::vector<double> acc(D, 0.0);
        for (int c = 0; c < n_kv; ++c) { if (s[c] == -INFINITY) continue; double w = exp(s[c] - mx); S += w; const float * kr = kd.data() + (size_t) c*D; for (int d = 0; d < D; ++d) acc[d] += w*kr[d]; }
        for (int d = 0; d < D; ++d) ref[((size_t) n*H + h)*D + d] = acc[d]/S; // dst layout [D, H, N]
    }

    std::vector<float> res((size_t) D*N*H);
    for (uint32_t mode : {0u, 1u}) {
        *ctl = mode; msync(ctl, 4, MS_SYNC);
        for (int i = 0; i < 10; ++i) ggml_backend_graph_compute(backend, gf);
        std::vector<double> times;
        for (int i = 0; i < reps; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, gf);
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(times.begin(), times.end());
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        double emax = 0, rmax = 0, rms = 0, refrms = 0;
        for (size_t i = 0; i < res.size(); ++i) { double e = fabs(res[i] - ref[i]); emax = std::max(emax, e); rms += e*e; refrms += ref[i]*ref[i]; }
        rmax = sqrt(rms/refrms);
        printf("mode %u: median %.3f ms  p10 %.3f ms  min %.3f ms | max abs err %.3e  rel rms err %.3e\n", mode, times[times.size()/2], times[times.size()/10], times[0], emax, rmax);
    }
    *ctl = 0; msync(ctl, 4, MS_SYNC);
    return 0;
}
