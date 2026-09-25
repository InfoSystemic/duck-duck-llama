// li_fast_check.cpp <libggml-cpu.so> [n_kv] [reps] -- bit-compare and time the lightning indexer kernel with F18_LI_FAST off and on.
//
// Loads the library, builds q [128, 32, 3], k [128, 1, n_kv] f32, w [32, 3], mask f16 [n_kv, 3] with random values (some pools
// masked to -inf), runs ggml_compute_forward_lightning_indexer single-threaded through a control file that flips bit 4 between
// runs, and compares the outputs byte for byte. Then times both variants.
//
// build: c++ -std=c++17 -O2 -I$ENG/ggml/include -I$ENG/ggml/src -I$ENG/ggml/src/ggml-cpu li_fast_check.cpp -ldl -o li_fast_check
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <random>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cpu-impl.h"   // ggml_compute_params

extern "C" {
typedef void (*li_fn)(const struct ggml_compute_params *, struct ggml_tensor *);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <libggml-cpu.so> [n_kv] [reps]\n", argv[0]); return 2; }
    const int n_kv = argc > 2 ? atoi(argv[2]) : 34178;
    const int reps = argc > 3 ? atoi(argv[3]) : 20;
    const char * ctl = "/dev/shm/li-fast-check.u32";
    { FILE * f = fopen(ctl, "wb"); uint32_t z = 0; fwrite(&z, 4, 1, f); fclose(f); }
    setenv("GGML_F18_CONTROL_FILE", ctl, 1);
    void * lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto li = (li_fn) dlsym(lib, "ggml_compute_forward_lightning_indexer");
    if (!li) { fprintf(stderr, "symbol missing\n"); return 2; }
    const int fd = open(ctl, O_RDWR); uint32_t * word = (uint32_t *) mmap(nullptr, 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); close(fd);

    ggml_init_params ip = { (size_t(64) << 20) + size_t(n_kv) * 2048, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 32, 3, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 1, n_kv, 1);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 3, 1, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 3, 1, 1);
    ggml_tensor * d0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, 3, 1, 1);
    ggml_tensor * d1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, 3, 1, 1);
    std::mt19937_64 rng(42); std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < 128*32*3; ++i) ((float *) q->data)[i] = nd(rng);
    for (int i = 0; i < 128*n_kv; ++i) ((float *) k->data)[i] = nd(rng);
    for (int i = 0; i < 32*3; ++i) ((float *) w->data)[i] = nd(rng) * 0.1f;
    std::uniform_int_distribution<int> coin(0, 9);
    for (int i = 0; i < n_kv*3; ++i) ((ggml_fp16_t *) m->data)[i] = ggml_fp32_to_fp16(coin(rng) == 0 ? -INFINITY : 0.0f);
    for (ggml_tensor * d : {d0, d1}) { d->op = GGML_OP_LIGHTNING_INDEXER; d->src[0] = q; d->src[1] = k; d->src[2] = w; d->src[3] = m; }

    ggml_compute_params params = {}; params.ith = 0; params.nth = 1;
    std::vector<char> wdata((128 + 64) * sizeof(float) * 2); params.wdata = wdata.data(); params.wsize = wdata.size();

    auto run = [&](uint32_t bits, ggml_tensor * d) { __atomic_store_n(word, bits, __ATOMIC_RELEASE); li(&params, d); };
    run(0, d0); run(16, d1);
    const int same = memcmp(d0->data, d1->data, ggml_nbytes(d0)) == 0;
    int mism = 0; for (int i = 0; i < n_kv*3; ++i) if (((float *) d0->data)[i] != ((float *) d1->data)[i]) ++mism;
    printf("n_kv %d: outputs %s (%d of %d values differ)\n", n_kv, same ? "IDENTICAL" : "DIFFER", mism, n_kv*3);
    for (uint32_t bits : {0u, 16u}) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) run(bits, d0);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        printf("  F18_LI_FAST %s: %.3f ms per call (1 thread), %.1f GFLOP/s\n", bits ? "on " : "off", ms, 2.0*128*32*3*n_kv/ms/1e6);
    }
    return same ? 0 : 1;
}
