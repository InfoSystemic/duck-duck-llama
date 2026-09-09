#include "qwen-q6-packed-batch-0908.h"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <sched.h>
#include <vector>

__attribute__((noinline,noclone)) static void batched_kernel(int n, int rows,
        const block_q6_K_x16 * w, const block_q8_K * const * activation,
        float * const * output, int count) {
    qwen_q6_packed_batch(n, rows, w, activation, output, count);
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    const int rows = std::atoi(argv[1]), matrices = std::atoi(argv[2]);
    const int activations = std::atoi(argv[3]);
    const double duration = std::atof(argv[4]);
    if (activations != 1 && activations != 2 && activations != 3 && activations != 5) return 2;
    if ((rows != 32 && rows != 64) || (matrices != 1 && matrices != 512) || duration < .2 || duration > 2) return 2;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0,sizeof(mask),&mask) || CPU_COUNT(&mask) != 128) return 3;
    CPU_ZERO(&mask); CPU_SET(127,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)) return 3;
    Dl_info info{};
    if (!dladdr((void *) ggml_gemv_q6_K_x16_q8_K, &info) || !info.dli_fname) return 5;
    constexpr int n = 2560, blocks = n / 256;
    const size_t count = size_t(rows/16) * blocks;
    std::vector<block_q6_K_x16> weights(count * matrices);
    std::vector<block_q8_K> y(blocks * activations);
    std::vector<const block_q8_K *> activation(activations);
    std::vector<float> outputs(rows * activations), expected(rows * activations);
    std::vector<float *> output(activations);
    for (int r = 0; r < activations; ++r) {
        activation[r] = y.data() + r * blocks;
        output[r] = outputs.data() + r * rows;
    }
    std::vector<int> order(matrices);
    std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(984291);
    std::shuffle(order.begin(),order.end(),rng);
    for (auto & w : weights) {
        for (auto & d : w.d) d = ggml_fp32_to_fp16(.003f * (rng()%13+1));
        for (auto & scales : w.scales) for (auto & s : scales) s = int(rng()%256)-128;
        for (auto & q : w.q) q = rng()%256;
    }
    for (auto & a : y) {
        a.d = .004f;
        for (auto & q : a.qs) q = int(rng()%256)-128;
        for (int sub = 0; sub < 16; ++sub) {
            int sum = 0;
            for (int j = 0; j < 16; ++j) sum += a.qs[sub*16+j];
            a.bsums[sub] = sum;
        }
    }
    for (int matrix = 0; matrix < matrices; ++matrix) {
        const auto * w = weights.data() + matrix * count;
        for (int r = 0; r < activations; ++r)
            ggml_gemv_q6_K_x16_q8_K(n,expected.data()+r*rows,0,w,activation[r],1,rows);
        batched_kernel(n,rows,w,activation.data(),output.data(),activations);
        if (std::memcmp(outputs.data(),expected.data(),outputs.size()*sizeof(float))) return 4;
    }
    volatile float checksum = 0;
    std::printf("{\"n\":%d,\"rows\":%d,\"matrices\":%d,\"cpu\":127,\"weight_working_set_bytes\":%zu,\"weight_storage_unchanged\":true,\"output_equality_checked\":true,\"runs\":[",n,rows,matrices,weights.size()*sizeof(weights[0]));
    int run = 0;
    for (int variant : {0,1,1,0}) {
        size_t iterations = 0;
        const auto start = std::chrono::steady_clock::now();
        double elapsed;
        do {
            for (int step = 0; step < 32; ++step) {
                const auto * w = weights.data() + order[iterations % matrices] * count;
                if (variant) batched_kernel(n,rows,w,activation.data(),output.data(),activations);
                else for (int r = 0; r < activations; ++r)
                    ggml_gemv_q6_K_x16_q8_K(n,output[r],0,w,activation[r],1,rows);
                for (int r = 0; r < activations; ++r)
                    checksum = checksum + output[r][(iterations + r) % rows];
                ++iterations;
            }
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        } while (elapsed < duration);
        std::printf("%s{\"variant\":\"%s\",\"iterations\":%zu,\"seconds\":%.9f,\"matrices_per_second\":%.9f}",run++ ? "," : "",variant ? "packed_batch" : "native",iterations,elapsed,iterations/elapsed);
    }
    std::printf("],\"activations_per_matrix\":%d,\"shuffled_matrix_order\":true,\"checksum\":%.9g,\"cpu_library\":\"%s\",\"scope\":\"Single-core component timings only; no model tok/s or DRAM utilization claim\"}\n",activations,double(checksum),info.dli_fname);
}
