#include "qwen-q6-lossless-bytes-0908.h"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sched.h>
#include <vector>

__attribute__((noinline,noclone)) static void expanded_kernel(int n, float * s,
        const qwen_q6_bytes_x16 * w, const block_q8_K * y, int rows) {
    qwen_gemv_q6_bytes_x16_q8_K(n,s,w,y,rows);
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    const int rows = std::atoi(argv[1]), matrices = std::atoi(argv[2]);
    const int activations = std::atoi(argv[3]);
    const double duration = std::atof(argv[4]);
    if (activations != 1 && activations != 3 && activations != 5) return 2;
    if ((rows != 32 && rows != 64 && rows != 160) || (matrices != 1 && matrices != 512) || duration < .2 || duration > 2) return 2;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0,sizeof(mask),&mask) || CPU_COUNT(&mask) != 128) return 3;
    CPU_ZERO(&mask); CPU_SET(127,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)) return 3;
    constexpr int n = 2560, blocks = n / 256;
    const size_t count = size_t(rows/16) * blocks;
    std::vector<block_q6_K_x16> native(count * matrices);
    std::vector<qwen_q6_bytes_x16> expanded(native.size());
    std::vector<block_q8_K> y(blocks * activations);
    std::vector<int> order(matrices);
    std::iota(order.begin(),order.end(),0);
    std::vector<float> output(rows), check(rows);
    std::mt19937 rng(984291);
    std::shuffle(order.begin(),order.end(),rng);
    for (auto & w : native) {
        for (auto & d : w.d) d = ggml_fp32_to_fp16(.003f * (rng()%13+1));
        for (auto & scales : w.scales) for (auto & s : scales) s = int(rng()%256)-128;
        for (auto & q : w.q) q = rng()%256;
    }
    qwen_q6_expand_x16(native.data(),expanded.data(),native.size());
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
        for (int activation = 0; activation < activations; ++activation) {
            const block_q8_K * a = y.data() + activation * blocks;
            ggml_gemv_q6_K_x16_q8_K(n,output.data(),0,native.data()+matrix*count,a,1,rows);
            expanded_kernel(n,check.data(),expanded.data()+matrix*count,a,rows);
            if (std::memcmp(output.data(),check.data(),rows*sizeof(float))) return 4;
        }
    }
    volatile float checksum = 0;
    std::printf("{\"n\":%d,\"rows\":%d,\"matrices\":%d,\"cpu\":127,\"native_working_set_bytes\":%zu,\"expanded_working_set_bytes\":%zu,\"output_equality_checked\":true,\"runs\":[",n,rows,matrices,native.size()*sizeof(native[0]),expanded.size()*sizeof(expanded[0]));
    int run = 0;
    for (int variant : {0,1,1,0}) {
        size_t iterations = 0;
        const auto start = std::chrono::steady_clock::now();
        double elapsed;
        do {
            for (int step = 0; step < 32; ++step) {
                const size_t matrix = order[iterations % matrices];
                for (int activation = 0; activation < activations; ++activation) {
                    const block_q8_K * a = y.data() + activation * blocks;
                    if (variant) expanded_kernel(n,output.data(),expanded.data()+matrix*count,a,rows);
                    else ggml_gemv_q6_K_x16_q8_K(n,output.data(),0,native.data()+matrix*count,a,1,rows);
                    checksum = checksum + output[(iterations + activation) % rows];
                }
                ++iterations;
            }
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        } while (elapsed < duration);
        std::printf("%s{\"variant\":\"%s\",\"iterations\":%zu,\"seconds\":%.9f,\"matrices_per_second\":%.9f}",run++ ? "," : "",variant ? "expanded" : "native",iterations,elapsed,iterations/elapsed);
    }
    std::printf("],\"activations_per_matrix\":%d,\"shuffled_matrix_order\":true,\"checksum\":%.9g,\"scope\":\"Single-core component timings only; no model tok/s or DRAM utilization claim\"}\n",activations,double(checksum));
}
