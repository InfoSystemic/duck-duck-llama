#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>
#include <vector>
#include "qwen-quantize-blocks-0909.h"

static ggml_from_float_t original_quantize;
static char * destination;
static size_t output_stride, block_bytes;
static int64_t block_elements, blocks_per_row, row_count;
static std::atomic<unsigned> * claims;
static thread_local uint64_t written_blocks;

static void require(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}

static void counted_quantize(const float * x, void * y, int64_t k) {
    const ptrdiff_t offset = (char *) y - destination;
    require(offset >= 0 && size_t(offset) < output_stride * row_count, "Output pointer is outside rows");
    const int64_t row = offset / output_stride;
    const size_t inside = offset % output_stride;
    require(inside % block_bytes == 0 && k > 0 && k % block_elements == 0, "Misaligned block range");
    const int64_t first = inside / block_bytes, count = k / block_elements;
    require(first + count <= blocks_per_row, "A quantizer call crosses a row");
    for (int64_t b = first; b < first + count; ++b) {
        require(claims[row * blocks_per_row + b].fetch_add(1, std::memory_order_relaxed) == 0, "Block written twice");
    }
    written_blocks += count;
    original_quantize(x, y, k);
}

static float probe_value(int probe, int64_t row, int64_t column) {
    const int value = int((column * 73 + row * 131) % 511) - 255;
    if (probe == 0) return 0.0f;
    if (probe == 1) return -0.0f;
    if (probe == 2) return column % 2 ? 1.0f : -1.0f;
    if (probe == 3) return float(value) * 0.03125f;
    return float(value) * ((column / 256) % 2 ? 1e-8f : 1e4f);
}

int main() {
    ggml_cpu_init();
    Dl_info runtime{};
    require(dladdr(reinterpret_cast<void *>(ggml_get_type_traits_cpu), &runtime), "Missing CPU binding");
    uint64_t cases = 0, bytes_compared = 0, blocks_checked = 0, parallel_cases = 0;
    for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q8_K})
    for (int64_t width : {256, 2560, 16384})
    for (const auto & shape : {std::array<int64_t, 3>{1,1,1}, {1,5,1}, {3,5,2}, {64,1,1}})
    for (int padding : {0, 7})
    for (int probe = 0; probe < 5; ++probe)
    for (int workers : {1, 4, 15}) {
        ggml_tensor src{};
        src.type = GGML_TYPE_F32;
        src.ne[0] = width;
        for (int d = 1; d < 4; ++d) src.ne[d] = shape[d - 1];
        src.nb[0] = sizeof(float);
        src.nb[1] = (width + padding) * sizeof(float);
        src.nb[2] = src.nb[1] * src.ne[1] + padding * sizeof(float);
        src.nb[3] = src.nb[2] * src.ne[2] + padding * sizeof(float);
        std::vector<float> storage(src.nb[3] * src.ne[3] / sizeof(float), -1234.5f);
        src.data = storage.data();
        row_count = src.ne[1] * src.ne[2] * src.ne[3];
        auto row_pointer = [&](int64_t row) {
            const size_t offset = (row % src.ne[1]) * src.nb[1] +
                ((row / src.ne[1]) % src.ne[2]) * src.nb[2] +
                (row / (src.ne[1] * src.ne[2])) * src.nb[3];
            return (float *) ((char *) src.data + offset);
        };
        for (int64_t row = 0; row < row_count; ++row)
            for (int64_t column = 0; column < width; ++column)
                row_pointer(row)[column] = probe_value(probe, row, column);
        const auto saved = storage;
        block_elements = ggml_blck_size(type);
        block_bytes = ggml_type_size(type);
        blocks_per_row = width / block_elements;
        output_stride = ggml_row_size(type, width) + (padding ? 64 : 0);
        std::vector<uint8_t> expected(output_stride * row_count + 128, 0xa5), actual = expected;
        original_quantize = ggml_get_type_traits_cpu(type)->from_float;
        for (int64_t row = 0; row < row_count; ++row)
            original_quantize(row_pointer(row), expected.data() + 64 + row * output_stride, width);
        destination = (char *) actual.data() + 64;
        std::vector<std::atomic<unsigned>> coverage(row_count * blocks_per_row);
        for (auto & value : coverage) value.store(0, std::memory_order_relaxed);
        claims = coverage.data();
        std::vector<uint64_t> worker_blocks(workers);
        std::vector<std::thread> threads;
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                ggml_compute_params params{};
                params.ith = worker; params.nth = workers;
                written_blocks = 0;
                qwen_quantize_blocks(&params, &src, type, counted_quantize, destination, output_stride);
                worker_blocks[worker] = written_blocks;
            });
        }
        for (auto & thread : threads) thread.join();
        require(expected == actual, "Quantized bytes or output padding differ");
        require(std::memcmp(saved.data(), storage.data(), storage.size() * sizeof(float)) == 0, "Input or padding changed");
        for (const auto & value : coverage) require(value.load() == 1, "Missing quantized block");
        const int active = std::count_if(worker_blocks.begin(), worker_blocks.end(), [](uint64_t value) { return value > 0; });
        require(active == std::min<int64_t>(workers, row_count * blocks_per_row), "Unexpected worker coverage");
        parallel_cases += active > 1;
        bytes_compared += output_stride * row_count;
        blocks_checked += row_count * blocks_per_row;
        ++cases;
    }
    std::printf("{\"passed\":true,\"cases\":%llu,\"bytes_compared\":%llu,\"blocks_checked\":%llu,\"parallel_cases\":%llu,\"cpu_library\":\"%s\",\"exact_bytes\":true,\"single_writer_per_block\":true,\"inputs_and_padding_preserved\":true}\n",
                (unsigned long long) cases, (unsigned long long) bytes_compared,
                (unsigned long long) blocks_checked, (unsigned long long) parallel_cases, runtime.dli_fname);
}
