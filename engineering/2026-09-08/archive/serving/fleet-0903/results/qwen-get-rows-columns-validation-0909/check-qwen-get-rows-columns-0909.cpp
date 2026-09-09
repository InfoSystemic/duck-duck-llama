#include "ggml.h"
#include "ggml-cpu-impl.h"
#include "ops.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

static bool fixed = false;
static bool enabled = false;
static int cases = 0;
static int known_failures = 0;
static int unexpected = 0;
static uint64_t checked_values = 0;

static uint64_t hash_bytes(const void * data, size_t n) {
    auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t value = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; ++i) value = (value ^ bytes[i]) * UINT64_C(1099511628211);
    return value;
}

static void check(ggml_type type, int64_t columns, int64_t rows, int groups, bool padded, int threads, bool ownership) {
    const bool quant = ggml_is_quantized(type);
    auto * ctx = ggml_init({ggml_tensor_overhead() * 32, nullptr, true});
    GGML_ASSERT(ctx);
    const int64_t physical_columns = columns + (padded ? (quant ? 512 : 16) : 0);
    const int64_t physical_rows = quant ? 24 : 7;
    auto * storage = ggml_new_tensor_4d(ctx, type, physical_columns, physical_rows, groups, groups);
    std::vector<uint8_t> source_bytes(ggml_nbytes(storage));
    if (quant) {
        std::vector<float> values(ggml_nelements(storage));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i * 131 + i / 97) % 1021) - 510) / 79.0f;
        GGML_ASSERT(ggml_quantize_chunk(type, values.data(), source_bytes.data(), 0, physical_rows * groups * groups, physical_columns, nullptr) == source_bytes.size());
    } else {
        for (size_t i = 0; i < source_bytes.size() / 4; ++i) {
            uint32_t value = (uint32_t(i + 137) * 2654435761U & 0x807fffffU) | 0x3f000000U;
            if (i % 101 == 0) value = 0x7fc12345U;
            std::memcpy(source_bytes.data() + 4 * i, &value, 4);
        }
    }
    storage->data = source_bytes.data();
    const size_t offset = padded ? (quant ? ggml_row_size(type, 256) : 2 * sizeof(float)) : 0;
    auto * source = ggml_view_4d(ctx, storage, columns, 5, groups, groups, storage->nb[1], storage->nb[2], storage->nb[3], offset);
    const int64_t physical_ids = rows + (padded ? 1 : 0);
    auto * ids_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, physical_ids, groups, groups);
    std::vector<int32_t> indices(ggml_nelements(ids_storage));
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = (i * 3 + 1) % 5;
    ids_storage->data = indices.data();
    auto * ids = padded ? ggml_view_3d(ctx, ids_storage, rows, groups, groups, ids_storage->nb[1], ids_storage->nb[2], sizeof(int32_t)) : ids_storage;
    auto * output = ggml_get_rows(ctx, source, ids);
    const size_t count = ggml_nelements(output);
    const uint32_t sentinel = 0x7f801234U;
    std::vector<uint32_t> actual(count, sentinel), reference(count);
    for (int i2 = 0; i2 < groups; ++i2) for (int i1 = 0; i1 < groups; ++i1) for (int64_t i0 = 0; i0 < rows; ++i0) {
        int32_t index;
        std::memcpy(&index, static_cast<const char *>(ids->data) + i0 * ids->nb[0] + i1 * ids->nb[1] + i2 * ids->nb[2], 4);
        const char * in = static_cast<const char *>(source->data) + index * source->nb[1] + i1 * source->nb[2] + i2 * source->nb[3];
        char * out = reinterpret_cast<char *>(reference.data()) + i0 * output->nb[1] + i1 * output->nb[2] + i2 * output->nb[3];
        if (quant) ggml_get_type_traits(type)->to_float(in, reinterpret_cast<float *>(out), columns);
        else std::memcpy(out, in, columns * 4);
    }
    const uint64_t source_before = hash_bytes(source_bytes.data(), source_bytes.size());
    const uint64_t ids_before = hash_bytes(indices.data(), indices.size() * 4);
    output->data = actual.data();
    ggml_compute_params params = {};
    params.nth = threads;
    std::vector<size_t> writes;
    std::vector<uint8_t> owners;
    std::vector<uint32_t> combined;
    if (ownership) {
        GGML_ASSERT(!quant && type == GGML_TYPE_F32);
        owners.resize(count);
        combined.resize(count, sentinel);
    }
    for (params.ith = 0; params.ith < threads; ++params.ith) {
        if (ownership) std::fill(actual.begin(), actual.end(), sentinel);
        ggml_compute_forward_get_rows(&params, output);
        if (ownership) {
            size_t written = 0;
            for (size_t i = 0; i < count; ++i) if (actual[i] != sentinel) {
                GGML_ASSERT(!owners[i]);
                owners[i] = 1;
                combined[i] = actual[i];
                ++written;
            }
            writes.push_back(written);
        }
    }
    if (ownership) actual = std::move(combined);
    const bool equal = actual == reference;
    const bool known_bug = quant && columns >= 4096 && enabled && !fixed;
    if (known_bug && !equal) ++known_failures;
    else if (!equal || known_bug) ++unexpected;
    GGML_ASSERT(hash_bytes(source_bytes.data(), source_bytes.size()) == source_before);
    GGML_ASSERT(hash_bytes(indices.data(), indices.size() * 4) == ids_before);
    if (ownership) {
        const size_t active = std::count_if(writes.begin(), writes.end(), [](size_t n) { return n != 0; });
        if (fixed && enabled) GGML_ASSERT(active == size_t(threads));
        std::printf("{\"ownership\":true,\"columns\":%lld,\"rows\":%lld,\"threads\":%d,\"active_workers\":%zu,\"min_worker_values\":%zu,\"max_worker_values\":%zu,\"exact\":%s}\n", (long long) columns, (long long) rows, threads, active, *std::min_element(writes.begin(), writes.end()), *std::max_element(writes.begin(), writes.end()), equal ? "true" : "false");
    }
    if (!equal) std::printf("{\"mismatch\":true,\"type\":\"%s\",\"columns\":%lld,\"rows\":%lld,\"groups\":%d,\"padded\":%s,\"threads\":%d,\"expected_parent_bug\":%s}\n", ggml_type_name(type), (long long) columns, (long long) rows, groups, padded ? "true" : "false", threads, known_bug ? "true" : "false");
    ++cases;
    checked_values += count;
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    fixed = std::strcmp(argv[1], "fixed") == 0;
    GGML_ASSERT(fixed || std::strcmp(argv[1], "parent") == 0);
    enabled = ggml_cpu_parallel_copy_enabled();
    Dl_info library = {};
    GGML_ASSERT(dladdr(reinterpret_cast<void *>(ggml_compute_forward_get_rows), &library));
    std::printf("{\"cpu_library\":\"%s\",\"fixed\":%s,\"parallel_copy\":%s}\n", library.dli_fname, fixed ? "true" : "false", enabled ? "true" : "false");
    for (int threads : {1, 3, 15}) for (int64_t columns : {17, 4095, 4096, 4097, 262144, 786432}) {
        for (int64_t rows : {1, 5}) for (bool padded : {false, true}) {
            check(GGML_TYPE_F32, columns, rows, columns <= 4097 ? 2 : 1, padded, threads, false);
        }
    }
    for (int threads : {1, 15}) for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K}) {
        for (int64_t columns : {2560, 4096, 5120}) for (bool padded : {false, true}) {
            check(type, columns, 3, 2, padded, threads, false);
        }
    }
    for (int threads : {1, 15}) for (bool padded : {false, true}) check(GGML_TYPE_I32, 4097, 5, 2, padded, threads, false);
    for (int64_t rows : {1, 5}) check(GGML_TYPE_F32, 786432, rows, 1, true, 15, true);
    std::printf("{\"summary\":true,\"cases\":%d,\"known_quantized_failures\":%d,\"unexpected_failures\":%d,\"checked_values\":%llu,\"inputs_preserved\":true}\n", cases, known_failures, unexpected, (unsigned long long) checked_values);
    return unexpected ? 1 : 0;
}
