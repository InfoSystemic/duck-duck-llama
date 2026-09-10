// SPDX-License-Identifier: MIT
// Compose the frozen, separately validated hash and native FP8 decode components.
#include "deepseek-v41-engram-lookup-0910.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

extern "C" {
void * deepseek_v41_hash_create(const uint32_t *, int64_t, int64_t, int64_t,
    const uint64_t *, const uint64_t *, const uint64_t *, int64_t, int);
void deepseek_v41_hash_destroy(void *);
void * deepseek_v41_hash_clone(const void *);
int64_t deepseek_v41_hash_length(const void *);
int deepseek_v41_hash_rewind(void *, int64_t);
int deepseek_v41_hash_forward(void *, const int64_t *, const uint8_t *, int64_t,
    int64_t, uint64_t *, uint64_t);
int deepseek_v41_engram_has_avx512();
int deepseek_v41_engram_gather_bf16(const uint8_t *, const uint8_t *, int64_t,
    const int64_t *, int64_t, uint16_t *, int);
}

namespace {
constexpr uint64_t columns = 48;
constexpr uint64_t width = 256;
struct hash_deleter { void operator()(void * p) const { deepseek_v41_hash_destroy(p); } };
using hash_pointer = std::unique_ptr<void, hash_deleter>;
struct tables {
    std::array<std::vector<deepseek_v41_lookup_shard>, 2> layers;
    int vectorize;
};
struct lookup {
    hash_pointer hash;
    std::shared_ptr<const tables> table;
    std::vector<uint64_t> ids;
    int64_t max_chunk;
};
}

extern "C" void * deepseek_v41_lookup_create(const deepseek_v41_lookup_config * config,
        const deepseek_v41_lookup_shard * shards, int64_t shard_count,
        int64_t max_sequence, int64_t max_chunk, int reciprocal, int vectorize) {
    if (!config || !config->table_rows || !shards || shard_count < 2 || shard_count > 128 ||
        max_sequence <= 0 || max_chunk <= 0 || max_chunk > max_sequence ||
        uint64_t(max_chunk) > std::numeric_limits<size_t>::max() / (columns * sizeof(uint64_t)) ||
        uint64_t(max_chunk) > uint64_t(INT64_MAX) / (columns * width) ||
        (vectorize != 0 && vectorize != 1) ||
        (vectorize && !deepseek_v41_engram_has_avx512())) return nullptr;
    try {
        auto table = std::make_shared<tables>();
        table->vectorize = vectorize;
        for (int64_t i = 0; i < shard_count; ++i) {
            const auto & s = shards[i];
            if (s.layer < 0 || s.layer > 1 || !s.rows || !s.weight || !s.scale ||
                s.rows > uint64_t(INT64_MAX) / width || s.rows > SIZE_MAX / width ||
                s.first_row > UINT64_MAX - s.rows ||
                s.weight_bytes < s.rows * width || s.scale_bytes < s.rows * 8) return nullptr;
            table->layers[s.layer].push_back(s);
        }
        for (int layer = 0; layer < 2; ++layer) {
            auto & parts = table->layers[layer];
            std::sort(parts.begin(), parts.end(), [](const auto & a, const auto & b) {
                return a.first_row < b.first_row;
            });
            uint64_t end = 0;
            for (const auto & s : parts) {
                if (s.first_row != end) return nullptr;
                end += s.rows;
            }
            if (!end || end != config->table_rows[layer]) return nullptr;
        }
        auto result = std::make_unique<lookup>();
        result->hash.reset(deepseek_v41_hash_create(config->token_map, config->vocabulary,
            config->compressed_vocabulary, config->pad_token, config->moduli,
            config->multipliers, config->table_rows, max_sequence, reciprocal));
        if (!result->hash) return nullptr;
        result->table = std::move(table);
        result->max_chunk = max_chunk;
        result->ids.resize(uint64_t(max_chunk) * columns);
        return result.release();
    } catch (...) { return nullptr; }
}

extern "C" void deepseek_v41_lookup_destroy(void * handle) {
    delete static_cast<lookup *>(handle);
}

extern "C" void * deepseek_v41_lookup_clone(const void * handle) {
    if (!handle) return nullptr;
    try {
        const auto & source = *static_cast<const lookup *>(handle);
        auto result = std::make_unique<lookup>();
        result->hash.reset(deepseek_v41_hash_clone(source.hash.get()));
        if (!result->hash) return nullptr;
        result->table = source.table;
        result->max_chunk = source.max_chunk;
        result->ids.resize(source.ids.size());
        return result.release();
    } catch (...) { return nullptr; }
}

extern "C" int64_t deepseek_v41_lookup_length(const void * handle) {
    return handle ? deepseek_v41_hash_length(static_cast<const lookup *>(handle)->hash.get()) : -1;
}

extern "C" int deepseek_v41_lookup_rewind(void * handle, int64_t position) {
    return handle ? deepseek_v41_hash_rewind(static_cast<lookup *>(handle)->hash.get(), position) : -1;
}

extern "C" int deepseek_v41_lookup_forward(void * handle, const int64_t * tokens,
        const uint8_t * mask, int64_t count, int64_t start,
        uint16_t * output, uint64_t output_elements,
        uint64_t * ids_output, uint64_t ids_elements) {
    if (!handle || count < 0) return -1;
    auto & current = *static_cast<lookup *>(handle);
    if (count > current.max_chunk || (count && !output) ||
        output_elements < uint64_t(count) * columns * width ||
        (ids_output && ids_elements < uint64_t(count) * columns) ||
        (!ids_output && ids_elements)) return -1;
    // All table spans, modes, capacities and token/history inputs are validated
    // before history changes. Hash output is guaranteed inside the covered tables.
    const int status = deepseek_v41_hash_forward(current.hash.get(), tokens, mask,
        count, start, current.ids.data(), current.ids.size());
    if (status) return status;
    for (int64_t token = 0; token < count; ++token) {
        for (int layer = 0; layer < 2; ++layer) {
            const uint64_t base = uint64_t(token) * columns + uint64_t(layer) * 24;
            const auto & parts = current.table->layers[layer];
            size_t part = 0;
            int column = 0;
            // Disjoint hash buckets make IDs monotonic within each 24-column layer.
            // Runs landing in one shard remain contiguous in the output tensor.
            while (column < 24) {
                const uint64_t id = current.ids[base + column];
                while (id >= parts[part].first_row + parts[part].rows) ++part;
                const auto & shard = parts[part];
                std::array<int64_t, 24> local{};
                int n = 0;
                while (column + n < 24 && current.ids[base + column + n] < shard.first_row + shard.rows) {
                    local[n] = int64_t(current.ids[base + column + n] - shard.first_row);
                    ++n;
                }
                // No fallible allocation or input validation remains after the hash commit.
                // A failure here is an internal contract violation, not a recoverable input error.
                if (deepseek_v41_engram_gather_bf16(shard.weight, shard.scale, int64_t(shard.rows),
                        local.data(), n, output + (base + column) * width, current.table->vectorize)) {
                    std::terminate();
                }
                column += n;
            }
        }
    }
    if (count && ids_output) std::memcpy(ids_output, current.ids.data(), uint64_t(count) * columns * sizeof(uint64_t));
    return 0;
}
