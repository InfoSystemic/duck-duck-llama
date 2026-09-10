// Native token-history and Engram row selection for the pinned V4.1 layout.
// Algorithm reference: DeepSeek-V4.1-Flash inference/engram.py (MIT), revision
// fb2764a5cf321eaa5070ca8f9e892818f477c16d. See the accompanying source notice.
// This component does not load weights or implement the complete model graph.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {
constexpr int layers = 2;
constexpr int ngram = 4;
constexpr int heads = 8;
constexpr int columns = (ngram - 1) * heads;
constexpr int output_columns = layers * columns;
constexpr int32_t dead = -1;

struct divisor {
    uint64_t modulus;
    uint64_t reciprocal;
    uint64_t offset;
};

struct configuration {
    std::vector<uint32_t> token_map;
    std::array<divisor, output_columns> divisors;
    std::array<uint64_t, layers * ngram> multipliers;
    int32_t pad;
    bool reciprocal_mode;
};

struct state {
    std::shared_ptr<const configuration> config;
    std::vector<int32_t> history;
    int64_t valid = 0;
};

// floor(x/d) is either mulhi(x,floor(2^64/d)) or one greater, so one
// conditional subtraction gives the exact remainder for every uint64_t x.
uint64_t remainder(uint64_t value, const divisor & d, bool optimized) {
    if (!optimized) {
        return value % d.modulus;
    }
    const uint64_t quotient = static_cast<uint64_t>((static_cast<__uint128_t>(value) * d.reciprocal) >> 64);
    const uint64_t residual = value - quotient * d.modulus;
    return residual >= d.modulus ? residual - d.modulus : residual;
}
} // namespace

extern "C" {

// Array shapes: token_map[vocabulary], moduli[2][3][8], multipliers[2][4],
// table_rows[2]. The converter supplies the tokenizer-derived map and the
// publisher's seeded multipliers. The runtime needs no tokenizer or RNG library.
void * deepseek_v41_hash_create(
        const uint32_t * token_map, int64_t vocabulary, int64_t compressed_vocabulary,
        int64_t pad_token, const uint64_t * moduli, const uint64_t * multipliers,
        const uint64_t * table_rows, int64_t max_sequence, int reciprocal_mode) {
    if (!token_map || !moduli || !multipliers || !table_rows || vocabulary <= 0 ||
        compressed_vocabulary < 2 || compressed_vocabulary > std::numeric_limits<int32_t>::max() ||
        pad_token < 0 || pad_token >= vocabulary || max_sequence <= 0 ||
        static_cast<uint64_t>(max_sequence) > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        static_cast<uint64_t>(vocabulary) > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
        (reciprocal_mode != 0 && reciprocal_mode != 1)) {
        return nullptr;
    }
    try {
        auto cfg = std::make_shared<configuration>();
        for (int64_t i = 0; i < vocabulary; ++i) {
            if (token_map[i] >= static_cast<uint64_t>(compressed_vocabulary)) {
                return nullptr;
            }
        }
        const uint64_t multiplier_limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                                           static_cast<uint64_t>(compressed_vocabulary - 1);
        for (int i = 0; i < layers * ngram; ++i) {
            if (!(multipliers[i] & 1) || multipliers[i] > multiplier_limit) {
                return nullptr;
            }
            cfg->multipliers[i] = multipliers[i];
        }
        for (int layer = 0; layer < layers; ++layer) {
            uint64_t offset = 0;
            for (int column = 0; column < columns; ++column) {
                const int index = layer * columns + column;
                const uint64_t modulus = moduli[index];
                if (modulus < 2 || modulus > std::numeric_limits<uint32_t>::max() ||
                    offset > std::numeric_limits<uint64_t>::max() - modulus) {
                    return nullptr;
                }
                cfg->divisors[index] = divisor{modulus,
                    static_cast<uint64_t>((static_cast<__uint128_t>(1) << 64) / modulus), offset};
                offset += modulus;
            }
            if (offset != table_rows[layer]) {
                return nullptr;
            }
        }
        cfg->token_map.assign(token_map, token_map + vocabulary);
        cfg->pad = static_cast<int32_t>(token_map[pad_token]);
        cfg->reciprocal_mode = reciprocal_mode == 1;
        auto result = std::make_unique<state>();
        result->config = std::move(cfg);
        result->history.resize(static_cast<size_t>(max_sequence));
        return result.release();
    } catch (...) {
        return nullptr;
    }
}

void deepseek_v41_hash_destroy(void * handle) {
    delete static_cast<state *>(handle);
}

int64_t deepseek_v41_hash_length(const void * handle) {
    return handle ? static_cast<const state *>(handle)->valid : -1;
}

// Clone shares immutable model metadata but copies token history. Independent
// handles may be used by separate requests; one handle is not concurrently mutable.
void * deepseek_v41_hash_clone(const void * handle) {
    if (!handle) {
        return nullptr;
    }
    try {
        return new state(*static_cast<const state *>(handle));
    } catch (...) {
        return nullptr;
    }
}

int deepseek_v41_hash_rewind(void * handle, int64_t position) {
    if (!handle || position < 0 || position > static_cast<state *>(handle)->valid) {
        return -1;
    }
    static_cast<state *>(handle)->valid = position;
    return 0;
}

// Returns [tokens][2 Engram layers][24 hash columns]. start_position may append
// or replace a populated suffix; gaps are rejected. Replacing a suffix truncates
// its logical length, so stale tokens cannot leak into a later request or branch.
// Validate all arguments and token IDs before changing history or output.
int deepseek_v41_hash_forward(void * handle, const int64_t * input_ids,
        const uint8_t * token_mask, int64_t count, int64_t start_position,
        uint64_t * output, uint64_t output_elements) {
    if (!handle || count < 0 || start_position < 0) {
        return -1;
    }
    auto & current = *static_cast<state *>(handle);
    const auto & cfg = *current.config;
    if (start_position > current.valid ||
        static_cast<uint64_t>(count) > current.history.size() - static_cast<size_t>(start_position) ||
        static_cast<uint64_t>(count) > std::numeric_limits<uint64_t>::max() / output_columns ||
        output_elements < static_cast<uint64_t>(count) * output_columns ||
        (count && (!input_ids || !output))) {
        return -1;
    }
    if (!count) {
        return 0;
    }
    for (int64_t i = 0; i < count; ++i) {
        if (input_ids[i] < 0 || static_cast<uint64_t>(input_ids[i]) >= cfg.token_map.size() ||
            (token_mask && token_mask[i] > 1)) {
            return -1;
        }
    }
    for (int64_t i = 0; i < count; ++i) {
        current.history[start_position + i] = token_mask && !token_mask[i] ? dead :
                                             static_cast<int32_t>(cfg.token_map[input_ids[i]]);
    }
    for (int64_t i = 0; i < count; ++i) {
        const int64_t position = start_position + i;
        std::array<uint64_t, ngram> tokens;
        bool blocked = false;
        for (int shift = 0; shift < ngram; ++shift) {
            const int32_t source = position >= shift ? current.history[position - shift] : dead;
            blocked = blocked || source == dead;
            tokens[shift] = static_cast<uint64_t>(blocked ? cfg.pad : source);
        }
        for (int layer = 0; layer < layers; ++layer) {
            uint64_t rolling = tokens[0] * cfg.multipliers[layer * ngram];
            for (int shift = 1; shift < ngram; ++shift) {
                rolling ^= tokens[shift] * cfg.multipliers[layer * ngram + shift];
                for (int head = 0; head < heads; ++head) {
                    const int column = layer * columns + (shift - 1) * heads + head;
                    const auto & d = cfg.divisors[column];
                    output[static_cast<uint64_t>(i) * output_columns + column] =
                        remainder(rolling, d, cfg.reciprocal_mode) + d.offset;
                }
            }
        }
    }
    current.valid = start_position + count;
    return 0;
}

// Exposed only for independent exhaustive/boundary arithmetic validation.
int deepseek_v41_hash_remainder(uint64_t value, uint64_t modulus, uint64_t * output) {
    if (!output || modulus < 2 || modulus > std::numeric_limits<uint32_t>::max()) {
        return -1;
    }
    const divisor d{modulus, static_cast<uint64_t>((static_cast<__uint128_t>(1) << 64) / modulus), 0};
    *output = remainder(value, d, true);
    return 0;
}

} // extern "C"

#ifdef DEEPSEEK_V41_HASH_BENCH
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    uint32_t header[3];
    uint64_t moduli[output_columns], multipliers[layers * ngram], rows[layers];
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    file.read(reinterpret_cast<char *>(moduli), sizeof(moduli));
    file.read(reinterpret_cast<char *>(multipliers), sizeof(multipliers));
    file.read(reinterpret_cast<char *>(rows), sizeof(rows));
    if (!file || header[0] != 129280 || header[1] != 99092) return 2;
    std::vector<uint32_t> map(header[0]);
    file.read(reinterpret_cast<char *>(map.data()), map.size() * sizeof(uint32_t));
    if (!file || file.peek() != std::char_traits<char>::eof()) return 2;
    const int mode = std::atoi(argv[2]);
    const int count = std::atoi(argv[3]);
    const int repeats = std::atoi(argv[4]);
    if (count <= 0 || count > 8192 || repeats < 1 || repeats > 100000) return 2;
    void * handle = deepseek_v41_hash_create(map.data(), header[0], header[1], header[2],
        moduli, multipliers, rows, count, mode);
    if (!handle) return 2;
    std::vector<int64_t> input(count);
    std::vector<uint8_t> mask(count, 1);
    std::vector<uint64_t> output(count * output_columns);
    for (int i = 0; i < count; ++i) {
        input[i] = (static_cast<uint64_t>(i) * 97531 + 7) % header[0];
        mask[i] = i % 31 != 17;
    }
    for (int i = 0; i < 100; ++i) {
        if (deepseek_v41_hash_forward(handle, input.data(), mask.data(), count, 0, output.data(), output.size())) return 2;
    }
    std::vector<double> timings;
    uint64_t guard = 0;
    for (int sample = 0; sample < 21; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            input[0] = (static_cast<uint64_t>(repeat + sample * repeats) * 131 + 7) % header[0];
            if (deepseek_v41_hash_forward(handle, input.data(), mask.data(), count, 0, output.data(), output.size())) return 2;
            guard += output[repeat % output.size()];
        }
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        timings.push_back(ns / repeats);
    }
    std::sort(timings.begin(), timings.end());
    uint64_t digest = 1469598103934665603ULL;
    for (uint64_t value : output) { digest ^= value; digest *= 1099511628211ULL; }
    std::printf("{\"mode\":%d,\"tokens\":%d,\"repeats\":%d,\"samples\":21,\"median_ns_per_call\":%.9f,\"median_ns_per_token\":%.9f,\"output_digest\":\"%016llx\",\"guard\":\"%016llx\"}\n",
        mode, count, repeats, timings[10], timings[10] / count,
        static_cast<unsigned long long>(digest), static_cast<unsigned long long>(guard));
    deepseek_v41_hash_destroy(handle);
    return 0;
}
#endif
