#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-io.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

struct writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * src, size_t n) override {
        const auto * p = static_cast<const uint8_t *>(src);
        bytes.insert(bytes.end(), p, p+n);
    }
    void write_tensor(ggml_tensor *, size_t, size_t) override { throw std::runtime_error("unexpected tensor data"); }
    size_t n_bytes() override { return bytes.size(); }
};

struct reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t offset = 0;
    explicit reader(const std::vector<uint8_t> & b) : bytes(b) {}
    void read(void * dst, size_t n) override {
        if (n > bytes.size()-offset) throw std::runtime_error("state read overflow");
        std::memcpy(dst, bytes.data()+offset, n);
        offset += n;
    }
    void read_tensor(ggml_tensor *, size_t, size_t) override { throw std::runtime_error("unexpected tensor data"); }
    size_t n_bytes() override { return offset; }
};

struct one_token {
    llama_pos pos;
    int32_t count;
    llama_seq_id ids[2];
    llama_seq_id * idp = ids;
    int32_t indices[LLAMA_MAX_SEQ];
    llama_ubatch batch{};
    one_token(llama_pos p, int sid, int second = -1) : pos(p), count(second < 0 ? 1 : 2), ids{sid, second} {
        std::fill(std::begin(indices), std::end(indices), -1);
        indices[sid] = 0;
        if (second >= 0) indices[second] = 1;
        batch.b_equal_seqs = 1;
        batch.n_tokens = batch.n_seq_tokens = batch.n_seqs = batch.n_pos = 1;
        batch.n_seqs_unq = count;
        batch.pos = &pos;
        batch.n_seq_id = &count;
        batch.seq_id = &idp;
        batch.seq_id_unq = ids;
        batch.seq_idx = indices;
    }
};

static std::unique_ptr<llama_kv_cache> make_cache(llama_model & m, uint32_t size, bool unified, llama_kv_cache * other = nullptr) {
    return std::make_unique<llama_kv_cache>(m, m.hparams, GGML_TYPE_F16, GGML_TYPE_F16,
        false, false, unified, size, 4, 1, 0, LLAMA_SWA_TYPE_NONE, other, nullptr, nullptr, nullptr);
}

static void insert(llama_kv_cache & cache, bool unified, uint32_t idx, llama_pos pos, int sid, int second = -1) {
    one_token one(pos, sid, second);
    llama_kv_cache::slot_info slot;
    const uint32_t stream = unified ? 0 : sid;
    slot.s0 = stream;
    slot.s1 = stream+1;
    slot.strm = { static_cast<llama_seq_id>(stream) };
    slot.idxs = { { idx } };
    cache.apply_ubatch(slot, one.batch);
}

static void record(std::ofstream & file, llama_kv_cache & cache, uint32_t case_id, uint32_t step) {
    writer state;
    state.write(&case_id, sizeof(case_id));
    state.write(&step, sizeof(step));
    cache.state_write(state);
    for (int sid = 0; sid < 4; ++sid) {
        const auto & cells = cache.get_cells(sid);
        uint32_t meta[] = { cells.size(), cells.get_used(), cells.used_min(), cells.used_max_p1(), cells.get_has_shift() };
        state.write(meta, sizeof(meta));
        llama_pos limits[] = { cells.seq_pos_min(sid), cells.seq_pos_max(sid) };
        state.write(limits, sizeof(limits));
        one_token one(std::max<llama_pos>(0, limits[1]+1), sid);
        auto next = cache.find_slot(one.batch, true);
        uint32_t count = next.empty() ? 0 : next.size();
        state.write(&count, sizeof(count));
        if (count) {
            for (const auto & indices : next.idxs) state.write(indices.data(), indices.size()*sizeof(indices[0]));
        }
        // Serialized state omits physical indices and accumulated shifts.
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) continue;
            llama_pos values[] = { static_cast<llama_pos>(i), cells.pos_get(i), cells.get_shift(i) };
            state.write(values, sizeof(values));
            auto ext = cells.ext_get(i);
            state.write(&ext, sizeof(ext));
            uint32_t members = 0;
            for (int s = 0; s < 4; ++s) members |= uint32_t(cells.seq_has(i, s)) << s;
            state.write(&members, sizeof(members));
        }
    }
    uint64_t size = state.bytes.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    file.write(reinterpret_cast<const char *>(state.bytes.data()), state.bytes.size());
    if (!file) throw std::runtime_error("state output failed");
}

static void parity(llama_model & model, const char * output) {
    std::ofstream file(output, std::ios::binary);
    std::mt19937 rng(0x194253);
    uint32_t case_id = 0, operations = 0, records = 0;
    for (uint32_t size : {1u, 32u, 128u, 1024u, 4096u, 32768u, 100000u, 1048576u}) {
        for (bool unified : {true, false}) {
            auto cache = make_cache(model, size, unified);
            auto shared = make_cache(model, size, unified, cache.get());
            ++case_id;
            auto save = [&] {
                record(file, *cache, case_id, operations);
                ++records;
                if (const char * control = std::getenv("TEST_KV_MODE_CONTROL")) {
                    std::fstream word(control, std::ios::in | std::ios::out | std::ios::binary);
                    uint32_t value = records%2;
                    word.write(reinterpret_cast<const char *>(&value), sizeof(value));
                    if (!word) throw std::runtime_error("control write failed");
                }
            };
            save();
            for (int sid = 0; sid < 4; ++sid) {
                insert(*cache, unified, size-1, 700+sid, sid);
                ++operations;
                save();
                shared->seq_rm(-1, -1, -1);
                ++operations;
                save();
                cache->seq_rm(sid, 700+sid, -1);
                ++operations;
                save();
            }
            cache->clear(false);
            for (uint32_t i = 0; i < std::min(size, 256u); ++i) {
                const int sid = unified ? 0 : i%4;
                insert(*cache, unified, i, i, sid, unified && i%3 == 0 ? 1 : -1);
                ++operations;
            }
            save();
            for (int step = 0; step < 160; ++step) {
                int sid = rng()%4;
                llama_pos p0 = int(rng()%400)-40;
                llama_pos p1 = (rng()%3 == 0) ? -1 : p0+int(rng()%80);
                switch (rng()%9) {
                    case 0: cache->seq_rm(sid, p0, p1); break;
                    case 1: cache->seq_rm(-1, p0, p1); break;
                    case 2: cache->seq_rm(sid, -1, 0); break;
                    case 3: cache->seq_cp(sid, (sid+1)%4, unified ? p0 : 0, unified ? p1 : -1); break;
                    case 4: cache->seq_add(sid, p0, p1, int(rng()%31)-15); break;
                    case 5: cache->seq_div(sid, p0, p1, 2); break;
                    case 6: {
                        auto & cells = cache->get_cells(sid);
                        uint32_t idx = rng()%size;
                        // Avoid overwriting a shared cell, which apply_ubatch rejects.
                        if (!cells.is_empty(idx) && cells.seq_count(idx) > 1) break;
                        insert(*cache, unified, idx, rng()%400, sid);
                        break;
                    }
                    case 7: cache->seq_keep(sid); break;
                    case 8: {
                        writer saved;
                        cache->state_write(saved);
                        reader restore(saved.bytes);
                        cache->clear(false);
                        cache->state_read(restore);
                        if (restore.n_bytes() != saved.bytes.size()) throw std::runtime_error("incomplete restore");
                        break;
                    }
                }
                ++operations;
                save();
            }
            cache->seq_rm(-1, -1, -1);
            ++operations;
            save();
        }
    }
    std::cout << "{\"cases\":" << case_id << ",\"operations\":" << operations << ",\"records\":" << records << "}\n";
}

static void benchmark(llama_model & model) {
    std::cout << "[";
    bool first = true;
    for (uint32_t capacity : {32768u, 1048576u}) {
        for (uint32_t used : {0u, 4096u, 32768u, 100000u}) {
            if (used > capacity) continue;
            auto cache = make_cache(model, capacity, true);
            auto & cells = const_cast<llama_kv_cells &>(cache->get_cells(0));
            for (uint32_t i = 0; i < used; ++i) { cells.pos_set(i, i); cells.seq_add(i, 0); }
            for (int sid : {0, -1}) {
                std::vector<double> samples;
                constexpr int repeats = 7, iterations = 120;
                for (int repeat = 0; repeat < repeats+1; ++repeat) {
                    auto begin = std::chrono::steady_clock::now();
                    for (int i = 0; i < iterations; ++i) cache->seq_rm(sid, used+3, -1);
                    auto end = std::chrono::steady_clock::now();
                    if (repeat) samples.push_back(std::chrono::duration<double, std::micro>(end-begin).count()/iterations);
                }
                std::sort(samples.begin(), samples.end());
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"capacity\":" << capacity << ",\"used\":" << used << ",\"seq\":" << sid << ",\"median_us\":" << samples[samples.size()/2] << "}";
            }
        }
    }
    std::cout << "]\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        llama_log_set([](ggml_log_level, const char * text, void *) { if (std::strstr(text, "KV_SEQ_RM_MODE")) std::cerr << text; }, nullptr);
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
        if (!model || model->hparams.n_layer_all) throw std::runtime_error("unexpected model state");
        if (std::string(argv[1]) == "parity") parity(*model, argv[2]);
        else if (std::string(argv[1]) == "bench") benchmark(*model);
        else return 2;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
