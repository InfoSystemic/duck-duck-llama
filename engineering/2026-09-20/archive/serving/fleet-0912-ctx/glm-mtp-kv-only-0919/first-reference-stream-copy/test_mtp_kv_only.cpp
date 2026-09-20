#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct recorder {
    FILE * out;
    uint64_t records = 0;
    uint64_t bytes = 0;
    explicit recorder(const char * path) : out(std::fopen(path, "wb")) { check(out, "open output"); }
    ~recorder() { if (out) std::fclose(out); }
    void add(const std::string & label, const void * data, size_t size) {
        const uint64_t sizes[2] = {label.size(), size};
        check(std::fwrite(sizes, sizeof(sizes), 1, out) == 1, "write lengths");
        check(std::fwrite(label.data(), 1, label.size(), out) == label.size(), "write label");
        check(std::fwrite(data, 1, size, out) == size, "write payload");
        ++records; bytes += sizeof(sizes) + label.size() + size;
    }
};

static std::vector<uint8_t> state(llama_context * ctx, llama_seq_id seq) {
    const size_t size = llama_state_seq_get_size(ctx, seq);
    std::vector<uint8_t> data(size);
    check(llama_state_seq_get_data(ctx, data.data(), size, seq) == size, "get sequence state");
    return data;
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    try {
        const int devices_n = std::atoi(argv[3]);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.load_mtp = true;
        mp.use_extra_bufts = true;
        std::vector<ggml_backend_dev_t> devices;
        if (devices_n > 0) {
            for (int i = 0; i < devices_n; ++i) {
                auto * dev = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(i)).c_str());
                check(dev != nullptr, "missing NUMA device");
                devices.push_back(dev);
            }
            devices.push_back(nullptr);
            mp.devices = devices.data();
            mp.n_gpu_layers = 999;
            mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        } else {
            mp.n_gpu_layers = 0;
        }
        auto * model = llama_model_load_from_file(argv[1], mp);
        check(model != nullptr, "load MTP model");
        const int width = llama_model_n_embd_out(model);
        const int vocab_n = llama_vocab_n_tokens(llama_model_get_vocab(model));
        recorder output(argv[2]);
        uint64_t calls = 0, output_rows = 0, zero_output_calls = 0;
        double zero_ms = 0, nonzero_ms = 0;
        for (bool unified : {true, false}) {
            auto cp = llama_context_default_params();
            cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cp.n_ctx = 1024;
            cp.n_batch = 128;
            cp.n_ubatch = 128;
            cp.n_seq_max = 2;
            cp.n_threads = 1;
            cp.n_threads_batch = 1;
            cp.kv_unified = unified;
            cp.embeddings = false;
            cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            auto * ctx = llama_init_from_model(model, cp);
            check(ctx != nullptr, "create MTP context");
            llama_set_embeddings_nextn(ctx, true, true);
            auto batch = llama_batch_init(128, width, 1);
            batch.token = static_cast<llama_token *>(std::malloc(128 * sizeof(llama_token)));
            check(batch.token != nullptr, "allocate tokens");
            auto * memory = llama_get_memory(ctx);
            int serial = 0;
            const std::string prefix = unified ? "unified/" : "separate/";
            auto save = [&](const std::string & name) {
                for (int seq = 0; seq < 2; ++seq) {
                    const auto data = state(ctx, seq);
                    output.add(prefix + name + "/seq" + std::to_string(seq), data.data(), data.size());
                }
            };
            auto decode = [&](const std::string & name, int seq, int pos, int count, int outputs, bool two_sequences = false) {
                check(count <= 128 && outputs <= count, "batch capacity");
                batch.n_tokens = count;
                for (int j = 0; j < count; ++j) {
                    const int s = two_sequences ? (j < count / 2 ? 0 : 1) : seq;
                    const int p = pos + (two_sequences ? j % (count / 2) : j);
                    batch.token[j] = 100 + ((p + 17 * serial + 11 * s) % 3000);
                    batch.pos[j] = p;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = s;
                    batch.logits[j] = outputs == count || (outputs > 0 && j >= count - outputs);
                    for (int k = 0; k < width; ++k) {
                        batch.embd[j * width + k] = std::sin(float(k + 17 * p + 29 * serial + 7 * s) * 0.03f);
                    }
                }
                const auto start = std::chrono::steady_clock::now();
                check(llama_decode(ctx, batch) == 0, "MTP decode");
                llama_synchronize(ctx);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                ++calls;
                if (outputs == 0) { ++zero_output_calls; zero_ms += ms; }
                else nonzero_ms += ms;
                for (int j = 0; j < count; ++j) {
                    if (!batch.logits[j]) continue;
                    const float * logits = llama_get_logits_ith(ctx, j);
                    const float * hidden = llama_get_embeddings_nextn_ith(ctx, j);
                    check(logits && hidden, "missing output");
                    for (int k = 0; k < vocab_n; ++k) check(std::isfinite(logits[k]), "non-finite logits");
                    for (int k = 0; k < width; ++k) check(std::isfinite(hidden[k]), "non-finite hidden state");
                    output.add(prefix + name + "/logits" + std::to_string(j), logits, vocab_n * sizeof(float));
                    output.add(prefix + name + "/hidden" + std::to_string(j), hidden, width * sizeof(float));
                    ++output_rows;
                }
                save(name);
                ++serial;
            };
            save("empty");
            decode("prefill32", 0, 0, 32, 0);
            decode("prefill32-repeated-shape", 0, 32, 32, 0);
            decode("prefill64", 0, 64, 64, 0);
            decode("first-prediction", 0, 128, 1, 1);
            decode("second-prediction", 0, 129, 1, 1);
            check(llama_memory_seq_rm(memory, 0, 120, -1), "rollback before catch-up");
            save("rollback");
            decode("catchup-three", 0, 120, 3, 0);
            decode("prediction-after-catchup", 0, 123, 1, 1);
            decode("mixed-three", 0, 124, 3, 1);
            decode("all-output-three", 0, 127, 3, 3);
            const auto checkpoint = state(ctx, 0);
            decode("catchup-one", 0, 130, 1, 0);
            decode("catchup-one-repeated-shape", 0, 131, 1, 0);
            decode("prediction-after-one", 0, 132, 1, 1);
            check(llama_state_seq_set_data(ctx, checkpoint.data(), checkpoint.size(), 0) == checkpoint.size(), "restore checkpoint");
            save("restore");
            decode("catchup-after-restore", 0, 130, 8, 0);
            decode("prediction-after-restore", 0, 138, 2, 2);
            llama_memory_seq_cp(memory, 0, 1, 0, -1);
            save("copy-sequence");
            decode("second-sequence-catchup", 1, 140, 3, 0);
            decode("second-sequence-prediction", 1, 143, 1, 1);
            decode("first-sequence-catchup", 0, 140, 3, 0);
            decode("first-sequence-prediction", 0, 143, 1, 1);
            decode("two-sequence-catchup", 0, 144, 6, 0, true);
            decode("two-sequence-prediction", 0, 147, 2, 2, true);
            check(llama_memory_seq_rm(memory, 1, 0, -1), "remove second sequence");
            save("remove-second-sequence");
            check(llama_memory_seq_rm(memory, 0, 140, -1), "remove first suffix");
            decode("rewritten-catchup", 0, 140, 4, 0);
            decode("rewritten-prediction", 0, 144, 1, 1);
            llama_batch_free(batch);
            llama_free(ctx);
        }
        std::printf("{\"passed\":true,\"devices\":%d,\"contexts\":2,\"calls\":%llu,\"zero_output_calls\":%llu,\"output_rows\":%llu,\"records\":%llu,\"bytes\":%llu,\"zero_output_ms\":%.6f,\"other_decode_ms\":%.6f}\n", devices_n, (unsigned long long) calls, (unsigned long long) zero_output_calls, (unsigned long long) output_rows, (unsigned long long) output.records, (unsigned long long) output.bytes, zero_ms, nonzero_ms);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MTP cache-only test failed: %s\n", error.what());
        return 1;
    }
}
