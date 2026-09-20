#!/usr/bin/env python3
"""make_f18.py -- derive speculative.f18.cpp from the production speculative.cpp (string-anchored, each anchor must match once).

Changes, all inside common_speculative_impl_draft_mtp and all default OFF:
  GGML_F18_MTP_MERGE=1     small verify batches are not caught up with their own draft decode; the accepted rows ride in the
                           first draft pass of the next cycle (one graph less per cycle).
  GGML_F18_MTP_FAST_PICK=1 the draft token is picked from the logits directly (top-10 softmax, same value the sampler chain gives)
                           instead of building a full-vocabulary candidate array twice per cycle.
  GGML_F18_SPEC_TRACE=1    per-phase wall time of the MTP driver, one WARN line every 256 draft() calls.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
s = (HERE/'speculative.orig.cpp').read_text()
s = s.replace('#include <cassert>\n', '#include <cassert>\n#include <cmath>\n#include <cstdlib>\n#include <fcntl.h>\n#include <sys/mman.h>\n#include <unistd.h>\n', 1)

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)

# ---- members
rep('''    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)''',
'''    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // f18: rows of a small target batch whose draft-cache catch-up is deferred to the next draft pass
    struct deferred_rows {
        std::vector<llama_token> tok;
        std::vector<llama_pos>   pos;
        std::vector<float>       h;   // [tok.size()][n_embd]
    };
    std::vector<deferred_rows> deferred;

    bool f18_merge     = false;
    bool f18_fast_pick = false;
    bool f18_trace     = false;

    int64_t f18_t_process = 0, f18_t_decode1 = 0, f18_t_pick1 = 0, f18_t_decode2 = 0, f18_t_pick2 = 0;
    int64_t f18_n_draft = 0, f18_n_merged = 0, f18_n_flushed = 0;

    static bool f18_env(const char * name) {
        const char * v = getenv(name);
        return v && atoi(v) != 0;
    }

    // optional 4-byte control file: bit 0 = merge, bit 1 = fast pick. Change it only while the server is idle.
    const uint32_t * f18_control = nullptr;
    bool f18_merge_allowed = false;

    void f18_refresh() {
        if (f18_control) {
            const uint32_t mask = __atomic_load_n(f18_control, __ATOMIC_ACQUIRE);
            f18_merge     = f18_merge_allowed && (mask & 1u);
            f18_fast_pick = (mask & 2u) != 0;
        }
    }

    // same result as the TOP_K(10)+dist chain read through get_candidates: the best token and its softmax share of the top 10
    static llama_token f18_pick(const float * logits, int n_vocab, float & p_top) {
        const int K = 10;
        float best[K];
        int   best_id[K];
        int   n = 0;
        float thr = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) {
            const float l = logits[i];
            if (n == K && l <= thr) {
                continue;
            }
            int j = n < K ? n++ : K - 1;
            while (j > 0 && best[j - 1] < l) {
                best[j] = best[j - 1];
                best_id[j] = best_id[j - 1];
                --j;
            }
            best[j] = l;
            best_id[j] = i;
            if (n == K) {
                thr = best[K - 1];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += exp((double) (best[j] - best[0]));
        }
        p_top = (float) (1.0/sum);
        return best_id[0];
    }

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)''')

rep('''        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {''',
'''        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        deferred.assign(n_seq, {});
        f18_merge_allowed = !is_mem_shared && !chain_heads;
        f18_merge     = f18_env("GGML_F18_MTP_MERGE") && f18_merge_allowed;
        f18_fast_pick = f18_env("GGML_F18_MTP_FAST_PICK");
        f18_trace     = f18_env("GGML_F18_SPEC_TRACE");
        if (const char * path = getenv("GGML_F18_SPEC_CONTROL_FILE")) {
            const int fd = open(path, O_RDONLY | O_CLOEXEC);
            GGML_ASSERT(fd >= 0 && "GGML_F18_SPEC_CONTROL_FILE must exist");
            void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            GGML_ASSERT(mapping != MAP_FAILED);
            f18_control = (const uint32_t *) mapping;
            f18_refresh();
        }
        SPC_WRN("f18: mtp merge=%d fast_pick=%d trace=%d\\n", (int) f18_merge, (int) f18_fast_pick, (int) f18_trace);
    }

    // f18: write the deferred rows of seq_id that are still part of the sequence (pos < pos_end) to the draft cache now
    bool f18_flush(llama_seq_id seq_id, llama_pos pos_end) {
        auto & d = deferred[seq_id];
        if (d.tok.empty()) {
            return true;
        }
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        common_batch_clear(batch);
        for (size_t i = 0; i < d.tok.size(); ++i) {
            if (d.pos[i] >= pos_end || d.pos[i] <= pos_max) {
                continue;
            }
            if (d.pos[i] != pos_max + 1 + batch.n_tokens) {
                break; // not contiguous with the cache: leave the hole to the normal recovery
            }
            common_batch_add(batch, d.tok[i], d.pos[i], { seq_id }, 0);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, d.h.data() + i * (size_t) n_embd, (size_t) n_embd * sizeof(float));
        }
        d.tok.clear(); d.pos.clear(); d.h.clear();

        if (batch.n_tokens == 0) {
            return true;
        }
        f18_n_flushed++;
        const int32_t rc = llama_decode(ctx_dft, batch);
        common_batch_clear(batch);
        if (rc != 0) {
            SPC_ERR("f18 flush llama_decode(ctx_dft) failed rc=%d\\n", (int) rc);
            return false;
        }
        return true;
    }

    ~common_speculative_impl_draft_mtp() override {''')

# ---- process(): defer small batches
rep('''        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);
''',
'''        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        const int64_t f18_t0 = f18_trace ? ggml_time_us() : 0;

        // f18: rows deferred by an earlier call must reach the cache before this batch does
        f18_refresh();
        bool f18_defer = f18_merge;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] >= 0 && !f18_flush(seq_id, batch_in.pos[i_batch_beg[seq_id]])) {
                return false;
            }
        }
        if (f18_merge) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                // defer only short per-sequence runs (a verify batch); prompt chunks take the normal path
                if (i_batch_end[seq_id] - i_batch_beg[seq_id] + 1 > params.n_max + 1) {
                    f18_defer = false;
                }
            }
            for (int k = 0; k < n_tokens && f18_defer; ++k) {
                const llama_seq_id seq_id = batch_in.seq_id[k][0];
                if (k < i_batch_beg[seq_id] || k > i_batch_end[seq_id]) {
                    f18_defer = false; // interleaved sequences: the shifted-row layout below does not hold
                }
            }
        }

        if (f18_defer) {
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            for (int k = 0; k < n_tokens; ++k) {
                const llama_seq_id seq_id = batch_in.seq_id[k][0];
                auto & d = deferred[seq_id];
                d.tok.push_back(batch_in.token[k]);
                d.pos.push_back(batch_in.pos[k]);
                const float * h_row = k == i_batch_beg[seq_id] ? pending_h[seq_id].data() : h_tgt + (size_t) (k - 1) * n_embd;
                d.h.insert(d.h.end(), h_row, h_row + n_embd);
            }
        }

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared && !f18_defer) {
            common_batch_clear(batch);
''')

rep('''            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }
''',
'''            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        if (f18_trace) {
            f18_t_process += ggml_time_us() - f18_t0;
        }

        return true;
    }
''')

# ---- draft(): prepend the accepted deferred rows to the first pass
rep('''            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);
''',
'''            if (!deferred[seq_id].tok.empty()) {
                // rows at pos >= n_past were rejected; the rest continue the draft cache
                auto & d = deferred[seq_id];
                llama_pos pos_next = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) + 1;
                if (f18_trace && f18_n_draft < 48) {
                    SPC_WRN("F18_SPEC_DBG draft seq=%d n_past=%d dft_pos_next=%d deferred=%zu first=%d last=%d\\n", (int) seq_id, (int) dp.n_past,
                            (int) pos_next, d.tok.size(), (int) d.pos.front(), (int) d.pos.back());
                }
                for (size_t i = 0; i < d.tok.size(); ++i) {
                    if (d.pos[i] >= dp.n_past || d.pos[i] != pos_next) {
                        continue;
                    }
                    common_batch_add(batch, d.tok[i], d.pos[i], { seq_id }, false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, d.h.data() + i * (size_t) n_embd, row_bytes);
                    pos_next++;
                    f18_n_merged++;
                }
                d.tok.clear(); d.pos.clear(); d.h.clear();
            }

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);
''')

# ---- draft(): timing + fast pick
rep('''            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the''',
'''            const int64_t f18_td = f18_trace ? ggml_time_us() : 0;
            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\\n", i, ret);
                break;
            }
            const int64_t f18_tp = f18_trace ? ggml_time_us() : 0;
            if (f18_trace) {
                (i == 0 ? f18_t_decode1 : f18_t_decode2) += f18_tp - f18_td;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the''')

rep('''                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);''',
'''                auto * smpl = smpls[seq_id].get();

                llama_token id;
                float       p_top;
                const float * h_row;

                if (f18_fast_pick && backend_chains[seq_id] == nullptr) {
                    llama_synchronize(ctx_dft);
                    const float * logits = llama_get_logits_ith(ctx_dft, i_last[seq_id]);
                    GGML_ASSERT(logits != nullptr);
                    id    = f18_pick(logits, llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft))), p_top);
                    h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);
                } else {
                    common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                    h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    // add drafted token for each sequence
                    id    = cur_p->data[0].id;
                    p_top = cur_p->data[0].p;
                }

                // only collect very high-confidence draft tokens
                if (p_top < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                if (!(f18_fast_pick && backend_chains[seq_id] == nullptr)) {
                    common_sampler_accept(smpl, id, true);
                }

                auto & dp = dparams.at(seq_id);''')

rep('''            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];''',
'''            if (f18_trace) {
                (i == 0 ? f18_t_pick1 : f18_t_pick2) += ggml_time_us() - f18_tp;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (f18_trace && ++f18_n_draft % 256 == 0) {
            SPC_WRN("F18_SPEC_TRACE drafts=%lld per-cycle ms: process=%.3f decode1=%.3f pick1=%.3f decode2=%.3f pick2=%.3f merged_rows=%.2f flushes=%lld\\n",
                    (long long) f18_n_draft, f18_t_process/1000.0/f18_n_draft, f18_t_decode1/1000.0/f18_n_draft, f18_t_pick1/1000.0/f18_n_draft,
                    f18_t_decode2/1000.0/f18_n_draft, f18_t_pick2/1000.0/f18_n_draft, (double) f18_n_merged/f18_n_draft, (long long) f18_n_flushed);
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];''')

(HERE/'speculative.f18.cpp').write_text(s)
print('wrote speculative.f18.cpp')
