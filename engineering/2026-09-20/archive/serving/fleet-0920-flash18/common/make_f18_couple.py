#!/usr/bin/env python3
"""make_f18_couple.py -- stage 3: coupled draft/verifier sampling (GGML_F18_COUPLED=1, control bits 3 and 4).

With a sampled request (Codex sends no sampling fields, the GGUF says temp 1.0) the verifier draws a token at random and the
greedy MTP draft is accepted only when the draw happens to be the draft's argmax: 72% of drafts against 82% greedy.
Here the verifier's final pick becomes Gumbel-max with counter-based noise (exact sample from the same distribution, see
f18-coupling.inc), and the drafter adds the same noise to its own distribution, so both follow the same random draw.

  sampling.f18c.cpp     <- engine common/sampling.cpp : noise-keyed final pick, per-request salt, accept counter, lookup
  speculative.f18c.cpp  <- speculative.f18b.cpp       : MTP fast pick uses the verifier's salt and counter

Control word (GGML_F18_SPEC_CONTROL_FILE): bit 3 (8) verifier picks by Gumbel-max, bit 4 (16) drafter couples to it.
Without a control file GGML_F18_COUPLED=1 turns both on. Default off: stock behaviour, bit for bit.

Also here, because it is the same file: the host-side sampler cost (GGML_F18_FAST_SAMPLER=1, control bit 5 = 32).
Measured per verify cycle on this vocabulary (154,880): 0.45 ms to build the candidate array and 0.19 ms for the chain PER
SAMPLED TOKEN, plus 0.58 ms for the sampler clone the server takes every cycle (it copies the 1.86 MB candidate array).
When the first active sampler of the chain is top-k (the default chain), the k best logits are selected straight from the
logits with an AVX-512 threshold scan and the chain starts from those k candidates, already sorted: same candidates, same
order, same token. The clone stops copying candidates that nobody reads.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
ENG  = Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')

def patcher(text):
    box = [text]
    def rep(old, new, count=1):
        assert box[0].count(old) == count, (box[0].count(old), old[:80])
        box[0] = box[0].replace(old, new)
    return box, rep

# ---------------------------------------------------------------- sampling.cpp
box, rep = patcher((ENG/'common/sampling.cpp').read_text())

rep('''#include <unordered_map>
#include <vector>
''', '''#include <unordered_map>
#include <vector>

// ---- f18 coupled sampling (f18-coupling.inc) ----
#include "f18-coupling.inc"

#include <fcntl.h>
#include <mutex>
#include <random>
#include <sys/mman.h>
#include <unistd.h>

// GGML_F18_COUPLED=1, or bit 3 of the mapped control word when GGML_F18_SPEC_CONTROL_FILE is set (flip it only while idle)
static bool f18_coupled_enabled() {
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = getenv("GGML_F18_SPEC_CONTROL_FILE");
        if (!path || !*path) {
            return nullptr;
        }
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return nullptr;
        }
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        return mapping == MAP_FAILED ? nullptr : (const uint32_t *) mapping;
    }();
    static const bool env_on = [] {
        const char * v = getenv("GGML_F18_COUPLED");
        return v && atoi(v) != 0;
    }();
    if (control) {
        return (__atomic_load_n(control, __ATOMIC_ACQUIRE) & 8u) != 0;
    }
    return env_on;
}

// GGML_F18_FAST_SAMPLER=1, or bits of the control word: bit 5 (32) fast top-k candidates, bit 6 (64) clone without candidates
static bool f18_fast_sampler_enabled(uint32_t bit = 32u) {
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = getenv("GGML_F18_SPEC_CONTROL_FILE");
        if (!path || !*path) {
            return nullptr;
        }
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return nullptr;
        }
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        return mapping == MAP_FAILED ? nullptr : (const uint32_t *) mapping;
    }();
    static const bool env_on = [] {
        const char * v = getenv("GGML_F18_FAST_SAMPLER");
        return v && atoi(v) != 0;
    }();
    if (control) {
        return (__atomic_load_n(control, __ATOMIC_ACQUIRE) & bit) != 0;
    }
    return env_on;
}

#include "f18-topk-scan.inc"

// GGML_F18_SPEC_TRACE=1: where the host-side sampling time goes (main loop is single-threaded)
static const bool f18_smpl_trace = [] {
    const char * v = getenv("GGML_F18_SPEC_TRACE");
    return v && atoi(v) != 0;
}();
static int64_t f18_t_set_logits = 0, f18_t_chain = 0, f18_t_clone = 0, f18_n_sample = 0, f18_n_clone = 0;

// a fixed request seed gives a fixed noise field (same prompt + same seed = same text, with or without a drafter);
// the default seed gives a fresh one per request
static uint64_t f18_new_salt(uint32_t seed) {
    if (seed != LLAMA_DEFAULT_SEED) {
        return f18_mix64(0xF18C0DE5EEDull ^ (uint64_t) seed);
    }
    static std::mutex mtx;
    static std::mt19937_64 rng = [] {
        std::random_device rd;
        const uint64_t s = ((uint64_t) rd() << 32) ^ (uint64_t) rd() ^ (uint64_t) ggml_time_us();
        return std::mt19937_64(s);
    }();
    std::lock_guard<std::mutex> lock(mtx);
    return rng();
}
''')

rep('''    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);
''', '''    void set_logits(struct llama_context * ctx, int idx, bool f18_fast = false) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (f18_fast && f18_fast_k > 0 && f18_fast_k < n_vocab && !sampled_probs && !sampled_logits) {
            // the chain's first active sampler is top-k: hand it its own result (k best, sorted) instead of the vocabulary
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            float   best[F18_FAST_K_MAX];
            int32_t best_id[F18_FAST_K_MAX];
            const int m = f18_topk_scan(logits, n_vocab, f18_fast_k, best, best_id);
            if (m == f18_fast_k) {
                cur.resize(m);
                for (int i = 0; i < m; ++i) {
                    cur[i] = llama_token_data{best_id[i], best[i], 0.0f};
                }
                cur_p = { cur.data(), cur.size(), -1, true };
                return;
            }
        }
''')

rep('''    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }
''', '''    void reset() {
        prev.clear();

        llama_sampler_reset(chain);

        f18_n_acc = 0;
        f18_salt  = f18_new_salt(params.seed);
    }
''')

rep('''    mutable int64_t t_total_us = 0;
};
''', '''    mutable int64_t t_total_us = 0;

    // f18 coupled sampling: the final pick of a chain that ends in `dist` is argmax(logit + G(f18_salt, f18_n_acc, id))
    int32_t  f18_fast_k   = 0;     // > 0: the first active sampler of the chain is top-k with this k (fast candidate path)
    bool     f18_eligible = false;
    uint64_t f18_salt     = 0;
    uint64_t f18_n_acc    = 0; // tokens accepted since reset (prompt + generated) = noise index of the next token
};

static std::mutex                    f18_reg_mtx;
static std::vector<common_sampler *> f18_reg; // live samplers made by common_sampler_init (clones are not listed)

bool common_sampler_f18_coupling_find(const int32_t * tail, size_t n_tail, f18_coupling & out) {
    if (n_tail == 0 || !f18_coupled_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(f18_reg_mtx);
    const common_sampler * found = nullptr;
    for (const common_sampler * s : f18_reg) {
        if (!s->f18_eligible || s->params.backend_sampling || s->prev.size() < n_tail) {
            continue;
        }
        bool same = true;
        for (size_t i = 0; i < n_tail && same; ++i) {
            same = s->prev.rat(i) == tail[n_tail - 1 - i];
        }
        if (!same) {
            continue;
        }
        if (found) {
            return false; // two requests with the same recent tokens: do not guess
        }
        found = s;
    }
    if (!found) {
        return false;
    }
    out.salt  = found->f18_salt;
    out.n_acc = found->f18_n_acc;
    out.temp  = found->params.temp;
    out.top_p = found->params.top_p;
    out.min_p = found->params.min_p;
    out.top_k = found->params.top_k;
    return true;
}

// Replace the pick of the final `dist` sampler. dist leaves p_i = softmax(logit_i) over the candidates that survived the
// chain, so argmax(logit_i + G_i) is an exact sample from p. `stream` separates the grammar resample from the first draw.
static void f18_coupled_select(common_sampler * gsmpl, uint64_t stream) {
    auto & cur_p = gsmpl->cur_p;
    if (!gsmpl->f18_eligible || cur_p.selected < 0 || !f18_coupled_enabled()) {
        return;
    }
    if (cur_p.size < 2) {
        if (FILE * lf = f18_couple_log()) { // a forced pick still tells the offline fit what the verifier emitted
            fprintf(lf, "V %llu %llu %llu %d 1 %d 0\\n", (unsigned long long) gsmpl->f18_salt, (unsigned long long) stream,
                    (unsigned long long) gsmpl->f18_n_acc, (int) cur_p.data[cur_p.selected].id, (int) cur_p.data[cur_p.selected].id);
        }
        return;
    }
    const uint64_t salt = gsmpl->f18_salt ^ stream;
    double  best   = -INFINITY;
    int64_t best_i = -1;
    for (size_t i = 0; i < cur_p.size; ++i) {
        const float l = cur_p.data[i].logit;
        if (!(l > -INFINITY)) {
            continue;
        }
        const double v = (double) l + f18_gumbel(salt, gsmpl->f18_n_acc, (uint32_t) cur_p.data[i].id);
        if (v > best) {
            best   = v;
            best_i = (int64_t) i;
        }
    }
    if (best_i >= 0) {
        cur_p.selected = best_i;
    }
    if (FILE * lf = f18_couple_log()) {
        fprintf(lf, "V %llu %llu %llu %d %zu", (unsigned long long) gsmpl->f18_salt, (unsigned long long) stream,
                (unsigned long long) gsmpl->f18_n_acc, (int) cur_p.data[cur_p.selected].id, cur_p.size);
        for (size_t i = 0; i < cur_p.size; ++i) {
            fprintf(lf, " %d %.6g", (int) cur_p.data[i].id, (double) cur_p.data[i].logit);
        }
        fputc('\\n', lf);
    }
}
''')

rep('''    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    return result;
''', '''    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    // the override below is only valid where `dist` makes the final pick
    result->f18_eligible = params.mirostat == 0 &&
        std::find(params.samplers.begin(), params.samplers.end(), COMMON_SAMPLER_TYPE_ADAPTIVE_P) == params.samplers.end();
    result->f18_salt = f18_new_salt(params.seed);
    {
        std::lock_guard<std::mutex> lock(f18_reg_mtx);
        f18_reg.push_back(result);
    }

    // fast candidate path: disabled samplers are "?name" placeholders; the first real one must be top-k (so no logit bias,
    // penalties, DRY or top-n-sigma ahead of it), and nobody may ask for probabilities of the full vocabulary
    if (params.mirostat == 0 && params.n_probs == 0) {
        for (int i = 0; i < llama_sampler_chain_n(chain); ++i) {
            const char * name = llama_sampler_name(llama_sampler_chain_get(chain, i));
            if (name && name[0] == '?') {
                continue;
            }
            if (name && strcmp(name, "top-k") == 0 && params.top_k > 0 && params.top_k <= F18_FAST_K_MAX) {
                result->f18_fast_k = params.top_k;
            }
            break;
        }
    }

    return result;
''')

rep('''    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
''', '''    {
        std::lock_guard<std::mutex> lock(f18_reg_mtx);
        f18_reg.erase(std::remove(f18_reg.begin(), f18_reg.end(), gsmpl), f18_reg.end());
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
''')

rep('''    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}
''', '''    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
    gsmpl->f18_n_acc++;
}
''')

rep('''struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };
}
''', '''struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    const int64_t f18_t0 = f18_smpl_trace ? ggml_time_us() : 0;
    // the candidates of the last sample are rebuilt by the next one; the server's per-cycle clone never reads them
    const bool f18_skip_cur = f18_fast_sampler_enabled(64u);
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ f18_skip_cur ? std::vector<llama_token_data>() : gsmpl->cur,
        /* .cur_p   = */ f18_skip_cur ? llama_token_data_array{ nullptr, 0, -1, false } : gsmpl->cur_p,
    };
    result->f18_fast_k = gsmpl->f18_fast_k;

    // a clone continues the same noise field (the server restores from one after a checkpoint replay); it is not listed
    result->f18_eligible = gsmpl->f18_eligible;
    result->f18_salt     = gsmpl->f18_salt;
    result->f18_n_acc    = gsmpl->f18_n_acc;

    if (f18_smpl_trace) {
        f18_t_clone += ggml_time_us() - f18_t0;
        f18_n_clone++;
    }

    return result;
}
''')

rep('''    dst->t_total_us = src->t_total_us;
}
''', '''    dst->t_total_us = src->t_total_us;

    dst->f18_fast_k   = src->f18_fast_k;
    dst->f18_eligible = src->f18_eligible;
    dst->f18_salt     = src->f18_salt;
    dst->f18_n_acc    = src->f18_n_acc;
}
''')

rep('''    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
''', '''    const int64_t f18_t0 = f18_smpl_trace ? ggml_time_us() : 0;
    const bool f18_fast = gsmpl->f18_fast_k > 0 && f18_fast_sampler_enabled() &&
        !(grammar_first && grammar_should_apply(gsmpl)) &&
        !(gsmpl->rbudget && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING);
    gsmpl->set_logits(ctx, idx, f18_fast);
    const int64_t f18_t1 = f18_smpl_trace ? ggml_time_us() : 0;

    // Check if a backend sampler has already sampled a token in which case we
''')

rep('''    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }
''', '''    llama_sampler_apply(chain, &cur_p);
    f18_coupled_select(gsmpl, 0);

    id = cur_p.data[cur_p.selected].id;

    if (f18_smpl_trace) {
        f18_t_set_logits += f18_t1 - f18_t0;
        f18_t_chain      += ggml_time_us() - f18_t1;
        if (++f18_n_sample % 512 == 0) {
            LOG_WRN("F18_SAMPLER_TRACE samples=%lld per-sample ms: set_logits=%.3f chain=%.3f | clones=%lld per-clone ms=%.3f | candidates=%zu\\n",
                    (long long) f18_n_sample, f18_t_set_logits/1000.0/f18_n_sample, f18_t_chain/1000.0/f18_n_sample,
                    (long long) f18_n_clone, f18_n_clone ? f18_t_clone/1000.0/f18_n_clone : 0.0, gsmpl->cur.size());
        }
    }

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }
''')

rep('''    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");
''', '''    llama_sampler_apply(chain, &cur_p);
    // stock draws the resample independently of the rejected first draw; a second noise stream keeps exactly that law
    f18_coupled_select(gsmpl, 0x5EC0DD12A11ull);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");
''')

(HERE/'sampling.f18c.cpp').write_text(box[0])

# ---------------------------------------------------------------- speculative.cpp
box, rep = patcher((HERE/'speculative.f18b.cpp').read_text())

rep('#include "sampling.h"\n', '#include "sampling.h"\n#include "f18-coupling.inc"\n')

rep("    bool f18_pad       = false;\n", "    bool f18_pad       = false;\n    bool f18_couple    = false; // draft against the verifier's noise (f18-coupling.inc)\n")
rep("            f18_pad       = f18_merge_allowed && (mask & 4u);\n",
    "            f18_pad       = f18_merge_allowed && (mask & 4u);\n            f18_couple    = (mask & 16u) != 0;\n")
rep('        f18_trace     = f18_env("GGML_F18_SPEC_TRACE");\n',
    '        f18_trace     = f18_env("GGML_F18_SPEC_TRACE");\n        f18_couple    = f18_env("GGML_F18_COUPLED");\n        f18_cpl.assign(n_seq, {});\n        f18_cpl_ok.assign(n_seq, false);\n')
rep('        SPC_WRN("f18: mtp merge=%d fast_pick=%d pad=%d trace=%d\\n", (int) f18_merge, (int) f18_fast_pick, (int) f18_pad, (int) f18_trace);',
    '        SPC_WRN("f18: mtp merge=%d fast_pick=%d pad=%d coupled=%d trace=%d\\n", (int) f18_merge, (int) f18_fast_pick, (int) f18_pad, (int) f18_couple, (int) f18_trace);')

# the top-K scan becomes its own function; the coupled pick shares it
rep('''    // same result as the TOP_K(10)+dist chain read through get_candidates: the best token and its softmax share of the top 10
    static llama_token f18_pick(const float * logits, int n_vocab, float & p_top) {
        const int K = 10;
        float best[K];
        int   best_id[K];
        int   n = 0;
        float thr = -INFINITY;
''', '''    std::vector<f18_coupling> f18_cpl;    // per sequence: the verifier's salt, counter and truncation for this draft call
    std::vector<bool>         f18_cpl_ok;
    int64_t f18_n_coupled = 0, f18_n_uncoupled = 0;

    static constexpr int F18_PICK_K = 10;

    // the F18_PICK_K largest logits, descending
    static int f18_top(const float * logits, int n_vocab, float * best, int * best_id) {
        const int K = F18_PICK_K;
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
        return n;
    }

    // Draft the token the verifier is most likely to DRAW: cut the draft distribution the way the verifier's chain will cut
    // its own (top_k, top_p, min_p on the untempered distribution, temperature last), add the verifier's noise for this
    // position, take the argmax. `step` is the draft depth: the verifier samples that token with counter n_acc + step.
    static llama_token f18_pick_coupled(const float * logits, int n_vocab, const f18_coupling & c, uint64_t step, float & p_top) {
        float best[F18_PICK_K]    = { 0.0f };
        int   best_id[F18_PICK_K] = { 0 };
        const int n = f18_top(logits, n_vocab, best, best_id);

        double pr[F18_PICK_K];
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            pr[j] = exp((double) (best[j] - best[0]));
            sum += pr[j];
        }
        int n_keep = n;
        if (c.top_k > 0) {
            n_keep = std::min(n_keep, (int) c.top_k);
        }
        if (c.top_p < 1.0f) {
            double cum = 0.0;
            for (int j = 0; j < n_keep; ++j) {
                cum += pr[j]/sum;
                if (cum >= c.top_p) {
                    n_keep = j + 1;
                    break;
                }
            }
        }
        if (c.min_p > 0.0f) {
            for (int j = 1; j < n_keep; ++j) {
                if (pr[j] < (double) c.min_p*pr[0]) {
                    n_keep = j;
                    break;
                }
            }
        }

        // GGML_F18_COUPLED_DRAFT_TEMP scales the drafter's temperature relative to the verifier's (1.0 = trust the head as it is)
        static const double temp_scale = [] {
            const char * v = getenv("GGML_F18_COUPLED_DRAFT_TEMP");
            const double x = v ? atof(v) : 1.0;
            return x > 0.0 ? x : 1.0;
        }();
        const double inv_t = 1.0/((double) c.temp*temp_scale);
        double best_v = -INFINITY;
        int    best_j = 0;
        double sum_t  = 0.0;
        for (int j = 0; j < n_keep; ++j) {
            const double lt = (double) (best[j] - best[0])*inv_t;
            sum_t += exp(lt);
            const double v = lt + f18_gumbel(c.salt, c.n_acc + step, (uint32_t) best_id[j]);
            if (v > best_v) {
                best_v = v;
                best_j = j;
            }
        }
        p_top = (float) (exp((double) (best[best_j] - best[0])*inv_t)/sum_t);
        if (FILE * lf = f18_couple_log()) {
            fprintf(lf, "D %llu %llu %llu %d %d %d %.6g %.6g %.6g %d", (unsigned long long) c.salt, (unsigned long long) step,
                    (unsigned long long) (c.n_acc + step), best_id[best_j], n_keep, n, (double) c.temp, (double) c.top_p, (double) c.min_p, (int) c.top_k);
            for (int j = 0; j < n; ++j) {
                fprintf(lf, " %d %.6g", best_id[j], (double) best[j]);
            }
            fputc('\\n', lf);
        }
        return best_id[best_j];
    }

    // same result as the TOP_K(10)+dist chain read through get_candidates: the best token and its softmax share of the top 10
    static llama_token f18_pick(const float * logits, int n_vocab, float & p_top) {
        const int K = 10;
        float best[K];
        int   best_id[K];
        int   n = 0;
        float thr = -INFINITY;
''')

# per draft call: find the verifier sampler of this sequence by its recent tokens
rep('''            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            const int f18_rows_begin = batch.n_tokens;
''', '''            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            f18_cpl_ok[seq_id] = false;
            if (f18_couple && f18_fast_pick && backend_chains[seq_id] == nullptr && dp.prompt) {
                // the verifier's sampler has accepted the prompt and everything generated, id_last included
                llama_token tail[16];
                size_t n_tail = 0;
                const auto & pr = *dp.prompt;
                for (size_t k = pr.size() - std::min<size_t>(15, pr.size()); k < pr.size(); ++k) {
                    tail[n_tail++] = pr[k];
                }
                tail[n_tail++] = dp.id_last;
                f18_coupling c;
                if (common_sampler_f18_coupling_find(tail, n_tail, c) && c.temp > 0.0f) {
                    f18_cpl[seq_id]    = c;
                    f18_cpl_ok[seq_id] = true;
                }
                (f18_cpl_ok[seq_id] ? f18_n_coupled : f18_n_uncoupled)++;
            }

            const int f18_rows_begin = batch.n_tokens;
''')

rep('''                    id    = f18_pick(logits, llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft))), p_top);
''', '''                    const int n_vocab_dft = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
                    id    = f18_cpl_ok[seq_id] ? f18_pick_coupled(logits, n_vocab_dft, f18_cpl[seq_id], (uint64_t) i, p_top)
                                               : f18_pick(logits, n_vocab_dft, p_top);
''')

rep('''            SPC_WRN("F18_SPEC_TRACE drafts=%lld per-cycle ms: process=%.3f decode1=%.3f pick1=%.3f decode2=%.3f pick2=%.3f merged_rows=%.2f flushes=%lld\\n",
                    (long long) f18_n_draft, f18_t_process/1000.0/f18_n_draft, f18_t_decode1/1000.0/f18_n_draft, f18_t_pick1/1000.0/f18_n_draft,
                    f18_t_decode2/1000.0/f18_n_draft, f18_t_pick2/1000.0/f18_n_draft, (double) f18_n_merged/f18_n_draft, (long long) f18_n_flushed);''',
'''            SPC_WRN("F18_SPEC_TRACE drafts=%lld per-cycle ms: process=%.3f decode1=%.3f pick1=%.3f decode2=%.3f pick2=%.3f merged_rows=%.2f flushes=%lld coupled=%lld uncoupled=%lld\\n",
                    (long long) f18_n_draft, f18_t_process/1000.0/f18_n_draft, f18_t_decode1/1000.0/f18_n_draft, f18_t_pick1/1000.0/f18_n_draft,
                    f18_t_decode2/1000.0/f18_n_draft, f18_t_pick2/1000.0/f18_n_draft, (double) f18_n_merged/f18_n_draft, (long long) f18_n_flushed,
                    (long long) f18_n_coupled, (long long) f18_n_uncoupled);''')

(HERE/'speculative.f18c.cpp').write_text(box[0])
print('wrote sampling.f18c.cpp and speculative.f18c.cpp')
