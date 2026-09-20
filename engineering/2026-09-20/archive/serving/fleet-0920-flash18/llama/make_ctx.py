#!/usr/bin/env python3
"""make_ctx.py -- derive llama-context.f18.cpp from the engine's llama-context.cpp (each anchor must match once).

LLAMA_F18_GRAPH_SLOTS=N (default 0 = off) keeps up to N extra (scheduler, graph result) pairs per context for small ubatches
(<= LLAMA_F18_GRAPH_SLOTS_MAX_TOKENS, default 8). llama.cpp reuses only the previous graph, so a speculative loop whose draft
passes alternate between two shapes rebuilds and reallocates a graph on every call (2.2 ms each here). With slots, a small
ubatch that does not match the current graph looks for a matching slot before rebuilding, and rebuilds into the least recently
used slot otherwise; the primary pair keeps whatever it held (the verify graph, a prefill graph).
State lives in a side table keyed by the context so the class layout, and every other object file, stays untouched.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/src/llama-context.cpp')
s = SRC.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)

rep('#include <string>\n', '''#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

// f18: alternate graph slots, see process_ubatch
namespace {
struct f18_graph_slot {
    ggml_backend_sched_ptr sched;
    llm_graph_result_ptr   res;
    uint64_t               tick = 0;
};
struct f18_graph_slots {
    std::vector<f18_graph_slot> alt;
    int      active = -1; // index of the slot whose pair currently sits in the context members
    uint64_t tick   = 0;
    uint64_t n_hit  = 0, n_build = 0;
};
std::mutex f18_slots_mu;
std::unordered_map<const llama_context *, f18_graph_slots> f18_slots_tab;

int f18_slots_n() {
    static const int n = [] { const char * v = getenv("LLAMA_F18_GRAPH_SLOTS"); return v ? std::max(0, atoi(v)) : 0; }();
    return n;
}
uint32_t f18_slots_max_tokens() {
    static const uint32_t n = [] { const char * v = getenv("LLAMA_F18_GRAPH_SLOTS_MAX_TOKENS"); return v ? (uint32_t) std::max(1, atoi(v)) : 8u; }();
    return n;
}
f18_graph_slots * f18_slots_get(const llama_context * ctx, bool create) {
    std::lock_guard<std::mutex> lock(f18_slots_mu);
    auto it = f18_slots_tab.find(ctx);
    if (it == f18_slots_tab.end()) {
        if (!create) {
            return nullptr;
        }
        it = f18_slots_tab.emplace(ctx, f18_graph_slots{}).first;
    }
    return &it->second;
}
void f18_slots_erase(const llama_context * ctx) {
    std::lock_guard<std::mutex> lock(f18_slots_mu);
    f18_slots_tab.erase(ctx);
}
}

// f18: put the primary (scheduler, graph) pair back into the members; drop the slots or only what they hold
#define f18_graph_slots_clear(drop) do {                                              \\
    if (f18_slots_n() > 0) {                                                          \\
        if (f18_graph_slots * fs_ = f18_slots_get(this, false)) {                     \\
            if (fs_->active >= 0) {                                                   \\
                std::swap(sched,       fs_->alt[fs_->active].sched);                  \\
                std::swap(gf_res_prev, fs_->alt[fs_->active].res);                    \\
                fs_->active = -1;                                                     \\
            }                                                                         \\
            if (drop) {                                                               \\
                f18_slots_erase(this);                                                \\
            } else {                                                                  \\
                for (auto & slot_ : fs_->alt) {                                       \\
                    if (slot_.res) {                                                  \\
                        slot_.res->reset();                                           \\
                    }                                                                 \\
                }                                                                     \\
            }                                                                         \\
        }                                                                             \\
    }                                                                                 \\
} while (0)
''')

# destructor: drop the slots while the backends still exist
rep('''llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
''', '''llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    f18_graph_slots_clear(true);
''')

# helper methods cannot be added to the class without touching the header, so use free functions taking the members by reference
rep('''void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;
''', '''void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    // f18: the scheduler is rebuilt below, so the slots go too
    f18_graph_slots_clear(true);
''')

rep('''        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();
''', '''        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        f18_graph_slots_clear(false);
        gf_res_prev->reset();
''')

rep('''    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();
''', '''    // f18: bring the primary pair back before touching it, and forget what the slots hold
    f18_graph_slots_clear(false);

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();
''')

rep('''    const int64_t phase_applied = phase_profile ? ggml_time_us() : 0;
    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();
''', '''    const int64_t phase_applied = phase_profile ? ggml_time_us() : 0;

    // f18: look for a slot that already holds this small graph before rebuilding the current one
    if (f18_slots_n() > 0 && !graph_reuse_disable) {
        f18_graph_slots * fs = f18_slots_get(this, true);
        if (fs->active >= 0) {
            std::swap(sched,       fs->alt[fs->active].sched);
            std::swap(gf_res_prev, fs->alt[fs->active].res);
            fs->active = -1;
        }
        if (ubatch.n_tokens <= f18_slots_max_tokens() && !cparams.pipeline_parallel &&
                !gf_res_prev->can_reuse(graph_params(gf_res_prev.get(), ubatch, mctx, gtype))) {
            if (fs->alt.empty()) {
                fs->alt.resize(f18_slots_n());
            }
            int pick = -1;
            for (int j = 0; j < (int) fs->alt.size() && pick < 0; ++j) {
                if (!fs->alt[j].res) {
                    continue;
                }
                std::swap(sched,       fs->alt[j].sched);
                std::swap(gf_res_prev, fs->alt[j].res);
                if (gf_res_prev->can_reuse(graph_params(gf_res_prev.get(), ubatch, mctx, gtype))) {
                    pick = j;
                    fs->n_hit++;
                } else {
                    std::swap(sched,       fs->alt[j].sched);
                    std::swap(gf_res_prev, fs->alt[j].res);
                }
            }
            if (pick < 0) {
                // no slot holds it: build into an empty slot, else into the least recently used one
                for (int j = 0; j < (int) fs->alt.size(); ++j) {
                    if (!fs->alt[j].res) {
                        pick = j;
                        break;
                    }
                    if (pick < 0 || fs->alt[j].tick < fs->alt[pick].tick) {
                        pick = j;
                    }
                }
                auto & slot = fs->alt[pick];
                if (!slot.res) {
                    const size_t max_nodes = this->graph_max_nodes(f18_slots_max_tokens());
                    slot.res.reset(new llm_graph_result(max_nodes));
                    slot.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                }
                std::swap(sched,       slot.sched);
                std::swap(gf_res_prev, slot.res);
                fs->n_build++;
            }
            fs->active = pick;
            fs->alt[pick].tick = ++fs->tick;
        }
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();
''')

(HERE/'llama-context.f18.cpp').write_text(s)
print('wrote llama-context.f18.cpp')
