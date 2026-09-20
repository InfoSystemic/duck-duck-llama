#!/usr/bin/env python3
"""make_f18_pad.py -- stage 2 over speculative.f18.cpp: constant-shape MTP draft batches (GGML_F18_MTP_PAD=1, control bit 2).

llama.cpp reuses only the previous graph. The two draft passes of a cycle have different shapes (k+2 rows, then 1 row), so
both graphs are rebuilt and reallocated every cycle (2.2 ms each on this engine). Padding every draft batch of a sequence to
n_max + 2 rows makes all draft decodes the same shape, so every one reuses the previous graph.
Pad rows sit at FUTURE positions of the same sequence: the real row cannot attend to them (causal mask), they request no
output, and their cache cells are removed before the next pass and by the server's draft-region cleanup.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
s = (HERE/'speculative.f18.cpp').read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)

rep("    bool f18_fast_pick = false;\n", "    bool f18_fast_pick = false;\n    bool f18_pad       = false;\n    // pad rows attend densely in the draft layer; past this context that costs more than the graph rebuild it saves\n    llama_pos f18_pad_max_ctx = 12288;\n")
rep("            f18_fast_pick = (mask & 2u) != 0;\n",
    "            f18_fast_pick = (mask & 2u) != 0;\n            f18_pad       = f18_merge_allowed && (mask & 4u);\n")
rep('        f18_fast_pick = f18_env("GGML_F18_MTP_FAST_PICK");\n',
    '        f18_fast_pick = f18_env("GGML_F18_MTP_FAST_PICK");\n        f18_pad       = f18_env("GGML_F18_MTP_PAD") && f18_merge_allowed;\n        if (const char * v = getenv("GGML_F18_MTP_PAD_MAX_CTX")) { f18_pad_max_ctx = (llama_pos) atoll(v); }\n')
rep('        SPC_WRN("f18: mtp merge=%d fast_pick=%d trace=%d\\n", (int) f18_merge, (int) f18_fast_pick, (int) f18_trace);',
    '        SPC_WRN("f18: mtp merge=%d fast_pick=%d pad=%d trace=%d\\n", (int) f18_merge, (int) f18_fast_pick, (int) f18_pad, (int) f18_trace);')

# pass 1
rep('''            if (!deferred[seq_id].tok.empty()) {
                // rows at pos >= n_past were rejected; the rest continue the draft cache''',
'''            const int f18_rows_begin = batch.n_tokens;

            if (!deferred[seq_id].tok.empty()) {
                // rows at pos >= n_past were rejected; the rest continue the draft cache''')
rep('''            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {''',
'''            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (f18_pad && dp.n_past < f18_pad_max_ctx) {
                // same row count for every draft decode of this sequence
                for (int j = 1; batch.n_tokens - f18_rows_begin < params.n_max + 2; ++j) {
                    common_batch_add(batch, dp.id_last, dp.n_past + j, { seq_id }, false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);
                }
            }

            if (chain_heads) {''')

# later passes
rep('''                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }
''',
'''                } else {
                    if (f18_pad && dp.n_past < f18_pad_max_ctx) {
                        // drop the pad rows of the pass that just ran
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, dp.n_past + i + 1, -1);
                    }
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;

                if (f18_pad && dp.n_past < f18_pad_max_ctx && !chain_heads && !is_mem_shared) {
                    for (int j = 1; j < params.n_max + 2; ++j) {
                        common_batch_add(batch, id, dp.n_past + i + 1 + j, { seq_id }, false);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                    }
                }
            }
''')
(HERE/'speculative.f18b.cpp').write_text(s)
print('wrote speculative.f18b.cpp')
