#!/usr/bin/env python3
"""make_mtp_qrows.py -- glm5next.f18.cpp: in the MTP draft graph, build the query side of attention only for output rows.

Parent: the deployed glm-mtp-kv-only-0919/candidate/glm5next.cpp. Switch: LLAMA_F18_MTP_QROWS=1 (optional 4-byte control file
LLAMA_F18_MTP_QROWS_CONTROL_FILE for in-process A/B; a flip forces a graph rebuild).

A draft batch is [accepted rows that only have to reach the draft cache][the one row that predicts][pad rows that keep the
graph shape constant]. The stock graph runs the whole DSA layer on all of them and throws the extra rows away just before
the FFN. Every row needs its latent K in the cache; only the predicting row needs a query, the dense attention over the
whole context, and the output projection. Here Q is built from get_rows(cur, out_ids) and the mask is sliced the same way.
The predicting row sees the same K cache and the same mask row, so its result is unchanged; the other rows cost a K
projection and nothing else, at any context length.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-mtp-kv-only-0919/candidate/glm5next.cpp')
s = SRC.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70]); s = s.replace(old, new)

rep('''class glm5next_input_k_store final : public llm_graph_input_i {''', '''static bool glm5next_mtp_qrows_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_F18_MTP_QROWS");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return false;
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = std::getenv("LLAMA_F18_MTP_QROWS_CONTROL_FILE");
        if (!path || !*path) return nullptr;
        const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        GGML_ASSERT(fd >= 0);
        struct stat info;
        GGML_ASSERT(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size == sizeof(uint32_t));
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        GGML_ASSERT(mapping != MAP_FAILED);
        return static_cast<const uint32_t *>(mapping);
    }();
    if (!control) return true;
    return __atomic_load_n(control, __ATOMIC_ACQUIRE) == 1;
}

// a graph built for one setting of the switch must not be reused under the other
class glm5next_input_qrows_mode final : public llm_graph_input_i {
public:
    explicit glm5next_input_qrows_mode(bool enabled) : enabled(enabled) {}
    void set_input(const llama_ubatch *) override {}
    bool can_reuse(const llm_graph_params &) override {
        return enabled == glm5next_mtp_qrows_enabled();
    }
    const bool enabled;
};

class glm5next_input_k_store final : public llm_graph_input_i {''')

rep('''        cur = build_dsa_layer(layer, inp_attn, /*inp_kp =*/ nullptr, /*scoring =*/ false, cur, il);
        cb(cur, "mtp_attn_out", il);

        if (inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }
''', '''        const bool qrows_enabled = glm5next_mtp_qrows_enabled();
        res->add_input(std::make_unique<glm5next_input_qrows_mode>(qrows_enabled));
        const bool qrows = qrows_enabled && inp_out_ids && cparams.flash_attn &&
            n_outputs > 0 && n_outputs < (int64_t) n_tokens &&
            inp_attn->get_kq_mask()->ne[2] == 1 && inp_attn->get_kq_mask()->ne[3] == 1;

        if (qrows) {
            // build_dsa_layer + build_attn(attn_k), with the query side restricted to the output rows
            const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
            const int64_t kv_lora_rank = hparams.n_lora_kv;
            const float   kq_scale     = 1.0f/sqrtf(float(qk_head_dim));

            ggml_tensor * cur_q = ggml_get_rows(ctx0, cur, inp_out_ids);

            ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur_q);
            qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "dsa_q_a_norm", il);

            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, qk_head_dim, n_head, cur_q->ne[1]);
            cb(q, "dsa_q_b", il);

            // every row of the batch still writes its latent K
            ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv, "dsa_kv_a_norm", il);

            q = ggml_permute(ctx0, q, 0, 2, 1, 3);
            q = ggml_mul_mat(ctx0, layer.wk_b, q);
            q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
            cb(q, "dsa_q_absorbed", il);

            ggml_tensor * k = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
            cb(k, "dsa_kv_latent", il);

            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, k);

            const auto * mctx_cur = inp_attn->mctx;
            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k, inp_attn->get_k_idxs(), il));

            // mask rows of the output tokens (get_rows yields F32; 0 and -inf survive the round trip)
            ggml_tensor * mask = ggml_cast(ctx0, ggml_get_rows(ctx0, inp_attn->get_kq_mask(), inp_out_ids), GGML_TYPE_F16);
            cb(mask, "mtp_kq_mask_out", il);

            ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
            ggml_tensor * v_all = ggml_view_4d(ctx0, k_all, k->ne[0], k_all->ne[1], k_all->ne[2], k_all->ne[3],
                    k_all->nb[1], k_all->nb[2], k_all->nb[3], 0);

            cur = build_attn_mha(q, k_all, v_all, nullptr, mask, nullptr, layer.wv_b, kq_scale, il);
            cb(cur, "kqv_out", il);

            cur = build_lora_mm(layer.wo, cur);
            cb(cur, "dsa_out", il);
            cb(cur, "mtp_attn_out", il);

            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        } else {
            cur = build_dsa_layer(layer, inp_attn, /*inp_kp =*/ nullptr, /*scoring =*/ false, cur, il);
            cb(cur, "mtp_attn_out", il);

            if (inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }
        }
''')
(HERE/'glm5next.f18.cpp').write_text(s)
print('wrote glm5next.f18.cpp')
