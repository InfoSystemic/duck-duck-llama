// vision-embd: run ONE image through an mmproj (clip) exactly as llama-server does, and dump
//   - the projector output embeddings  [n_tokens x n_embd] float32   -> <out>
//   - optionally every named ViT intermediate (patch_embed, layer_out-N, ...) -> <dump_dir>/<name>.bin
// so the llama.cpp vision tower can be compared tensor-by-tensor against the PyTorch reference
// (vision-ref.py). Only the text model's VOCAB is loaded (vocab_only), so this takes seconds.
//
// usage: vision-embd <mmproj.gguf> <text-model.gguf> <image> <out.bin> [fa 0|1] [dump_dir] [n_threads]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_dump_dir;

static bool want_tensor(const char * name) {
    static const char * prefixes[] = {
        "patch_embed", "layer_out-", "attn_out-", "ffn_inp-", "kqv_out-", "Qcur_rope-", "Kcur_rope-",
        "reorder_to_", "post_ln", "vit_out",
    };
    for (const char * p : prefixes) {
        if (strncmp(name, p, strlen(p)) == 0) {
            return true;
        }
    }
    return false;
}

static bool g_stats_all = getenv("VE_STATS") != nullptr;

static bool dump_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const char * name = ggml_get_name(t);
    if (ask) {
        return g_stats_all || want_tensor(name);
    }
    if (g_stats_all) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return true;
        }
        const int64_t n = ggml_nelements(t);
        std::vector<float> b(n);
        ggml_backend_tensor_get(t, b.data(), 0, ggml_nbytes(t));
        double mx = 0; int64_t bad = 0;
        for (float v : b) { if (!std::isfinite(v)) bad++; else mx = std::max(mx, (double) std::fabs(v)); }
        fprintf(stderr, "STAT %-28s %-10s ne=[%lld,%lld,%lld] maxabs %.4g nonfinite %lld\n", name, ggml_op_desc(t),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], mx, (long long) bad);
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        return true;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        // gather a non-contiguous view element by element (small tensors only)
        std::vector<char> raw(ggml_nbytes(t) + t->nb[3] * t->ne[3]);
        const size_t span = (t->ne[3] - 1) * t->nb[3] + (t->ne[2] - 1) * t->nb[2] + (t->ne[1] - 1) * t->nb[1] + (t->ne[0]) * t->nb[0];
        raw.resize(span);
        ggml_backend_tensor_get(t, raw.data(), 0, span);
        int64_t o = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
        for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
        for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
            buf[o++] = *(const float *)(raw.data() + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3]);
        }
    }
    int64_t n_bad = 0;
    for (float v : buf) {
        n_bad += !std::isfinite(v);
    }
    std::string path = g_dump_dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (f) {
        int64_t hdr[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(buf.data(), sizeof(float), n, f);
        fclose(f);
    }
    if (n_bad) {
        fprintf(stderr, "NONFINITE %s: %lld of %lld\n", name, (long long) n_bad, (long long) n);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <mmproj.gguf> <text-model.gguf> <image> <out.bin> [fa 0|1] [dump_dir] [n_threads]\n", argv[0]);
        return 1;
    }
    const char * mmproj = argv[1];
    const char * tmodel = argv[2];
    const char * image  = argv[3];
    const char * out    = argv[4];
    const int    fa     = argc > 5 ? atoi(argv[5]) : 1;
    if (argc > 6 && argv[6][0]) {
        g_dump_dir = argv[6];
    }
    const int n_threads = argc > 7 ? atoi(argv[7]) : 8;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(tmodel, mp);
    if (!model) {
        fprintf(stderr, "failed to load text model vocab\n");
        return 1;
    }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu         = false;
    cp.print_timings   = true;
    cp.n_threads       = n_threads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.warmup          = false;
    if (!g_dump_dir.empty()) {
        cp.cb_eval           = dump_cb;
        cp.cb_eval_user_data = nullptr;
    }
    mtmd_context * ctx = mtmd_init_from_file(mmproj, model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to init mtmd\n");
        return 1;
    }

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image, false);
    if (!bw.bitmap) {
        fprintf(stderr, "failed to load image %s\n", image);
        return 1;
    }
    fprintf(stderr, "image %s: %u x %u\n", image, mtmd_bitmap_get_nx(bw.bitmap), mtmd_bitmap_get_ny(bw.bitmap));

    mtmd_input_text txt;
    txt.text          = mtmd_default_marker();
    txt.text_len      = strlen(txt.text);
    txt.add_special   = false;
    txt.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = { bw.bitmap };
    if (mtmd_tokenize(ctx, chunks, &txt, bitmaps, 1) != 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }

    // vocab_only loads no hparams, so n_embd_inp reads 0 there; the projector width comes from the env or 6144
    const int n_embd = llama_model_n_embd_inp(model) > 0 ? llama_model_n_embd_inp(model) : (getenv("VE_N_EMBD") ? atoi(getenv("VE_N_EMBD")) : 6144);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        const auto ctype = mtmd_input_chunk_get_type(c);
        if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            continue;
        }
        const size_t n_tok = mtmd_input_chunk_get_n_tokens(c);
        const mtmd_image_tokens * it = ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE ? mtmd_input_chunk_get_tokens_image(c) : nullptr;
        fprintf(stderr, "%s chunk: n_tokens %zu  grid %zu x %zu\n", it ? "image" : "audio", n_tok,
                it ? (size_t) mtmd_image_tokens_get_nx(it) : n_tok, it ? (size_t) mtmd_image_tokens_get_ny(it) : (size_t) 1);
        if (mtmd_encode_chunk(ctx, c) != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        const float * e = mtmd_get_output_embd(ctx);
        int64_t n_bad = 0;
        double ss = 0;
        for (size_t k = 0; k < n_tok * (size_t) n_embd; k++) {
            if (!std::isfinite(e[k])) { n_bad++; } else { ss += (double) e[k] * e[k]; }
        }
        fprintf(stderr, "embd: nonfinite %lld  rms %.5f\n", (long long) n_bad, std::sqrt(ss / (n_tok * (double) n_embd)));
        FILE * f = fopen(out, "wb");
        int64_t hdr[4] = { n_embd, (int64_t) n_tok, it ? (int64_t) mtmd_image_tokens_get_nx(it) : (int64_t) n_tok, it ? (int64_t) mtmd_image_tokens_get_ny(it) : 1 };
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(e, sizeof(float), n_tok * n_embd, f);
        fclose(f);
        break;
    }

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
