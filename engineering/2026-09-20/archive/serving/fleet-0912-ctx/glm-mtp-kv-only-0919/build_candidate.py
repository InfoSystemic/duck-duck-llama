#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, subprocess, time, difflib
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reference=json.loads((HERE/'reference-build.json').read_text())
assert reference['object_exact'] and sha(HERE/'reference/libllama.so.0.3.0')==reference['expected_library_sha256']
s=(HERE/'reference/glm5next.cpp').read_text()
s=s.replace('#include "llama-kv-cache-kpool.h"','#include "llama-kv-cache-kpool.h"\n\n#include <algorithm>\n#include <cstdlib>\n#include <cstring>\n#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>')
needle='// positions the indexer keeps:'
idx=s.index(needle)
s=s[:idx]+'''namespace {
static bool glm5next_kv_only_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_GLM5N_MTP_KV_ONLY");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return false;
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = std::getenv("GGML_GLM5N_MTP_KV_ONLY_CONTROL_FILE");
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
    const uint32_t value = __atomic_load_n(control, __ATOMIC_ACQUIRE);
    GGML_ASSERT(value <= 1);
    return value == 1;
}

class glm5next_input_kv_mode final : public llm_graph_input_i {
public:
    explicit glm5next_input_kv_mode(bool enabled) : enabled(enabled) {}
    void set_input(const llama_ubatch *) override {}
    bool can_reuse(const llm_graph_params &) override {
        return enabled == glm5next_kv_only_enabled();
    }
    const bool enabled;
};

class glm5next_input_k_store final : public llm_graph_input_i {
public:
    glm5next_input_k_store(ggml_tensor * indices, const llama_kv_cache_context * memory)
        : indices(indices), memory(memory) {}

    void set_input(const llama_ubatch * batch) override {
        memory->set_input_k_idxs(indices, batch);
    }

    // Keep the base class's no-reuse policy for this cache-only graph.
    ggml_tensor * indices;
    const llama_kv_cache_context * memory;
};
}

'''+s[idx:]
start=s.index('llama_model_glm5next::graph_mtp::graph_mtp(')
a,b=s[:start],s[start:]
old='''    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA absorption leaves a K-only cache holding the latent, as in the trunk
    auto * inp_attn = build_attn_inp_k();'''
new='''    const bool kv_only_enabled = glm5next_kv_only_enabled();
    res->add_input(std::make_unique<glm5next_input_kv_mode>(kv_only_enabled));
    const bool kv_only = kv_only_enabled && n_outputs == 0 &&
        hparams.n_layer_nextn == 1 && !cparams.embeddings &&
        cparams.embeddings_nextn && cparams.embeddings_nextn_masked &&
        params.samplers.empty() &&
        std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                     [](bool value) { return value; });

    ggml_tensor * inp_out_ids = kv_only ? nullptr : build_inp_out_ids();

    // MLA absorption leaves a K-only cache holding the latent, as in the trunk.
    auto * inp_attn = kv_only ? nullptr : build_attn_inp_k();'''
assert b.count(old)==1
b=b.replace(old,new)
old='''        cb(cur, "mtp_attn_norm", il);

        cur = build_dsa_layer'''
new='''        cb(cur, "mtp_attn_norm", il);

        if (kv_only) {
            // With no outputs, this layer only needs to store its normalized KV latent.
            auto * memory = static_cast<const llama_kv_cache_context *>(mctx);
            auto * indices = memory->build_input_k_idxs(ctx0, ubatch);
            res->add_input(std::make_unique<glm5next_input_k_store>(indices, memory));

            auto * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv, "dsa_kv_a_norm", il);
            auto * k = ggml_reshape_3d(ctx0, kv, hparams.n_lora_kv, 1, n_tokens);
            cb(k, "dsa_kv_latent", il);
            ggml_build_forward_expand(gf, memory->cpy_k(ctx0, k, indices, il));

            static const bool probe = [] {
                const char * value = std::getenv("GGML_GLM5N_MTP_KV_ONLY_PROBE");
                return value && std::strcmp(value, "1") == 0;
            }();
            if (probe) {
                LLAMA_LOG_WARN("GLM_MTP_KV_ONLY tokens=%u outputs=%lld nodes=%d\\n",
                               ubatch.n_tokens, (long long) n_outputs, ggml_graph_n_nodes(gf));
            }
            return;
        }

        cur = build_dsa_layer'''
assert b.count(old)==1
b=b.replace(old,new)
source=HERE/'candidate/glm5next.cpp';source.write_text(a+b)
(HERE/'mtp-kv-only.patch').write_text(''.join(difflib.unified_diff((HERE/'reference/glm5next.cpp').read_text().splitlines(True),source.read_text().splitlines(True),fromfile='original/glm5next.cpp',tofile='candidate/glm5next.cpp')))
command=[x.replace(str(HERE/'reference'),str(HERE/'candidate')) for x in reference['compile']]
with (HERE/'logs/candidate-compile.log').open('w') as log:
    subprocess.run(['nice','-n','10']+command,cwd=reference['compile_cwd'],stdout=log,stderr=subprocess.STDOUT,check=True)
link=reference['link'][:]
link[link.index(str(HERE/'reference/glm5next.cpp.o'))]=str(HERE/'candidate/glm5next.cpp.o')
link[link.index('-o')+1]=str(HERE/'build/libllama.so.0.3.0')
subprocess.run(link,cwd=reference['link_cwd'],check=True)
for name in ['libllama.so','libllama.so.0']:
    p=HERE/'build'/name
    if not p.exists():p.symlink_to('libllama.so.0.3.0')
report={'built_at':time.time(),'parent_sha256':reference['expected_library_sha256'],'candidate_sha256':sha(HERE/'build/libllama.so.0.3.0'),'source_sha256':sha(source),'changed_objects':['models/glm5next.cpp.o'],'compile':command,'link':link,'flag':'GGML_GLM5N_MTP_KV_ONLY=1 (default off)','deployed':False}
(HERE/'candidate-build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
