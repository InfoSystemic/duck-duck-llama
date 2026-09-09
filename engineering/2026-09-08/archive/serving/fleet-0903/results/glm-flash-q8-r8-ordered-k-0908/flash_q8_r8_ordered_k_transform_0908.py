"""Create the private narrow Q8 projection path and its audited graph fixture."""


def transform_source(source):
    marker = '#include "ggml-quants.h"\n'
    assert source.count(marker) == 1
    additions = '''
#include "flash-q8-r8-ordered-k-0908.h"

static bool flash_q8_r8_ordered_k_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_Q8_R8_ORDERED_K");
        return value && std::strcmp(value,"1") == 0;
    }();
    return enabled;
}
static bool flash_q8_r8_ordered_k_audit() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_Q8_R8_ORDERED_K_AUDIT");
        return value && std::strcmp(value,"1") == 0;
    }();
    return enabled;
}
static std::atomic<uint64_t> flash_q8_r8_ordered_k_calls{0};
extern "C" uint64_t ggml_cpu_q8_r8_ordered_k_count(int);
extern "C" uint64_t ggml_cpu_q8_r8_ordered_k_count(int) {
    return flash_q8_r8_ordered_k_calls.load(std::memory_order_relaxed);
}
'''
    source = source.replace(marker, marker + additions)
    start = source.index('template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits')
    end = source.index('class tensor_traits_router_f16', start)
    body = source[start:end]
    marker = '    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {'
    assert body.count(marker) == 1
    eligible = '''    static bool ordered_k_eligible(int n_threads, const ggml_tensor * op) {
        if constexpr (!std::is_same_v<BLOC_TYPE, block_q8_0> || INTER_SIZE != 8 || NB_COLS != 8 || PARAM_TYPE != GGML_TYPE_Q8_0) {
            return false;
        }
        if (!flash_q8_r8_ordered_k_enabled() || n_threads <= 1 || op->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor * a = op->src[0], * b = op->src[1];
        return a->type == GGML_TYPE_Q8_0 && b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            a->ne[0] >= 4096 && a->ne[0] <= 32768 && a->ne[0] % QK8_0 == 0 && a->ne[1] == 24 &&
            a->ne[2] == 1 && a->ne[3] == 1 && b->ne[0] == a->ne[0] &&
            b->ne[1] >= 1 && b->ne[1] <= 3 && b->ne[2] == 1 && b->ne[3] == 1 &&
            op->ne[0] == 24 && op->ne[1] == b->ne[1] && op->ne[2] == 1 && op->ne[3] == 1 &&
            b->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && op->nb[1] % sizeof(float) == 0;
    }

'''
    body = body.replace(marker, eligible + marker)
    old = '''                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    return true;'''
    assert body.count(old) == 1
    body = body.replace(old, '''                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    if (ordered_k_eligible(n_threads, op)) {
                        const size_t blocks = op->src[0]->ne[0] / QK8_0;
                        size += alignof(flash_q8_r8_partial) - 1 +
                            sizeof(flash_q8_r8_partial) * blocks * 3 * op->src[1]->ne[1];
                    }
                    return true;''')
    mat_start = body.index('    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {')
    mat_end = body.index('    void forward_mul_mat_id(', mat_start)
    mat = body[mat_start:mat_end]
    marker = '        // INFO: Quantization is done in planes to avoid extra complexity in chunking.'
    assert mat.count(marker) == 1
    candidate = '''        if (!params->use_ref && ordered_k_eligible(nth, op)) {
            const int blocks = ne00 / QK8_0;
            const int first = ith * blocks / nth, last = (ith + 1) * blocks / nth;
            const uintptr_t begin = GGML_PAD((uintptr_t) wdata + nbw2, alignof(flash_q8_r8_partial));
            auto * partial = (flash_q8_r8_partial *) begin;
            GGML_ASSERT(begin + sizeof(*partial) * blocks * 3 * ne11 <= (uintptr_t) wdata + params->wsize);
            for (int row = 0; row < ne11; ++row) {
                if (last > first) from_float(
                    (const float *) ((const char *) src1->data + row * nb11 + first * QK8_0 * sizeof(float)),
                    wdata + row * nbw1 + first * sizeof(block_q8_0), (last - first) * QK8_0);
            }
            flash_q8_r8_prepare(ne00, 24, ne11, (const block_q8_0x8 *) src0->data,
                                (const block_q8_0 *) wdata, first, last, partial);
            ggml_barrier(params->threadpool);
            flash_q8_r8_finish(ne00, 24, ne11, partial, (float *) dst->data, nb1 / sizeof(float), ith, nth);
            if (ith == 0 && flash_q8_r8_ordered_k_audit()) {
                flash_q8_r8_ordered_k_calls.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

'''
    mat = mat.replace(marker, candidate + marker)
    body = body[:mat_start] + mat + body[mat_end:]
    return source[:start] + body + source[end:]


def transform_fixture(source):
    source = source.replace('ggml_cpu_rms_matmul_fusion_count', 'ggml_cpu_q8_r8_ordered_k_count')
    source = source.replace('GGML_CPU_RMS_MATMUL_AUDIT', 'GGML_CPU_Q8_R8_ORDERED_K_AUDIT')
    first = source.index('    const bool expected_fused =')
    last = source.index('    bool good = true;', first)
    source = source[:first] + '''    const bool expected_fused = counter && enabled("GGML_CPU_Q8_R8_ORDERED_K") &&
        !c.reference && c.threads > 1 && c.rows == 24 && c.tokens <= 3 && c.planes == 1 &&
        c.packed && c.type == GGML_TYPE_Q8_0 && c.k >= 4096 && c.k <= 32768;
''' + source[last:]
    marker = '        if (std::fclose(dump)) ++failures;'
    assert source.count(marker) == 1
    source = source.replace(marker, '''        for (bool weighted : {false,true}) {
            test_case c; c.k=1024; c.weighted=weighted;
            failures += !run_case(c,dump,false,cases++);
        }
        for (int k : {16416,32768}) for (int tokens : {1,3}) {
            test_case c; c.k=k; c.tokens=tokens;
            failures += !run_case(c,dump,false,cases++);
        }
        for (int variant : {0,1}) {
            test_case c; c.k=variant ? 65536 : 16384; c.rows=variant ? 24 : 40;
            failures += !run_case(c,dump,false,cases++);
        }
''' + marker)
    marker = 'int main(int argc, char ** argv) {'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\\n",runtime.dli_fname);
''')
    return source
