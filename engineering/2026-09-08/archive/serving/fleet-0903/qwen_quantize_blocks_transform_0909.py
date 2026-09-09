"""Share complete activation blocks across the existing x16 worker pool."""


def transform(source):
    marker = '#include "repack.h"'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n#include "qwen-quantize-blocks-0909.h"')
    marker = 'class tensor_traits_x16 : public tensor_traits_base {'
    assert source.count(marker) == 1
    source = source.replace(marker, '''static std::atomic<uint64_t> qwen_quantize_block_counts[3]{};

extern "C" {
GGML_BACKEND_API uint64_t ggml_cpu_qwen_quantize_blocks_count(int path);
}

extern "C" uint64_t ggml_cpu_qwen_quantize_blocks_count(int path) {
    return path >= 0 && path < 3 ? qwen_quantize_block_counts[path].load(std::memory_order_relaxed) : 0;
}

static bool qwen_quantize_blocks_try(const ggml_compute_params * params, const ggml_tensor * src,
        ggml_type type, char * dst, size_t row_bytes, int path) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_X16_QUANTIZE_BLOCKS");
        return value && strcmp(value, "1") == 0;
    }();
    if (!enabled) return false;
    qwen_quantize_blocks(params, src, type, ggml_get_type_traits_cpu(type)->from_float, dst, row_bytes);
    static const bool audit = [] {
        const char * value = getenv("GGML_CPU_X16_QUANTIZE_BLOCKS_AUDIT");
        return value && strcmp(value, "1") == 0;
    }();
    if (audit && params->ith == 0) qwen_quantize_block_counts[path].fetch_add(1, std::memory_order_relaxed);
    return true;
}

''' + marker)
    marker = '    void quantize_src1(ggml_compute_params * params, const ggml_tensor * src1, char * wdata, size_t row_bytes) {\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '        if (qwen_quantize_blocks_try(params, src1, sp.param_type, wdata, row_bytes, 0)) return;\n')
    marker = '        // quantize src1 (all planes)\n        {\n'
    assert source.count(marker) == 1
    source = source.replace(marker, '        // quantize src1 (all planes)\n        if (!qwen_quantize_blocks_try(params, src1, sp.param_type, wdata, row_bytes, 1)) {\n')
    original = '''        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(sp.param_type)->from_float;
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t sr = ith; sr < n_src_rows; sr += nth) {
                from_float((const float *) ((const char *) src1->data + token * src1->nb[2] + sr * src1->nb[1]),
                           wdata + token * plane_bytes + sr * row_bytes, k);
            }
        }
'''
    assert source.count(original) == 1
    changed = '        if (!qwen_quantize_blocks_try(params, src1, sp.param_type, wdata, row_bytes, 2)) {\n'
    changed += ''.join('    ' + line for line in original.splitlines(True)) + '        }\n'
    return source.replace(original, changed)
