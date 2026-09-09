"""Move the misplaced FP32 column-copy branch out of quantized gathers."""


def transform(source):
    q_start = source.index('static void ggml_compute_forward_get_rows_q(')
    q_end = source.index('static void ggml_compute_forward_get_rows_f16(', q_start)
    f_start = source.index('static void ggml_compute_forward_get_rows_f32(', q_end)
    f_end = source.index('void ggml_compute_forward_get_rows(', f_start)
    quant, floating = source[q_start:q_end], source[f_start:f_end]
    marker = '    if (ggml_cpu_parallel_copy_enabled() && nc >= 4096) {\n'
    assert quant.count(marker) == 1 and marker not in floating
    begin = quant.index(marker)
    end = quant.index('    // rows per thread\n', begin)
    branch = quant[begin:end]
    assert branch.count('std::memcpy(') == 1 and branch.endswith('        return;\n    }\n\n')
    assert 'dequantize_row_q' not in branch
    anchor = '    const int nth = params->nth;\n\n'
    assert floating.count(anchor) == 1
    corrected_quant = quant[:begin] + quant[end:]
    corrected_float = floating.replace(anchor, anchor + branch)
    assert marker not in corrected_quant and corrected_float.count(marker) == 1
    return source[:q_start] + corrected_quant + source[q_end:f_start] + corrected_float + source[f_end:]
