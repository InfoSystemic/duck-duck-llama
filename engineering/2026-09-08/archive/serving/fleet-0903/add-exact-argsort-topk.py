#!/usr/bin/env python3
from pathlib import Path
import difflib
import shutil

root = Path(__file__).resolve().parents[2]
for engine_name, prefix in [('llama.cpp-q4e-goal-0904', 'q4e'), ('llama.cpp-glm5n-goal-0904', 'glm')]:
    engine = root / 'engines' / engine_name
    patches = []
    for relative in ['ggml/src/ggml.c', 'ggml/src/ggml-cpu/ops.cpp']:
        path = engine / relative
        backup = path.with_name(path.name + '.before-goal-exact-argsort-topk')
        assert not backup.exists()
        old = path.read_text()
        text = old
        if relative.endswith('ggml.c'):
            start = text.index('struct ggml_tensor * ggml_argsort_top_k(')
            marker = '    struct ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);'
            assert marker in text[start:]
            text = text[:start] + text[start:].replace(marker, marker + '\n    ggml_set_op_params_i32(result, 1, k);', 1)
        else:
            marker = 'static void ggml_compute_forward_argsort_f32('
            helper = '''static bool argsort_top_k_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_ARGSORT_TOP_K");
        return value && atoi(value) != 0;
    }();
    return enabled;
}

static void argsort_descending(const float * data, int32_t * indices, int64_t n, int64_t k) {
    const auto cmp = cmp_argsort<GGML_SORT_ORDER_DESC>{data};
    if (argsort_top_k_enabled() && k > 0 && k < n &&
            std::all_of(data, data + n, [](float value) { return std::isfinite(value); })) {
        std::partial_sort(indices, indices + k, indices + n, cmp);
        bool tied = false;
        for (int64_t i = 1; i < k; ++i) {
            tied |= data[indices[i - 1]] == data[indices[i]];
        }
        int boundary_count = 0;
        for (int64_t i = 0; i < n; ++i) {
            boundary_count += data[i] == data[indices[k - 1]];
        }
        if (!tied && boundary_count == 1) {
            return;
        }
        // Keep the full sort's original tie ordering.
        for (int64_t i = 0; i < n; ++i) {
            indices[i] = i;
        }
    }
    std::sort(indices, indices + n, cmp);
}

'''
            assert marker in text
            text = text.replace(marker, helper + marker, 1)
            marker = 'std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});'
            assert text.count(marker) == 1
            text = text.replace(marker, 'argsort_descending(src_data, dst_data, ne0, ggml_get_op_params_i32(dst, 1));', 1)
        shutil.copy2(path, backup)
        path.write_text(text)
        patches += difflib.unified_diff(old.splitlines(True), text.splitlines(True),
            fromfile='a/' + relative, tofile='b/' + relative)
    Path(__file__).with_name(prefix + '-exact-argsort-topk.patch').write_text(''.join(patches))
