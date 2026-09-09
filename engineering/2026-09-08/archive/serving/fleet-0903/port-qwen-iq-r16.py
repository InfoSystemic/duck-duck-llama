#!/usr/bin/env python3
from pathlib import Path
import difflib
import shutil

root=Path(__file__).resolve().parents[2]
source=(root/'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp').read_text()
p=root/'engines/llama.cpp-q4e-goal-0904/ggml/src/ggml-cpu/repack.cpp'
backup=p.with_name(p.name+'.before-goal-iq-r16')
assert not backup.exists()
shutil.copy2(p,backup)
s=p.read_text()
begin=source.index('// Four codes per output channel')
end=source.index('static void qK_r8_get_scale_min(',begin)
body=source[begin:end].replace('make_block_iq2_r8(rows)', 'make_block_iq2_xs_r8(rows)')
anchor='static void qK_r8_get_scale_min('
assert s.count(anchor)==1
s=s.replace(anchor,body+anchor)
for name in ['iq2_xs','iq3_xxs']:
    for function,params in [('repack',''),('gemv',', GGML_TYPE_Q8_K'),('gemm',', GGML_TYPE_Q8_K')]:
        begin=source.index(f'{function}<block_{name}, 8, 16{params}>')
        begin=source.rindex('template <>',0,begin)
        end=source.index('\n}\n',begin)+3
        old=s.index(f'{function}<block_{name}, 8, 8{params}>')
        old=s.rindex('template <>',0,old)
        s=s[:old]+source[begin:end]+'\n'+s[old:]
    anchor=f'std::is_same_v<BLOC_TYPE, block_{name}> && INTER_SIZE == 8 && NB_COLS == 8;'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor.replace('NB_COLS == 8','(NB_COLS == 8 || NB_COLS == 16)'))
    anchor=f'    static const ggml::cpu::repack::tensor_traits<block_{name}, 8, 8, GGML_TYPE_Q8_K> {name}_r8_q8_K;'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'\n'+anchor.replace('8, 8,','8, 16,').replace('_r8_q8_K;','_r16_q8_K;'))
    anchor=f'            return &{name}_r8_q8_K;'
    s=s.replace(anchor,f'            if (iq_r16_enabled && cur->ne[1] % 16 == 0) {{\n                return &{name}_r16_q8_K;\n            }}\n'+anchor)
anchor='if constexpr (expanded_iq2_xs) {\n            GGML_ASSERT'
assert s.count(anchor)==1
s=s.replace(anchor,'if constexpr (expanded_iq && NB_COLS == 16) {\n            return (t->ne[0] / QK_K) * sizeof(block_iq_r16);\n        } else '+anchor)
anchor='static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {'
s=s.replace(anchor,anchor+'''
    static const bool iq_r16_enabled = []() {
        const char * value = getenv("GGML_CPU_IQ_R16_REPACK");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
''')
p.write_text(s)
patch=''.join(difflib.unified_diff(backup.read_text().splitlines(True),s.splitlines(True),fromfile='a/ggml/src/ggml-cpu/repack.cpp',tofile='b/ggml/src/ggml-cpu/repack.cpp'))
(root/'serving/fleet-0903/q4e-iq-r16.patch').write_text(patch)
print('Ported IQ2_XS and IQ3_XXS r16 to Qwen')
