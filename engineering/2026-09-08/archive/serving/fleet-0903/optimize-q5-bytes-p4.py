#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import difflib

root=Path(__file__).resolve().parents[2]
for engine_name,prefix in [('llama.cpp-glm5n-goal-0904','glm'),('llama.cpp-q4e-goal-0904','q4e')]:
    engine=root/'engines'/engine_name
    path=engine/'ggml/src/ggml-cpu/repack.cpp'
    backup=path.with_name(path.name+'.before-goal-q5-bytes-p4')
    assert not backup.exists()
    shutil.copy2(path,backup)
    text=path.read_text()
    start=text.index('static void ggml_gemv_q5_K_x16_bytes_q8_K(')
    end=text.index('static void ggml_x16_pack_q6_K(',start)
    body=text[start:end]
    body=body.replace('static void ggml_gemv_q5_K_x16_bytes_q8_K(', 'template <int NR>\nstatic void ggml_gemv_q5_K_x16_bytes_impl(')
    body=body.replace('const void * vy, int nr, int nc)', 'const void * vy, int nc)')
    body=re.sub(r'\bnr\b','NR',body)
    body=body.replace('__m512i dots[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};',
                      '__m512i dots[NR][4];\n                for (int y = 0; y < NR; ++y) for (int part = 0; part < 4; ++part) dots[y][part] = _mm512_setzero_si512();')
    body=body.replace('                for (int chunk = 0; chunk < 8; ++chunk) {',
                      '#if defined(__GNUC__)\n#pragma GCC unroll 8\n#endif\n                for (int chunk = 0; chunk < 8; ++chunk) {')
    body=body.replace('dots[y] = _mm512_dpbusd_epi32(dots[y], q, _mm512_set1_epi32(aq));',
                      'dots[y][chunk & 3] = _mm512_dpbusd_epi32(dots[y][chunk & 3], q, _mm512_set1_epi32(aq));')
    body=body.replace('isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dots[y], scale));',
                      'const __m512i dot = _mm512_add_epi32(_mm512_add_epi32(dots[y][0], dots[y][1]), _mm512_add_epi32(dots[y][2], dots[y][3]));\n                    isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dot, scale));')
    body+='''static void ggml_gemv_q5_K_x16_bytes_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(nr == 1 || nr == 2);
    if (nr == 1) {
        ggml_gemv_q5_K_x16_bytes_impl<1>(n, s, bs, vx, vy, nc);
    } else {
        ggml_gemv_q5_K_x16_bytes_impl<2>(n, s, bs, vx, vy, nc);
    }
}

'''
    text=text[:start]+body+text[end:]
    path.write_text(text)
    patch=''.join(difflib.unified_diff(backup.read_text().splitlines(True),text.splitlines(True),fromfile='a/ggml/src/ggml-cpu/repack.cpp',tofile='b/ggml/src/ggml-cpu/repack.cpp'))
    (root/f'serving/fleet-0903/{prefix}-q5-bytes-p4.patch').write_text(patch)
    print('Added four partial sums to',engine_name)
