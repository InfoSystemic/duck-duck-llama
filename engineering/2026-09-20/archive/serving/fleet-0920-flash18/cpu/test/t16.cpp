#include <immintrin.h>
#include <cstdio>
#include <cstring>
// out[j] = column j of the 16x16 block whose rows are r[0..15]
static inline void transpose16(const __m512 r[16], __m512 out[16]) {
    __m512 t[16], u[16], v[16];
    for (int i = 0; i < 8; ++i) { t[2*i] = _mm512_unpacklo_ps(r[2*i], r[2*i+1]); t[2*i+1] = _mm512_unpackhi_ps(r[2*i], r[2*i+1]); }
    for (int i = 0; i < 4; ++i) {
        u[4*i+0] = _mm512_shuffle_ps(t[4*i+0], t[4*i+2], 0x44);
        u[4*i+1] = _mm512_shuffle_ps(t[4*i+0], t[4*i+2], 0xEE);
        u[4*i+2] = _mm512_shuffle_ps(t[4*i+1], t[4*i+3], 0x44);
        u[4*i+3] = _mm512_shuffle_ps(t[4*i+1], t[4*i+3], 0xEE);
    }
    for (int i = 0; i < 4; ++i) {
        v[i]      = _mm512_shuffle_f32x4(u[i],   u[i+4],  0x88);
        v[i+4]    = _mm512_shuffle_f32x4(u[i],   u[i+4],  0xDD);
        v[i+8]    = _mm512_shuffle_f32x4(u[i+8], u[i+12], 0x88);
        v[i+12]   = _mm512_shuffle_f32x4(u[i+8], u[i+12], 0xDD);
    }
    for (int i = 0; i < 4; ++i) {
        out[i]    = _mm512_shuffle_f32x4(v[i],   v[i+8],  0x88);
        out[i+4]  = _mm512_shuffle_f32x4(v[i+4], v[i+12], 0x88);
        out[i+8]  = _mm512_shuffle_f32x4(v[i],   v[i+8],  0xDD);
        out[i+12] = _mm512_shuffle_f32x4(v[i+4], v[i+12], 0xDD);
    }
}
int main() {
    float a[16][16], b[16][16]; __m512 r[16], o[16];
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) a[i][j] = i*100 + j;
    for (int i = 0; i < 16; ++i) r[i] = _mm512_loadu_ps(a[i]);
    transpose16(r, o);
    for (int j = 0; j < 16; ++j) _mm512_storeu_ps(b[j], o[j]);
    int bad = 0;
    for (int j = 0; j < 16; ++j) for (int i = 0; i < 16; ++i) if (b[j][i] != a[i][j]) { if (bad < 8) printf("b[%d][%d]=%g want %g\n", j, i, b[j][i], a[i][j]); bad++; }
    printf("bad=%d\n", bad);
    return bad != 0;
}
