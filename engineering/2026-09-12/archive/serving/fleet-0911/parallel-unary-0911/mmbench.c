// Isolate one ggml mul_mat shape and time it vs thread count, using the PRODUCTION libraries.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double bench(enum ggml_type wt, int64_t k, int64_t n_out, int64_t n_tok, int nth, double secs, double *bytes_out){
    size_t need = (size_t)3<<30;
    void * buf = malloc(need);
    struct ggml_init_params ip = { need, buf, false };
    struct ggml_context * ctx = ggml_init(ip);
    // Rotate through enough distinct weight copies to exceed L3 (22 MiB/socket), so this
    // measures STREAMING from DRAM like the model does, not a cache-resident replay.
    struct ggml_tensor * a0 = ggml_new_tensor_2d(ctx, wt, k, n_out);
    const int NC = (int)((192u<<20) / (ggml_nbytes(a0) ? ggml_nbytes(a0) : 1)) + 1;
    struct ggml_tensor ** as = malloc(sizeof(void*)*NC);
    struct ggml_cgraph ** gs = malloc(sizeof(void*)*NC);
    as[0]=a0;
    for (int ci=1; ci<NC; ci++) as[ci] = ggml_new_tensor_2d(ctx, wt, k, n_out);
    for (int ci=0; ci<NC; ci++) memset(as[ci]->data, 0x11+ci, ggml_nbytes(as[ci]));
    struct ggml_tensor * a = as[0];
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_tok);
    float * bd = (float*)b->data; for (int64_t i=0;i<k*n_tok;i++) bd[i] = 0.01f*(i&7);
    for (int ci=0; ci<NC; ci++) {
        struct ggml_tensor * cc = ggml_mul_mat(ctx, as[ci], b);
        gs[ci] = ggml_new_graph(ctx);
        ggml_build_forward_expand(gs[ci], cc);
    }
    struct ggml_cgraph * gf = gs[0];
    ggml_graph_compute_with_ctx(ctx, gf, nth);          // warm
    double t0=now(); long it=0;
    while (now()-t0 < secs) { ggml_graph_compute_with_ctx(ctx, gs[it % NC], nth); it++; }
    double dt = now()-t0;
    *bytes_out = (double)ggml_nbytes(a);
    double us = 1e6*dt/it;
    ggml_free(ctx); free(buf);
    return us;
}
int main(void){
    struct { const char*name; enum ggml_type t; int64_t k, n_out, n_tok; } cases[] = {
        {"k=16384 n_out=24  n_tok=3  ", GGML_TYPE_Q8_0, 16384,   24, 3},
        {"k= 8192 n_out=24  n_tok=3  ", GGML_TYPE_Q8_0,  8192,   24, 3},
        {"k= 4096 n_out=24  n_tok=3  ", GGML_TYPE_Q8_0,  4096,   24, 3},
        {"k= 1024 n_out=24  n_tok=3  ", GGML_TYPE_Q8_0,  1024,   24, 3},
        {"k=16384 n_out=24  n_tok=1  ", GGML_TYPE_Q8_0, 16384,   24, 1},
        {"k=16384 n_out=2048 n_tok=3 ", GGML_TYPE_Q8_0, 16384, 2048, 3},
    };
    printf("%-30s %5s %10s %12s\n","case","nth","us/op","GB/s(weights)");
    for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
        for (int nth=1; nth<=15; nth+= (nth==1?14:1)) {
            double bytes=0; double us = bench(cases[i].t, cases[i].k, cases[i].n_out, cases[i].n_tok, nth, 1.2, &bytes);
            printf("%-30s %5d %10.1f %12.1f\n", cases[i].name, nth, us, bytes/(us*1e-6)/1e9);
        }
    }
    return 0;
}
