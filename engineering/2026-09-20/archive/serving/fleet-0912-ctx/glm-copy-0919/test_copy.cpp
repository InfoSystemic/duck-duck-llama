#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

struct AlignedBytes {
    unsigned char * ptr=nullptr; size_t n;
    explicit AlignedBytes(size_t count): n(count) { if(posix_memalign((void **)&ptr,64,n))std::abort(); }
    ~AlignedBytes() { std::free(ptr); }
    unsigned char * data() { return ptr; }
    unsigned char * begin() { return ptr; }
    unsigned char * end() { return ptr+n; }
    size_t size() const { return n; }
};
struct Case { int64_t n0,n1,n2,n3; ggml_type type; int mode; };
static uint32_t rng=0x136af483;
static uint32_t random_bits() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static void fail(const char * msg) { std::fprintf(stderr,"%s\n",msg); std::exit(1); }
static size_t span(ggml_tensor * t) { size_t n=ggml_type_size(t->type); for(int i=0;i<4;++i)n+=(t->ne[i]-1)*t->nb[i]; return n; }

int main(int argc,char ** argv) {
    if(argc<4 || argc>5) fail("usage: test_copy OUTPUT EXPECT_FLAT THREADS [bench]");
    const bool expected=std::atoi(argv[2]);
    const int threads=std::atoi(argv[3]);
    const bool bench=argc==5;
    FILE * out=std::fopen(argv[1],"wb"); if(!out) fail("open output");
    Dl_info cpu_info{},base_info{};
    if(!dladdr((void *)ggml_backend_cpu_init,&cpu_info) || !dladdr((void *)ggml_init,&base_info)) fail("dladdr");
    std::fprintf(stderr,"CPU_LIBRARY %s\nBASE_LIBRARY %s\n",cpu_info.dli_fname,base_info.dli_fname);
    auto count=reinterpret_cast<uint64_t (*)(void)>(dlsym(RTLD_DEFAULT,"ggml_cpu_cpy_flat_count"));
    auto backend=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend,threads);
    std::vector<Case> cases={
        {262144,1,2,1,GGML_TYPE_F32,0}, // Actual production stride/layout.
        {262144,1,1,1,GGML_TYPE_F32,0},
        {262145,1,3,1,GGML_TYPE_F32,0}, // Row/chunk tails.
        {4097,3,2,2,GGML_TYPE_F32,1},  // Padded higher dimensions.
        {32769,1,2,1,GGML_TYPE_F16,1},
        {8193,2,2,1,GGML_TYPE_I32,1},
        {8192,2,1,1,GGML_TYPE_F32,0}, // Exact 64 KiB threshold.
        {8191,2,1,1,GGML_TYPE_F32,0}, // Below threshold: fall back.
        {262144,1,2,1,GGML_TYPE_F32,2}, // Same-address alias: fall back.
        {16384,3,1,1,GGML_TYPE_F32,3}, // Overlapping source rows: fall back.
        {16384,2,2,1,GGML_TYPE_F32,4}, // Nonpacked dim0: fall back.
        {32768,1,2,1,GGML_TYPE_F32,5}, // Different types: fall back.
        {32768,1,2,1,GGML_TYPE_F32,6}, // Ordinary name: fall back.
        {32768,1,2,1,GGML_TYPE_F32,7}, // Reshape copy: fall back.
    };
    if(bench) cases.resize(1);
    for(size_t ci=0;ci<cases.size();++ci) {
        auto c=cases[ci];
        auto ctx=ggml_init({1024*1024,nullptr,true});
        auto src=ggml_new_tensor_4d(ctx,c.type,c.n0,c.n1,c.n2,c.n3);
        auto dst=ggml_new_tensor_4d(ctx,c.mode==5?GGML_TYPE_F16:c.type,c.n0,c.n1,c.n2,c.n3);
        const size_t es=ggml_type_size(src->type),de=ggml_type_size(dst->type);
        if(c.mode==7) { dst->ne[0]=c.n0/2; dst->ne[1]=2; }
        src->nb[0]=es*(c.mode==4?2:1);
        src->nb[1]=src->nb[0]*src->ne[0]+(c.mode==1?2*es:0);
        if(c.mode==3)src->nb[1]/=2;
        src->nb[2]=src->nb[1]*src->ne[1]+(c.mode==1?3*es:0);
        src->nb[3]=src->nb[2]*src->ne[2]+(c.mode==1?5*es:0);
        dst->nb[0]=de;
        dst->nb[1]=de*dst->ne[0]+(c.mode==1?3*de:0);
        dst->nb[2]=2*dst->nb[1]*dst->ne[1]+(c.mode==1?5*de:0);
        dst->nb[3]=dst->nb[2]*dst->ne[2]+(c.mode==1?7*de:0);
        if(c.mode==2) for(int i=0;i<4;++i)dst->nb[i]=src->nb[i];
        const size_t sb=span(src),db=span(dst);
        AlignedBytes src_bytes(sb+256),dst_bytes(db+256);
        auto src_buffer=ggml_backend_cpu_buffer_from_ptr(src_bytes.data(),src_bytes.size());
        auto dst_buffer=ggml_backend_cpu_buffer_from_ptr(dst_bytes.data(),dst_bytes.size());
        src->data=src_bytes.data()+64; src->buffer=src_buffer;
        dst->data=dst_bytes.data()+64; dst->buffer=dst_buffer;
        if(c.mode==2) { dst->data=src->data; dst->buffer=src_buffer; }
        auto copy=ggml_cpy(ctx,src,dst);
        ggml_set_name(copy,c.mode==6?"ordinary-copy":"cache_s_l0 (view) (copy)");
        ggml_set_output(copy);
        auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,copy);
        const uint64_t before=count?count():0;
        const int repeats=bench?100:2;
        std::vector<double> us;
        for(int rep=0;rep<repeats;++rep) {
            std::fill(src_bytes.begin(),src_bytes.end(),0x6b);
            std::fill(dst_bytes.begin(),dst_bytes.end(),0xa5);
            for(int64_t i3=0;i3<src->ne[3];++i3)for(int64_t i2=0;i2<src->ne[2];++i2)
                for(int64_t i1=0;i1<src->ne[1];++i1)for(int64_t i0=0;i0<src->ne[0];++i0) {
                    void * ptr=(char *)src->data+i0*src->nb[0]+i1*src->nb[1]+i2*src->nb[2]+i3*src->nb[3];
                    if(c.type==GGML_TYPE_F32) *(float *)ptr=(int(random_bits()%16385)-8192)/1024.f;
                    else if(c.type==GGML_TYPE_F16) *(ggml_fp16_t *)ptr=ggml_fp32_to_fp16((int(random_bits()%1025)-512)/256.f);
                    else *(int32_t *)ptr=(int32_t)random_bits();
                }
            auto start=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)fail("graph compute");
            auto stop=std::chrono::steady_clock::now();
            if(rep>=10)us.push_back(std::chrono::duration<double,std::micro>(stop-start).count());
            auto & output=c.mode==2?src_bytes:dst_bytes;
            if(!bench || rep+1==repeats)
                if(std::fwrite(output.data(),1,output.size(),out)!=output.size())fail("write output");
        }
        const uint64_t delta=(count?count():0)-before;
        const bool eligible=c.mode<2 && ggml_nelements(src)*(int64_t)es>=65536;
        const uint64_t want=expected && eligible ? repeats : 0;
        if(delta!=want) { std::fprintf(stderr,"case%zu count want=%llu got=%llu\n",ci,(unsigned long long)want,(unsigned long long)delta); return 2; }
        if(bench) { std::sort(us.begin(),us.end()); std::printf("BENCH threads=%d bytes=%lld median_us=%.3f p10=%.3f p90=%.3f\n",threads,(long long)(ggml_nelements(src)*es),us[us.size()/2],us[us.size()/10],us[us.size()*9/10]); }
        std::fprintf(stderr,"case%zu PASS shape=[%lld,%lld,%lld,%lld] type=%s mode=%d count=%llu\n",ci,(long long)c.n0,(long long)c.n1,(long long)c.n2,(long long)c.n3,ggml_type_name(c.type),c.mode,(unsigned long long)delta);
        ggml_backend_buffer_free(src_buffer);ggml_backend_buffer_free(dst_buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend); std::fclose(out); std::printf("PASS %zu cases threads=%d\n",cases.size(),threads);
}
