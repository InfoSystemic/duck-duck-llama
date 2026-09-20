#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <immintrin.h>
static void check(bool ok,const char *why){if(!ok){std::fprintf(stderr,"FAILED %s\n",why);std::abort();}}
static uint32_t random_state=0x8392f513;
static uint32_t random_bits(){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return random_state;}
static float random_float(){return (int(random_bits()%8193)-4096)/1024.f;}
static int mode_fd=-1;
static uint64_t computes=0,records=0,values=0;
static void compute(ggml_backend_t backend,ggml_cgraph *graph){
    if(mode_fd>=0){uint32_t mode=uint32_t(computes&1);check(pwrite(mode_fd,&mode,4,0)==4,"mode write");}
    ++computes;check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute");
}
struct State {
    ggml_backend_t backend;
    ggml_context *weights{},*inputs{},*ctx{};
    ggml_backend_buffer_t wb{},ib{};
    ggml_gallocr_t ga{};
    ggml_tensor *ape{},*kg{},*kg_store{},*cells{},*cell_store{},*product{},*final{};
    ggml_cgraph *graph{};
    int max_pools,streams,cell_count,pools;
    std::vector<unsigned char> original_kg,original_cells,original_ape;
    State(ggml_backend_t b,int maxp,int ns,int pad,bool extremes=false):backend(b),max_pools(maxp),streams(ns),cell_count(4*maxp+11) {
        weights=ggml_init({ggml_tensor_overhead()*8,nullptr,true});inputs=ggml_init({ggml_tensor_overhead()*8,nullptr,true});
        ape=ggml_new_tensor_2d(weights,GGML_TYPE_F32,128,4);wb=ggml_backend_alloc_ctx_tensors(weights,backend);check(wb,"weights buffer");ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        kg_store=ggml_new_tensor_3d(inputs,GGML_TYPE_F16,256+pad,cell_count,streams);
        kg=ggml_view_3d(inputs,kg_store,256,cell_count,streams,kg_store->nb[1],kg_store->nb[2],0);
        cell_store=ggml_new_tensor_2d(inputs,GGML_TYPE_I32,4*maxp+pad,streams);
        cells=ggml_view_2d(inputs,cell_store,4*maxp,streams,cell_store->nb[1],0);
        ib=ggml_backend_alloc_ctx_tensors(inputs,backend);check(ib,"inputs buffer");
        std::memset(kg_store->data,0,ggml_nbytes(kg_store));std::memset(cell_store->data,0,ggml_nbytes(cell_store));
        for(int s=0;s<streams;++s)for(int i=0;i<cell_count;++i)for(int c=0;c<256;++c){
            auto *v=(ggml_fp16_t *)((char *)kg->data+s*kg->nb[2]+i*kg->nb[1]);
            if(extremes){uint16_t x=uint16_t(random_bits());if((x&0x7c00)==0x7c00)x^=0x0400;v[c]=x;}
            else v[c]=ggml_fp32_to_fp16(random_float());
        }
        for(int s=0;s<streams;++s)for(int p=0;p<4*maxp;++p)map(s,p)=p;
        for(int m=0;m<4;++m)for(int c=0;c<128;++c)position(m,c)=random_float()/8.f;
        auto copy=[](ggml_tensor *t){auto *p=(unsigned char *)t->data;return std::vector<unsigned char>(p,p+ggml_nbytes(t));};
        original_kg=copy(kg_store);original_cells=copy(cell_store);original_ape=copy(ape);rebuild(maxp);
    }
    int32_t &map(int s,int i){return *(int32_t *)((char *)cells->data+s*cells->nb[1]+i*4);}
    ggml_fp16_t &value(int s,int cell,int channel){return *(ggml_fp16_t *)((char *)kg->data+s*kg->nb[2]+cell*kg->nb[1]+channel*2);}
    float &position(int m,int c){return *(float *)((char *)ape->data+m*ape->nb[1]+c*4);}
    void clear_graph(){if(ga)ggml_gallocr_free(ga);if(ctx)ggml_free(ctx);ga=nullptr;ctx=nullptr;}
    void rebuild(int count){
        check(count>0&&count<=max_pools,"pool shape");clear_graph();pools=count;cells->ne[0]=4*count;
        ctx=ggml_init({1024*1024,nullptr,true});check(ctx,"graph context");
        auto members=ggml_get_rows(ctx,kg,cells);ggml_set_name(members,"indexer_pool_members-cache-test");
        auto k=ggml_view_4d(ctx,members,128,4,pools,streams,members->nb[1],4*members->nb[1],members->nb[2],0);
        auto g=ggml_view_4d(ctx,members,128,4,pools,streams,members->nb[1],4*members->nb[1],members->nb[2],128*sizeof(float));
        k=ggml_cont(ctx,ggml_permute(ctx,k,1,0,2,3));g=ggml_cont(ctx,ggml_permute(ctx,g,1,0,2,3));
        auto ac=ggml_cont(ctx,ggml_transpose(ctx,ape));g=ggml_add(ctx,g,ggml_reshape_4d(ctx,ac,4,128,1,1));
        auto probs=ggml_soft_max(ctx,g);product=ggml_mul(ctx,k,probs);auto sum=ggml_sum_rows(ctx,product);final=ggml_reshape_4d(ctx,sum,128,pools,1,streams);ggml_set_output(final);
        graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,final);ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type());check(ggml_gallocr_alloc_graph(ga,graph),"graph allocation");
    }
    void restore(){std::memcpy(kg_store->data,original_kg.data(),original_kg.size());std::memcpy(cell_store->data,original_cells.data(),original_cells.size());std::memcpy(ape->data,original_ape.data(),original_ape.size());}
    void emit(FILE *out,int fixture,int action){
        const size_t bytes=ggml_nbytes(final);compute(backend,graph);std::vector<unsigned char> first(bytes);std::memcpy(first.data(),final->data,bytes);
        std::memset(final->data,0xa5,bytes);compute(backend,graph);check(std::memcmp(first.data(),final->data,bytes)==0,"repeat/flag exact bytes");
        uint64_t header[]={uint64_t(fixture),uint64_t(action),uint64_t(pools),uint64_t(streams),bytes};check(std::fwrite(header,1,sizeof(header),out)==sizeof(header),"record header");check(std::fwrite(final->data,1,bytes,out)==bytes,"output bytes");++records;values+=bytes/4;
    }
    ~State(){clear_graph();ggml_backend_buffer_free(ib);ggml_backend_buffer_free(wb);ggml_free(inputs);ggml_free(weights);}
};
int main(int argc,char **argv){
    check(argc==3,"usage: test_cache OUTPUT WORKERS");int threads=std::atoi(argv[2]);check(threads>0&&threads<=15,"workers");
    Dl_info ci{};check(dladdr((void *)ggml_backend_cpu_init,&ci),"library identity");std::printf("{\"event\":\"library\",\"path\":\"%s\"}\n",ci.dli_fname);
    auto stat=(uint64_t(*)(int))dlsym(RTLD_DEFAULT,"ggml_cpu_glm_pool_cache_stat");
    const char *mode=std::getenv("POOL_CACHE_TEST_SWITCH_FILE");if(mode){mode_fd=open(mode,O_WRONLY|O_CLOEXEC);check(mode_fd>=0,"switch control");}
    auto backend=ggml_backend_cpu_init();check(backend,"backend");ggml_backend_cpu_set_n_threads(backend,threads);
    auto tp=ggml_threadpool_params_default(threads);auto pool=ggml_threadpool_new(&tp);check(pool,"threadpool");ggml_backend_cpu_set_threadpool(backend,pool);
    FILE *out=std::fopen(argv[1],"wb");check(out,"output file");
    struct Shape{int pools,streams,pad;bool extremes;};
    const Shape shapes[]={{73,1,0,false},{73,3,7,true},{1026,1,0,true},{8194,1,0,false},{25002,1,7,true}};
    for(int fi=0;fi<5;++fi){
        const auto shape=shapes[fi];State s(backend,shape.pools,shape.streams,shape.pad,shape.extremes);int action=0;
        auto emit=[&]{s.emit(out,fi,action++);};emit();emit();
        for(int ch:{0,127,128,255}){s.value(0,s.map(0,0),ch)^=1;emit();s.value(0,s.map(0,0),ch)^=1;emit();}
        s.position(3,127)+=0.25f;emit();s.restore();emit();
        s.map(0,3)=s.cell_count-1;emit();s.restore();emit();
        for(int st=0;st<s.streams;++st)for(int i=0;i<4*s.pools;++i)s.map(st,i)=(i+17)%s.cell_count;emit();
        s.restore();emit();
        if(shape.pools<=1026){
            for(int st=0;st<s.streams;++st)for(int m=0;m<4;++m){s.value(st,s.map(st,m),m*63)^=1;emit();}
            s.restore();emit();
            for(int st=0;st<s.streams;++st)for(int i=0;i<4*s.pools;++i)s.map(st,i)=0;emit();s.restore();emit();
            std::swap(s.product->src[0],s.product->src[1]);emit();std::swap(s.product->src[0],s.product->src[1]);emit();
            for(int count:{shape.pools-6,shape.pools-2,shape.pools,shape.pools/2,shape.pools}){s.rebuild(count);emit();}
            const unsigned mxcsr=_mm_getcsr();_mm_setcsr(mxcsr|0x8040u);emit();_mm_setcsr(mxcsr);emit();
            const int rounding=std::fegetround();check(std::fesetround(FE_DOWNWARD)==0,"round down");emit();check(std::fesetround(rounding)==0,"round restore");emit();
            for(int st=0;st<s.streams;++st)for(int i=0;i<s.cell_count;++i)for(int ch=0;ch<256;++ch)s.value(st,i,ch)=ggml_fp32_to_fp16(random_float());emit();s.restore();emit();
        }
        std::printf("{\"event\":\"fixture\",\"id\":%d,\"actions\":%d}\n",fi,action);std::fflush(stdout);
    }
    {
        std::vector<std::unique_ptr<State>> live;
        for(int i=0;i<18;++i)live.emplace_back(new State(backend,257,1,i%3,false));
        for(int pass=0;pass<2;++pass)for(int i=0;i<18;++i)live[i]->emit(out,100+i,pass);
    }
    check(std::fclose(out)==0,"output close");if(mode_fd>=0)check(close(mode_fd)==0,"control close");
    std::printf("{\"event\":\"done\",\"passed\":true,\"records\":%llu,\"values\":%llu,\"computes\":%llu,\"stats\":[",(unsigned long long)records,(unsigned long long)values,(unsigned long long)computes);
    for(int i=0;i<7;++i)std::printf("%s%llu",i?",":"",(unsigned long long)(stat?stat(i):0));std::printf("]}\n");
    ggml_backend_free(backend);ggml_threadpool_free(pool);
}
