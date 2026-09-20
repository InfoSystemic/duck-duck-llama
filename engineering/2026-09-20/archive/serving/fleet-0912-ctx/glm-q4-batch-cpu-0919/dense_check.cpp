#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"
#include "ggml-quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <sched.h>
#include <string>
#include <vector>

static void require(bool value, const char * what) {
    if (!value) { std::fprintf(stderr,"CHECK FAILED: %s\n",what); std::abort(); }
}
static uint64_t hash_bytes(const void * ptr,size_t size) {
    uint64_t h=14695981039346656037ULL;
    auto p=static_cast<const uint8_t *>(ptr);
    for(size_t i=0;i<size;++i)h=(h^p[i])*1099511628211ULL;
    return h;
}
using counter_fn=uint64_t (*)(int);
struct Weights {
    ggml_context *ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor *gate,*up;
    ggml_type type;
    int k,rows,experts;
    uint64_t digest;
    Weights(ggml_type qt,int nk,int nr,int ne):type(qt),k(nk),rows(nr),experts(ne) {
        ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});require(ctx,"weight context");
        gate=ggml_new_tensor_3d(ctx,type,k,rows,experts);
        up=ggml_new_tensor_3d(ctx,type,k,rows,experts);
        ggml_set_name(gate,"blk.0.ffn_gate_exps.weight");
        ggml_set_name(up,"blk.0.ffn_up_exps.weight");
        buf=ggml_backend_alloc_ctx_tensors_from_buft(ctx,ggml_backend_cpu_repack_buffer_type());
        require(buf&&gate->extra&&up->extra,"packed weight traits");
        // Every quant block is produced by the reference quantizer from finite floats.
        // A fixed palette keeps full 288-expert setup bounded without allocating F32 weights.
        const size_t bs=ggml_type_size(type),qb=ggml_blck_size(type),np=4096;
        std::vector<uint8_t> palette(np*bs),raw(ggml_nbytes(gate));
        std::mt19937 gen(72491+int(type));std::normal_distribution<float> dist(0.f,0.02f);
        std::vector<float> f(qb);
        for(size_t i=0;i<np;++i) {
            for(float &v:f)v=dist(gen);
            void *p=palette.data()+i*bs;
            if(type==GGML_TYPE_Q4_K)quantize_row_q4_K_ref(f.data(),static_cast<block_q4_K *>(p),qb);
            else if(type==GGML_TYPE_Q5_K)quantize_row_q5_K_ref(f.data(),static_cast<block_q5_K *>(p),qb);
            else if(type==GGML_TYPE_Q6_K)quantize_row_q6_K_ref(f.data(),static_cast<block_q6_K *>(p),qb);
            else require(false,"supported test quant type");
        }
        digest=0;
        for(auto w:{gate,up}) {
            for(size_t i=0;i<raw.size()/bs;++i)std::memcpy(raw.data()+i*bs,palette.data()+(gen()%np)*bs,bs);
            digest^=hash_bytes(raw.data(),raw.size());
            ggml_backend_tensor_set(w,raw.data(),0,raw.size());
        }
    }
    ~Weights(){ggml_backend_buffer_free(buf);ggml_free(ctx);}
};

static void dense(ggml_backend_t backend,Weights &w,int threads,int tokens,int broadcast,
                  const std::string &mode,const std::string &out_dir,counter_fn counter) {
    const int planes=w.experts*broadcast,repeats=3;
    auto ic=ggml_init({ggml_tensor_overhead()*6,nullptr,true});
    auto gc=ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead(),nullptr,true});
    auto storage=ggml_new_tensor_3d(ic,GGML_TYPE_F32,w.k+(mode=="padded"?32:0),tokens,planes);
    auto x=mode=="padded"?ggml_view_3d(ic,storage,w.k,tokens,planes,storage->nb[1],storage->nb[2],0):storage;
    auto ib=ggml_backend_alloc_ctx_tensors_from_buft(ic,ggml_backend_cpu_buffer_type());require(ib,"input buffer");
    ggml_set_input(x);
    auto gate=ggml_mul_mat(gc,w.gate,x);auto y=gate;
    if(mode=="fused")y=ggml_swiglu_split(gc,gate,ggml_mul_mat(gc,w.up,x));
    ggml_set_output(y);
    auto graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,y);
    auto alloc=ggml_gallocr_new(ggml_backend_cpu_buffer_type());require(ggml_gallocr_alloc_graph(alloc,graph),"allocation");
    std::vector<float> values(ggml_nelements(storage),-1234.f);
    for(int p=0;p<planes;++p)for(int t=0;t<tokens;++t)for(int i=0;i<w.k;++i)
        values[(p*tokens+t)*storage->ne[0]+i]=std::sin((i+t*71+p*17)*0.0123f)+0.3f*std::cos((i+t*11+p*37)*0.079f);
    ggml_backend_tensor_set(storage,values.data(),0,ggml_nbytes(storage));
    auto count=[&]{return counter?counter(2)+counter(3):0;};
    uint64_t before=count();
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"warm compute");
    std::vector<float> expected(ggml_nelements(y)),actual(expected.size());
    ggml_backend_tensor_get(y,expected.data(),0,ggml_nbytes(y));
    for(int r=0;r<repeats;++r){
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat compute");
        ggml_backend_tensor_get(y,actual.data(),0,ggml_nbytes(y));
        require(std::memcmp(actual.data(),expected.data(),ggml_nbytes(y))==0,"repeat parity");
    }
    for(float v:actual)require(std::isfinite(v),"finite output");
    const std::string key=std::string(ggml_type_name(w.type))+"-"+std::to_string(w.k)+"-"+std::to_string(w.rows)+"-p"+
        std::to_string(w.experts)+"-b"+std::to_string(broadcast)+"-t"+std::to_string(threads)+"-n"+std::to_string(tokens)+"-"+mode;
    std::ofstream file(out_dir+"/"+key+".f32",std::ios::binary);file.write((const char *)actual.data(),ggml_nbytes(y));require(file.good(),"output file");
    std::printf("{\"key\":\"%s\",\"type\":\"%s\",\"k\":%d,\"rows\":%d,\"planes\":%d,\"broadcast\":%d,\"threads\":%d,\"tokens\":%d,\"mode\":\"%s\",\"repeats\":%d,\"batch_calls\":%llu,\"weight_hash\":\"%016llx\",\"output_hash\":\"%016llx\"}\n",key.c_str(),ggml_type_name(w.type),w.k,w.rows,w.experts,broadcast,threads,tokens,mode.c_str(),repeats,(unsigned long long)(count()-before),(unsigned long long)w.digest,(unsigned long long)hash_bytes(actual.data(),ggml_nbytes(y)));
    std::fflush(stdout);ggml_gallocr_free(alloc);ggml_backend_buffer_free(ib);ggml_free(gc);ggml_free(ic);
}
int main(int argc,char **argv){
    require(argc==2,"usage: dense_check OUTPUT_DIR");
    auto backend=ggml_backend_cpu_init();require(backend,"backend");
    auto counter=(counter_fn)dlsym(RTLD_DEFAULT,"ggml_cpu_q4_batch_count");
    Dl_info info{};require(dladdr((void *)ggml_backend_cpu_init,&info),"library");
    std::printf("{\"cpu_library\":\"%s\"}\n",info.dli_fname);std::fflush(stdout);
    cpu_set_t original;require(sched_getaffinity(0,sizeof(original),&original)==0,"affinity");
    std::vector<int> cpus;for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&original))cpus.push_back(c);
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K})for(auto shape:{std::make_pair(256,48),std::make_pair(4096,512)})for(int planes:{1,3}){
        Weights w(type,shape.first,shape.second,planes);
        for(int threads:{1,3,15}){
            require(int(cpus.size())>=threads,"CPU count");auto params=ggml_threadpool_params_default(threads);
            for(int i=0;i<threads;++i){require(cpus[i]<GGML_MAX_N_THREADS,"mask bound");params.cpumask[cpus[i]]=true;}params.strict_cpu=true;
            auto pool=ggml_threadpool_new(&params);require(pool,"pool");ggml_backend_cpu_set_n_threads(backend,threads);ggml_backend_cpu_set_threadpool(backend,pool);
            for(int broadcast:{1,2})for(int tokens:{1,2,3,4,8,32})for(const std::string mode:{"plain","padded","fused"})dense(backend,w,threads,tokens,broadcast,mode,argv[1],counter);
            ggml_backend_cpu_set_threadpool(backend,nullptr);ggml_threadpool_free(pool);require(sched_setaffinity(0,sizeof(original),&original)==0,"restore mask");
        }
    }
    ggml_backend_free(backend);
}
