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

static void run(ggml_backend_t backend,Weights &w,int threads,int tokens,const std::string &mode,
                int pattern,int repeats,const std::string &out_dir,counter_fn counter) {
    const int used=8,src_rows=mode=="src8"?8:1;
    auto ic=ggml_init({ggml_tensor_overhead()*6,nullptr,true});
    auto gc=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead(),nullptr,true});
    require(ic&&gc,"graph contexts");
    auto storage=ggml_new_tensor_3d(ic,GGML_TYPE_F32,w.k+(mode=="padded"?32:0),src_rows,tokens);
    auto x=mode=="padded"?ggml_view_3d(ic,storage,w.k,src_rows,tokens,storage->nb[1],storage->nb[2],0):storage;
    auto ids=ggml_new_tensor_2d(ic,GGML_TYPE_I32,used,tokens);
    auto ib=ggml_backend_alloc_ctx_tensors_from_buft(ic,ggml_backend_cpu_buffer_type());require(ib,"input buffer");
    ggml_set_input(x);ggml_set_input(ids);
    auto gate=ggml_mul_mat_id(gc,w.gate,x,ids),up=ggml_mul_mat_id(gc,w.up,x,ids);
    auto cg=gate,cu=up;
    if(mode!="unclamped") {
        cg=ggml_clamp(gc,gate,-INFINITY,mode=="wide"?10.f:0.125f);
        cu=ggml_clamp(gc,up,mode=="wide"?-10.f:-0.25f,mode=="wide"?10.f:0.25f);
    }
    auto y=mode=="reverse"?ggml_swiglu_split(gc,cu,cg):ggml_swiglu_split(gc,cg,cu);
    if(mode=="side_mm")y=ggml_add(gc,y,gate);
    if(mode=="side_clamp")y=ggml_add(gc,y,cg);
    ggml_set_output(y);
    auto graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,y);
    auto alloc=ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    require(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
    std::vector<float> activations(ggml_nelements(storage),-1234.f);
    for(int t=0;t<tokens;++t)for(int sr=0;sr<src_rows;++sr)for(int i=0;i<w.k;++i) {
        const int logical=(t*src_rows+sr)*w.k+i;
        float v=std::sin(float(logical)*0.0123f)+0.3f*std::cos(float(logical)*0.079f);
        if(mode=="zero")v=0.f;
        activations[(t*src_rows+sr)*storage->ne[0]+i]=v;
    }
    ggml_backend_tensor_set(storage,activations.data(),0,ggml_nbytes(storage));
    std::vector<int32_t> routes(used*tokens);
    for(int t=0;t<tokens;++t) {
        std::vector<int> selected;
        for(int j=0;j<used;++j) {
            int e=j==0?0:j==1?w.experts-1:(j*29+250+(pattern?t*31:0))%w.experts;
            while(std::find(selected.begin(),selected.end(),e)!=selected.end())e=(e+1)%w.experts;
            selected.push_back(e);routes[t*used+j]=e;
        }
    }
    ggml_backend_tensor_set(ids,routes.data(),0,ggml_nbytes(ids));
    const uint64_t count_before=counter?(counter(2)+counter(3)):0;
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"warm graph compute");
    std::vector<float> expected(ggml_nelements(y)),values(expected.size());
    ggml_backend_tensor_get(y,expected.data(),0,ggml_nbytes(y));
    std::vector<double> samples;
    for(int r=0;r<repeats;++r) {
        const auto start=std::chrono::steady_clock::now();
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat graph compute");
        samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        ggml_backend_tensor_get(y,values.data(),0,ggml_nbytes(y));
        require(std::memcmp(expected.data(),values.data(),ggml_nbytes(y))==0,"repeat bit parity");
    }
    for(float v:values)require(std::isfinite(v),"finite output");
    const uint64_t calls=(counter?(counter(2)+counter(3)):0)-count_before;
    std::sort(samples.begin(),samples.end());
    double ms=(samples[(repeats-1)/2]+samples[repeats/2])*0.5;
    const std::string key=std::string(ggml_type_name(w.type))+"-"+std::to_string(w.k)+"-"+std::to_string(w.rows)+"-"+
        std::to_string(w.experts)+"-t"+std::to_string(threads)+"-n"+std::to_string(tokens)+"-"+mode+"-p"+std::to_string(pattern);
    std::ofstream output(out_dir+"/"+key+".f32",std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()),ggml_nbytes(y));require(output.good(),"result file");
    std::printf("{\"key\":\"%s\",\"type\":\"%s\",\"k\":%d,\"rows\":%d,\"experts\":%d,\"threads\":%d,\"tokens\":%d,\"mode\":\"%s\",\"pattern\":%d,\"repeats\":%d,\"batch_calls\":%llu,\"median_ms\":%.9f,\"weight_hash\":\"%016llx\",\"output_hash\":\"%016llx\",\"finite\":true,\"repeat_parity\":true}\n",
        key.c_str(),ggml_type_name(w.type),w.k,w.rows,w.experts,threads,tokens,mode.c_str(),pattern,repeats,
        (unsigned long long)calls,ms,(unsigned long long)w.digest,(unsigned long long)hash_bytes(values.data(),ggml_nbytes(y)));
    std::fflush(stdout);
    ggml_gallocr_free(alloc);ggml_backend_buffer_free(ib);ggml_free(gc);ggml_free(ic);
}
int main(int argc,char **argv) {
    require(argc==3,"usage: expert_check OUTPUT_DIR small|real|speed");
    const std::string out_dir=argv[1],suite=argv[2];
    require(suite=="small"||suite=="real"||suite=="speed","suite");
    auto backend=ggml_backend_cpu_init();require(backend,"CPU backend");
    Dl_info info{};require(dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&info),"loaded library path");
    auto counter=reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT,"ggml_cpu_q4_batch_count"));
    std::printf("{\"cpu_library\":\"%s\",\"counter_present\":%s}\n",info.dli_fname,counter?"true":"false");std::fflush(stdout);
    cpu_set_t original;require(sched_getaffinity(0,sizeof(original),&original)==0,"read affinity");
    std::vector<int> cpus;
    for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&original))cpus.push_back(c);
    const std::vector<ggml_type> types=suite=="small"?std::vector<ggml_type>{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}:std::vector<ggml_type>{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K};
    for(auto type:types) {
        const std::vector<std::pair<int,int>> shapes=suite=="small"?std::vector<std::pair<int,int>>{{256,48},{768,80},{256,24}}:std::vector<std::pair<int,int>>{{512,4096}};
        for(auto shape:shapes) {
            if(type==GGML_TYPE_Q6_K&&shape!=std::make_pair(256,48))continue;
            Weights w(type,shape.first,shape.second,suite=="small"?40:288);
            for(int threads:suite=="speed"?std::vector<int>{15}:std::vector<int>{1,3,4,15}) {
                require(int(cpus.size())>=threads,"enough CPUs");
                auto params=ggml_threadpool_params_default(threads);
                for(int i=0;i<threads;++i){require(cpus[i]<GGML_MAX_N_THREADS,"CPU mask bound");params.cpumask[cpus[i]]=true;}
                params.strict_cpu=true;
                auto pool=ggml_threadpool_new(&params);require(pool,"pinned pool");
                ggml_backend_cpu_set_n_threads(backend,threads);ggml_backend_cpu_set_threadpool(backend,pool);
                if(suite=="speed") {
                    for(int tokens:{1,2,3,8})for(int pattern:{0,1})run(backend,w,threads,tokens,"wide",pattern,61,out_dir,counter);
                } else {
                    for(int tokens:{1,2,3,4,8,32})run(backend,w,threads,tokens,"base",1,3,out_dir,counter);
                    if(threads==3||threads==15)for(const std::string mode:{"side_mm","side_clamp","padded","src8","unclamped","reverse","wide","zero"})
                        run(backend,w,threads,3,mode,0,3,out_dir,counter);
                }
                ggml_backend_cpu_set_threadpool(backend,nullptr);ggml_threadpool_free(pool);
                require(sched_setaffinity(0,sizeof(original),&original)==0,"restore affinity");
            }
        }
    }
    ggml_backend_free(backend);
    return 0;
}
