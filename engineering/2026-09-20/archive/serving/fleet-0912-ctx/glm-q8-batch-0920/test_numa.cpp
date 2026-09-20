#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <omp.h>
#include <thread>
static void check(bool ok,const char *what){if(!ok){std::fprintf(stderr,"CHECK FAILED: %s\n",what);std::abort();}}
static uint64_t hash(const void *v,size_t n){auto p=(const unsigned char *)v;uint64_t h=14695981039346656037ULL;for(size_t i=0;i<n;++i)h=(h^p[i])*1099511628211ULL;return h;}
static ggml_backend_meta_split_state split(const ggml_tensor *t,void *){
    ggml_backend_meta_split_state s{};s.nr[0]=1;s.n_segments=1;
    const std::string name=t->name;
    s.axis=name.find(".attn_up")!=std::string::npos?GGML_BACKEND_SPLIT_AXIS_1:name.find(".attn_output")!=std::string::npos?GGML_BACKEND_SPLIT_AXIS_0:GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if(int(s.axis)<4){check(t->ne[s.axis]%4==0,"equal split");for(int i=0;i<4;++i)s.ne[i]=t->ne[s.axis]/4;}
    return s;
}
struct Workload{
    ggml_backend_t backend{};ggml_context *wc{},*ic{},*gc{};
    ggml_backend_buffer_t wb{},ib{};ggml_gallocr_t alloc{};ggml_cgraph *graph{};ggml_tensor *input{},*output{};
    std::vector<float> expected,actual;
    Workload(ggml_backend_dev_t dev,int k,int hidden,int blocks,int sets,int tokens,int seed){
        backend=ggml_backend_dev_init(dev,nullptr);check(backend,"meta backend");
        auto regular=ggml_backend_dev_buffer_type(dev);
        auto extras=ggml_backend_meta_device_get_extra_bufts(dev);check(extras&&extras[0],"NUMA repack extra buffer");
        ggml_backend_buffer_type_t packed=nullptr;
        for(int i=0;extras[i];++i){std::string n=ggml_backend_buft_name(extras[i]);std::fprintf(stderr,"extra buft %s\n",n.c_str());std::transform(n.begin(),n.end(),n.begin(),[](unsigned char c){return std::tolower(c);});if(n.find("repack")!=std::string::npos)packed=extras[i];}
        check(packed,"repack buffer found");
        wc=ggml_init({ggml_tensor_overhead()*size_t(sets*2+4),nullptr,true});check(wc,"weight context");
        std::vector<ggml_tensor *> up,down;
        for(int i=0;i<sets;++i){auto u=ggml_new_tensor_2d(wc,GGML_TYPE_Q8_0,k,hidden);auto d=ggml_new_tensor_2d(wc,GGML_TYPE_Q8_0,hidden,k);ggml_format_name(u,"blk.%d.attn_up.weight",i);ggml_format_name(d,"blk.%d.attn_output.weight",i);up.push_back(u);down.push_back(d);}
        wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,packed);check(wb,"weight allocation");ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 gen(seed);std::normal_distribution<float> dist(0.f,0.02f);
        std::vector<block_q8_0> palette(4096);float f[32];
        for(auto &q:palette){for(float &v:f)v=dist(gen);quantize_row_q8_0_ref(f,&q,32);}
        for(int i=0;i<sets;++i)for(auto t:{up[i],down[i]}){
            std::vector<block_q8_0> data(ggml_nbytes(t)/sizeof(block_q8_0));
            for(auto &q:data)q=palette[gen()%palette.size()];
            ggml_backend_tensor_set(t,data.data(),0,ggml_nbytes(t));
        }
        ic=ggml_init({ggml_tensor_overhead()*4,nullptr,true});input=ggml_new_tensor_2d(ic,GGML_TYPE_F32,k,tokens);ggml_set_name(input,"input");ggml_set_input(input);
        ib=ggml_backend_alloc_ctx_tensors_from_buft(ic,regular);check(ib,"input allocation");
        std::vector<float> values(k*tokens);for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*0.013f);ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
        gc=ggml_init({ggml_tensor_overhead()*size_t(blocks*8+8)+ggml_graph_overhead_custom(8192,false),nullptr,true});check(gc,"graph context");
        auto x=input;
        for(int i=0;i<blocks;++i){auto h=ggml_silu(gc,ggml_mul_mat(gc,up[i%sets],x));auto y=ggml_mul_mat(gc,down[i%sets],h);x=ggml_rms_norm(gc,ggml_add(gc,y,x),1e-5f);ggml_format_name(x,"layer.%d.norm",i);}
        output=x;ggml_set_output(output);graph=ggml_new_graph_custom(gc,8192,false);ggml_build_forward_expand(graph,output);
        alloc=ggml_gallocr_new(regular);check(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
        expected.resize(ggml_nelements(output));actual.resize(expected.size());
        compute();ggml_backend_tensor_get(output,expected.data(),0,expected.size()*sizeof(float));
        for(float v:expected)check(std::isfinite(v),"finite warm output");
        std::printf("{\"setup\":true,\"tokens\":%d,\"k\":%d,\"hidden\":%d,\"blocks\":%d,\"weight_sets\":%d,\"nodes\":%d,\"output_hash\":\"%016llx\",\"weight_alloc_bytes\":%zu}\n",tokens,k,hidden,blocks,sets,ggml_graph_n_nodes(graph),(unsigned long long)hash(expected.data(),expected.size()*4),ggml_backend_buffer_get_size(wb));std::fflush(stdout);
    }
    void compute(){check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute");ggml_backend_synchronize(backend);}
    void verify(){ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));check(std::memcmp(expected.data(),actual.data(),actual.size()*4)==0,"repeat bit parity");}
    ~Workload(){ggml_backend_free(backend);ggml_gallocr_free(alloc);ggml_backend_buffer_free(ib);ggml_backend_buffer_free(wb);ggml_free(gc);ggml_free(ic);ggml_free(wc);}
};
static double cpu_time(double *user=nullptr,double *system=nullptr){rusage u{};check(getrusage(RUSAGE_SELF,&u)==0,"rusage");double a=u.ru_utime.tv_sec+u.ru_utime.tv_usec*1e-6,b=u.ru_stime.tv_sec+u.ru_stime.tv_usec*1e-6;if(user)*user=a;if(system)*system=b;return a+b;}
int main(int argc,char **argv){
    check(argc==6,"usage: probe K HIDDEN TARGET_BLOCKS DRAFT_BLOCKS ROUNDS");
    int k=std::atoi(argv[1]),hidden=std::atoi(argv[2]),target_blocks=std::atoi(argv[3]),draft_blocks=std::atoi(argv[4]),rounds=std::atoi(argv[5]);
    check(k>=256&&k<=4096&&hidden>=1024&&hidden<=8192&&target_blocks>0&&target_blocks<=512&&draft_blocks>0&&draft_blocks<=32&&rounds>0&&rounds<=100,"bounded sizes");
    Dl_info ci{},bi{};check(dladdr((void *)ggml_backend_cpu_reg,&ci)&&dladdr((void *)ggml_backend_meta_device,&bi),"library identities");
    std::printf("{\"cpu_library\":\"%s\",\"base_library\":\"%s\",\"workers\":\"%s\",\"spin_us\":\"%s\"}\n",ci.dli_fname,bi.dli_fname,std::getenv("GGML_CPU_NUMA_THREADS"),std::getenv("GGML_CPU_NUMA_DISPATCH_SPIN_US"));std::fflush(stdout);
    auto reg=ggml_backend_cpu_reg();ggml_backend_dev_t devs[4]{};
    for(size_t i=0;i<ggml_backend_reg_dev_count(reg);++i){auto d=ggml_backend_reg_dev_get(reg,i);for(int n=0;n<4;++n)if(std::string(ggml_backend_dev_name(d))=="CPU-NUMA"+std::to_string(n))devs[n]=d;}
    for(auto d:devs)check(d,"four NUMA devices");
    auto dev=ggml_backend_meta_device(devs,4,split,nullptr);check(dev,"meta device");
    Workload target(dev,k,hidden,target_blocks,std::min(target_blocks,4),3,8317);
    Workload draft(dev,k,hidden,draft_blocks,1,1,8829);
    const char *output_prefix=std::getenv("PROBE_OUTPUT_PREFIX");
    if(output_prefix){
        auto save_output=[&](const char *kind,const std::vector<float> &v){
            std::string path=std::string(output_prefix)+"."+kind+".bin";
            FILE *f=std::fopen(path.c_str(),"wb");check(f,"open output bytes");
            check(std::fwrite(v.data(),sizeof(float),v.size(),f)==v.size(),"save output bytes");
            check(std::fclose(f)==0,"close output bytes");
        };
        save_output("target",target.expected);save_output("draft",draft.expected);
    }
    for(int i=0;i<3;++i){target.compute();draft.compute();draft.compute();}
    double user_before=0,system_before=0,user_after=0,system_after=0;const double before=cpu_time(&user_before,&system_before);std::vector<double> samples;auto start=std::chrono::steady_clock::now();
    for(int i=0;i<rounds;++i){auto t=std::chrono::steady_clock::now();target.compute();draft.compute();draft.compute();samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());target.verify();draft.verify();}
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),cpu=cpu_time(&user_after,&system_after)-before;
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    std::printf("{\"completed\":true,\"rounds\":%d,\"median_cycle_ms\":%.6f,\"wall_seconds\":%.6f,\"cpu_seconds\":%.6f,\"cpu_cores\":%.6f,\"user_seconds\":%.6f,\"system_seconds\":%.6f,\"target_hash\":\"%016llx\",\"draft_hash\":\"%016llx\",\"samples_ms\":[",rounds,sorted[rounds/2],wall,cpu,cpu/wall,user_after-user_before,system_after-system_before,(unsigned long long)hash(target.expected.data(),target.expected.size()*4),(unsigned long long)hash(draft.expected.data(),draft.expected.size()*4));
    for(size_t i=0;i<samples.size();++i)std::printf("%s%.6f",i?",":"",samples[i]);std::printf("]}\n");std::fflush(stdout);
}
