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

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
static double median_ms(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}
static double sum_ms(const std::vector<double> &v) {double x=0;for(double a:v)x+=a;return x;}
static void print_samples(const char *name,const std::vector<double> &v) {
    std::printf(",\"%s\":[",name);for(size_t i=0;i<v.size();++i)std::printf("%s%.6f",i?",":"",v[i]);std::printf("]");
}
int main(int argc,char **argv) {
    check(argc==6,"usage: probe K HIDDEN TARGET_BLOCKS DRAFT_BLOCKS ROUNDS");
    const int k=std::atoi(argv[1]),hidden=std::atoi(argv[2]),blocks=std::atoi(argv[3]),draft_blocks=std::atoi(argv[4]),rounds=std::atoi(argv[5]);
    const char *sets_env=std::getenv("PROBE_WEIGHT_SETS");const int sets=sets_env?std::atoi(sets_env):blocks;
    check(k>=256&&k<=4096&&hidden>=1024&&hidden<=8192&&blocks>0&&blocks<=32&&draft_blocks>0&&draft_blocks<=4&&rounds>0&&rounds<=100&&sets>0&&sets<=blocks,"bounded sizes");
    const char *control=std::getenv("GGML_CPU_Q8_BATCH_FAST_CONTROL_FILE");check(control&&*control,"mode control");
    const int fd=open(control,O_WRONLY|O_CLOEXEC|O_NOFOLLOW);check(fd>=0,"control open");
    struct stat st{};check(fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_size==4,"control file");
    auto set_mode=[&](uint32_t value){check(pwrite(fd,&value,4,0)==4,"control write");};
    set_mode(0);
    Dl_info ci{},bi{};check(dladdr((void *)ggml_backend_cpu_reg,&ci)&&dladdr((void *)ggml_backend_meta_device,&bi),"library identities");
    std::printf("{\"event\":\"library\",\"cpu_library\":\"%s\",\"base_library\":\"%s\",\"workers\":\"%s\"}\n",ci.dli_fname,bi.dli_fname,std::getenv("GGML_CPU_NUMA_THREADS"));std::fflush(stdout);
    using Count=uint64_t(*)(int);auto count=(Count)dlsym(RTLD_DEFAULT,"ggml_cpu_q8_batch_fast_count");check(count,"candidate counter export");
    auto reg=ggml_backend_cpu_reg();ggml_backend_dev_t devs[4]{};
    for(size_t i=0;i<ggml_backend_reg_dev_count(reg);++i){auto d=ggml_backend_reg_dev_get(reg,i);for(int n=0;n<4;++n)if(std::string(ggml_backend_dev_name(d))=="CPU-NUMA"+std::to_string(n))devs[n]=d;}
    for(auto d:devs)check(d,"four NUMA devices");
    auto dev=ggml_backend_meta_device(devs,4,split,nullptr);check(dev,"meta device");
    Workload target(dev,k,hidden,blocks,sets,3,8317);Workload draft(dev,k,hidden,draft_blocks,1,1,8829);
    const char *sequence=std::getenv("PROBE_SEQUENCE");if(!sequence)sequence="01101001";
    check(std::strlen(sequence)<=8,"bounded sequence");
    const char *output=std::getenv("PROBE_OUTPUT_PREFIX");check(output,"output path");
    for(int kind=0;kind<2;++kind){auto &v=kind?draft.expected:target.expected;std::string path=std::string(output)+(kind?".draft.bin":".target.bin");FILE *f=std::fopen(path.c_str(),"wb");check(f,"output open");check(std::fwrite(v.data(),4,v.size(),f)==v.size(),"output save");check(std::fclose(f)==0,"output close");}
    for(size_t order=0;order<std::strlen(sequence);++order) {
        check(sequence[order]=='0'||sequence[order]=='1',"binary mode");const unsigned enabled=sequence[order]-'0';set_mode(enabled);
        uint64_t counts_before[3]={count(2),count(3),count(4)};
        for(int warm=0;warm<2;++warm){target.compute();draft.compute();draft.compute();target.verify();draft.verify();}
        std::vector<double> target_ms,draft_ms,cycle_ms;double user_before=0,system_before=0,user_after=0,system_after=0;
        const double cpu_before=cpu_time(&user_before,&system_before);auto start=std::chrono::steady_clock::now();
        for(int i=0;i<rounds;++i) {
            auto t=std::chrono::steady_clock::now();target.compute();auto m=std::chrono::steady_clock::now();draft.compute();draft.compute();auto e=std::chrono::steady_clock::now();
            target_ms.push_back(std::chrono::duration<double,std::milli>(m-t).count());draft_ms.push_back(std::chrono::duration<double,std::milli>(e-m).count());cycle_ms.push_back(std::chrono::duration<double,std::milli>(e-t).count());target.verify();draft.verify();
        }
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),cpu=cpu_time(&user_after,&system_after)-cpu_before;
        uint64_t calls[3]={count(2)-counts_before[0],count(3)-counts_before[1],count(4)-counts_before[2]};
        std::printf("{\"event\":\"arm\",\"order\":%zu,\"enabled\":%u,\"rounds\":%d,\"weight_sets\":%d,\"median_cycle_ms\":%.6f,\"median_target_ms\":%.6f,\"median_draft_ms\":%.6f,\"total_cycle_ms\":%.6f,\"total_target_ms\":%.6f,\"total_draft_ms\":%.6f,\"wall_seconds\":%.6f,\"cpu_seconds\":%.6f,\"user_seconds\":%.6f,\"system_seconds\":%.6f,\"calls\":[%llu,%llu,%llu],\"exact\":true",order,enabled,rounds,sets,median_ms(cycle_ms),median_ms(target_ms),median_ms(draft_ms),sum_ms(cycle_ms),sum_ms(target_ms),sum_ms(draft_ms),wall,cpu,user_after-user_before,system_after-system_before,(unsigned long long)calls[0],(unsigned long long)calls[1],(unsigned long long)calls[2]);
        print_samples("target_ms",target_ms);print_samples("draft_ms",draft_ms);print_samples("cycle_ms",cycle_ms);std::printf("}\n");std::fflush(stdout);
    }
    set_mode(0);check(close(fd)==0,"control close");std::printf("{\"event\":\"done\",\"passed\":true}\n");
}
