#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/resource.h>
static void check(bool ok,const char *why){if(!ok){std::fprintf(stderr,"FAILED %s\n",why);std::abort();}}
static uint32_t rng=0x518346a7;
static uint32_t bits(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static ggml_backend_meta_split_state split(const ggml_tensor *t,void *user){
    const int n=*(int *)user;ggml_backend_meta_split_state s{};s.nr[0]=1;s.n_segments=1;s.axis=GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if(std::strcmp(t->name,"pool_observer.weight")==0){s.axis=GGML_BACKEND_SPLIT_AXIS_1;for(int j=0;j<n;++j)s.ne[j]=128;}
    return s;
}
static double cpu_time(){rusage u{};check(getrusage(RUSAGE_SELF,&u)==0,"rusage");return u.ru_utime.tv_sec+u.ru_utime.tv_usec*1e-6+u.ru_stime.tv_sec+u.ru_stime.tv_usec*1e-6;}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char **argv){
    check(argc==5,"usage: bench POOLS LAYERS ROUNDS NUMA_DEVICES");int pools=std::atoi(argv[1]),layers=std::atoi(argv[2]),rounds=std::atoi(argv[3]),ndev=std::atoi(argv[4]);
    check(pools>0&&pools<=25002&&layers>0&&layers<=11&&rounds>0&&rounds<=61&&(ndev==1||ndev==4),"bounds");
    const bool observer=std::getenv("POOL_BENCH_OBSERVER")!=nullptr;
    const char *modepath=std::getenv("GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE");check(modepath,"control path");int fd=open(modepath,O_WRONLY|O_CLOEXEC);check(fd>=0,"control open");
    auto mode=[&](uint32_t v){check(pwrite(fd,&v,4,0)==4,"control write");};mode(0);
    auto stat=(uint64_t(*)(int))dlsym(RTLD_DEFAULT,"ggml_cpu_glm_pool_cache_stat");check(stat,"candidate stat");
    auto fused=(uint64_t(*)())dlsym(RTLD_DEFAULT,"ggml_cpu_glm_pool_fused_count");check(fused,"fusion count");
    Dl_info ci{};check(dladdr((void *)ggml_backend_cpu_reg,&ci),"library identity");std::printf("{\"event\":\"library\",\"path\":\"%s\",\"devices\":%d,\"workers\":\"%s\",\"observer\":%s}\n",ci.dli_fname,ndev,std::getenv("GGML_CPU_NUMA_THREADS"),observer?"true":"false");std::fflush(stdout);
    auto reg=ggml_backend_cpu_reg();ggml_backend_dev_t devs[4]{};
    for(size_t i=0;i<ggml_backend_reg_dev_count(reg);++i){auto d=ggml_backend_reg_dev_get(reg,i);for(int j=0;j<ndev;++j)if(std::string(ggml_backend_dev_name(d))=="CPU-NUMA"+std::to_string(j))devs[j]=d;}
    for(int j=0;j<ndev;++j)check(devs[j],"NUMA device");auto dev=ggml_backend_meta_device(devs,ndev,split,&ndev);check(dev,"meta device");
    auto backend=ggml_backend_dev_init(dev,nullptr);check(backend,"backend");auto buft=ggml_backend_dev_buffer_type(dev);
    auto wc=ggml_init({ggml_tensor_overhead()*64,nullptr,true}),ic=ggml_init({ggml_tensor_overhead()*64,nullptr,true});
    auto gc=ggml_init({ggml_tensor_overhead()*1024+ggml_graph_overhead_custom(2048,false),nullptr,true});check(wc&&ic&&gc,"contexts");
    std::vector<ggml_tensor *> ape,kg;
    for(int l=0;l<layers;++l){ape.push_back(ggml_new_tensor_2d(wc,GGML_TYPE_F32,128,4));kg.push_back(ggml_new_tensor_3d(ic,GGML_TYPE_F16,256,4*pools+7,1));ggml_format_name(ape.back(),"blk.%d.indexer_compressor_ape.weight",l*4+3);ggml_format_name(kg.back(),"cache_k_l%d",l*4+3);}
    auto cells=ggml_new_tensor_2d(ic,GGML_TYPE_I32,4*pools,1);ggml_set_name(cells,"kpool_pool_cells");
    ggml_tensor *observe=nullptr;if(observer){observe=ggml_new_tensor_2d(wc,GGML_TYPE_F32,128,128*ndev);ggml_set_name(observe,"pool_observer.weight");}
    auto wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,buft),ib=ggml_backend_alloc_ctx_tensors_from_buft(ic,buft);check(wb&&ib,"buffers");ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<int32_t> map(4*pools);for(size_t i=0;i<map.size();++i)map[i]=int32_t(i);ggml_backend_tensor_set(cells,map.data(),0,map.size()*4);
    std::vector<ggml_fp16_t> data(size_t(4*pools+7)*256);
    for(int l=0;l<layers;++l){for(auto &x:data)x=ggml_fp32_to_fp16((int(bits()%8193)-4096)/1024.f);ggml_backend_tensor_set(kg[l],data.data(),0,data.size()*2);float a[512];for(float &x:a)x=(int(bits()%8193)-4096)/8192.f;ggml_backend_tensor_set(ape[l],a,0,sizeof(a));}
    data.clear();data.shrink_to_fit();
    if(observer){std::vector<float>w(128*128*ndev,0.f);for(int j=0;j<ndev;++j)for(int c=0;c<128;++c)w[(j*128+c)*128+c]=1.f;ggml_backend_tensor_set(observe,w.data(),0,w.size()*4);}
    ggml_tensor *result=nullptr;
    for(int l=0;l<layers;++l){
        auto members=ggml_get_rows(gc,kg[l],cells);ggml_format_name(members,"indexer_pool_members-%d",l*4+3);
        auto k=ggml_view_4d(gc,members,128,4,pools,1,members->nb[1],members->nb[1]*4,members->nb[2],0);
        auto g=ggml_view_4d(gc,members,128,4,pools,1,members->nb[1],members->nb[1]*4,members->nb[2],128*4);
        k=ggml_cont(gc,ggml_permute(gc,k,1,0,2,3));g=ggml_cont(gc,ggml_permute(gc,g,1,0,2,3));auto a=ggml_cont(gc,ggml_transpose(gc,ape[l]));g=ggml_add(gc,g,ggml_reshape_4d(gc,a,4,128,1,1));
        auto p=ggml_soft_max(gc,g);auto v=ggml_reshape_2d(gc,ggml_sum_rows(gc,ggml_mul(gc,k,p)),128,pools);result=result?ggml_add(gc,result,v):v;
    }
    if(observer)result=ggml_mul_mat(gc,observe,result);ggml_set_output(result);
    auto graph=ggml_new_graph_custom(gc,2048,false);ggml_build_forward_expand(graph,result);auto alloc=ggml_gallocr_new(buft);check(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
    auto run=[&]{check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute");ggml_backend_synchronize(backend);};
    auto tail=[&](int step){ggml_fp16_t v[3*256];for(int l=0;l<layers;++l){for(int i=0;i<3*256;++i)v[i]=ggml_fp32_to_fp16(float((i*37+step*17+l*29)%8193-4096)/1024.f);ggml_backend_tensor_set(kg[l],v,size_t(4*pools-3)*256*2,sizeof(v));}};
    const uint64_t fused_before=fused();tail(0);run();
    const size_t n=ggml_nelements(result);std::vector<float> actual(n);std::vector<std::vector<float>> expected(rounds);std::vector<float> initial(n);ggml_backend_tensor_get(result,initial.data(),0,n*4);
    for(float x:initial)check(std::isfinite(x),"finite initial");
    const char *sequence=std::getenv("POOL_BENCH_SEQUENCE");if(!sequence)sequence="01101001";check(std::strlen(sequence)<=8&&sequence[0]=='0',"balanced sequence");
    const char *output_path=std::getenv("POOL_BENCH_OUTPUT");check(output_path,"output path");FILE *output=std::fopen(output_path,"wb");check(output,"output open");
    bool first_on=true;double first_on_ms=0;uint64_t total_graphs=1;
    for(size_t arm=0;arm<std::strlen(sequence);++arm){
        check(sequence[arm]=='0'||sequence[arm]=='1',"binary arm");const unsigned enabled=sequence[arm]-'0';mode(enabled);tail(0);
        const auto cold=std::chrono::steady_clock::now();run();const double prime_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold).count();run();total_graphs+=2;
        if(enabled&&first_on){first_on=false;first_on_ms=prime_ms;}
        ggml_backend_tensor_get(result,actual.data(),0,n*4);check(std::memcmp(actual.data(),initial.data(),n*4)==0,"prime parity");
        std::vector<double> samples;const double cpu_before=cpu_time();auto wall_start=std::chrono::steady_clock::now();
        for(int step=1;step<=rounds;++step){
            tail(step);const auto start=std::chrono::steady_clock::now();run();samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());++total_graphs;
            ggml_backend_tensor_get(result,actual.data(),0,n*4);
            if(arm==0){expected[step-1]=actual;check(std::fwrite(actual.data(),4,n,output)==n,"reference save");}else check(std::memcmp(actual.data(),expected[step-1].data(),n*4)==0,"arm output parity");
        }
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count(),cpu=cpu_time()-cpu_before;double total=0;for(double x:samples)total+=x;
        std::printf("{\"event\":\"arm\",\"order\":%zu,\"enabled\":%u,\"rounds\":%d,\"median_ms\":%.6f,\"prime_ms\":%.6f,\"total_ms\":%.6f,\"wall_seconds\":%.6f,\"cpu_seconds\":%.6f,\"exact\":true,\"samples_ms\":[",arm,enabled,rounds,med(samples),prime_ms,total,wall,cpu);
        for(size_t i=0;i<samples.size();++i)std::printf("%s%.6f",i?",":"",samples[i]);std::printf("]}\n");std::fflush(stdout);
    }
    check(std::fclose(output)==0,"output close");mode(0);check(close(fd)==0,"control close");
    const uint64_t fused_delta=fused()-fused_before;check(fused_delta==total_graphs*layers*ndev,"all pooling layers fused on all devices");
    std::printf("{\"event\":\"done\",\"passed\":true,\"graphs\":%llu,\"fusions\":%llu,\"first_on_ms\":%.6f,\"output_values_per_step\":%zu,\"stats\":[",(unsigned long long)total_graphs,(unsigned long long)fused_delta,first_on_ms,n);
    for(int i=0;i<7;++i)std::printf("%s%llu",i?",":"",(unsigned long long)stat(i));std::printf("]}\n");std::fflush(stdout);
    ggml_backend_free(backend);ggml_gallocr_free(alloc);ggml_backend_buffer_free(ib);ggml_backend_buffer_free(wb);ggml_free(gc);ggml_free(ic);ggml_free(wc);
}
