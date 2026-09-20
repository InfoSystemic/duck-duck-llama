#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "repack.h"
#include "q8-batch-candidates.h"

using Gemv = void (*)(int, float *, size_t, const void *, const void *, int, int);
template<int CHAINS, int SUM_MODE>
static void candidate(int n, float * s, size_t bs, const void * w, const void * a, int nr, int nc) {
    switch (nr) {
        case 2: return glm_q8_batch_candidate<2,CHAINS,SUM_MODE>(n,s,bs,w,a,nr,nc);
        case 3: return glm_q8_batch_candidate<3,CHAINS,SUM_MODE>(n,s,bs,w,a,nr,nc);
        case 4: return glm_q8_batch_candidate<4,CHAINS,SUM_MODE>(n,s,bs,w,a,nr,nc);
        default: std::abort();
    }
}
static const Gemv functions[] = {ggml_gemv_q8_0_x16_q8_0,candidate<1,0>,candidate<1,1>,candidate<1,2>,candidate<2,2>,candidate<4,2>,candidate<4,0>};
static const char * names[] = {"library","copy","sad","inline16","chains2-inline16","chains4-inline16","chains4"};
static uint64_t state=0x2f9401eac59d68b3ULL;
static uint64_t random_word() { state^=state<<13;state^=state>>7;state^=state<<17;return state; }
static void require(bool value,const char * message) {if(!value){std::fprintf(stderr,"%s\n",message);std::abort();}}
static void exact(const std::vector<float> & a,const std::vector<float> & b) {
    require(a.size()==b.size(),"Output size differs");
    if(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))!=0){
        for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(float))){std::fprintf(stderr,"Mismatch at %zu: %a / %a\n",i,a[i],b[i]);break;}
        std::abort();
    }
}
static size_t sum_checks() {
    alignas(32) int8_t q[32];size_t count=0;
    auto check=[&]{int sum=0;for(int j=0;j<32;++j)sum+=q[j];require(sum==glm_q8_activation_sum(q),"Integer sum differs");++count;};
    for(int v=-128;v<=127;++v){std::fill(q,q+32,v);check();}
    for(int p=0;p<32;++p)for(int v=-128;v<=127;++v){for(int j=0;j<32;++j)q[j]=int((j*37+p*19)%256)-128;q[p]=v;check();}
    for(int i=0;i<10000;++i){for(auto & v:q)v=int(random_word()%256)-128;check();}
    return count;
}
struct Data {
    int k,nr,nc,nb;size_t bs,tiles,blocks_per_tile,tile_bytes;
    std::vector<block_q8_0_x16> weights;
    std::vector<block_q8_0> activations;
    std::vector<float> output;
    Data(int k_,int nr_,int nc_,bool rotating,bool padded):k(k_),nr(nr_),nc(nc_),nb(k/32),bs(nc+(padded?11:0)){
        blocks_per_tile=size_t(nc/16)*nb;tile_bytes=blocks_per_tile*sizeof(block_q8_0_x16);
        tiles=rotating?std::max<size_t>(1,(32ULL<<20)/tile_bytes):1;
        weights.resize(tiles*blocks_per_tile);activations.resize(size_t(nr)*nb);output.resize(tiles*nr*bs,12345.25f);
        set_data(5);
    }
    void set_data(int probe){
        const uint16_t scales[]={0,0x8000,1,0x03ff,0x0400,0x1400,0x3c00,0xbc00,0x7bff,0x3555};
        for(auto & w:weights){
            for(auto & q:w.qs)q=probe==0?0:probe==1?255:random_word()%256;
            for(auto & d:w.d)d=scales[random_word()%10];
        }
        for(size_t b=0;b<activations.size();++b){auto & a=activations[b];a.d=scales[(b+probe*3)%10];
            for(int j=0;j<32;++j)a.qs[j]=probe==0?-128:probe==1?127:probe==2?0:probe==3?(j%2?127:-128):int(random_word()%256)-128;
        }
    }
    void run(Gemv fn){
        asm volatile("" : : "r"(weights.data()),"r"(activations.data()) : "memory");
        for(size_t tile=0;tile<tiles;++tile)fn(k,output.data()+tile*nr*bs,bs,weights.data()+tile*blocks_per_tile,activations.data(),nr,nc);
    }
    std::vector<float> canonical()const{
        std::vector<float> result(output.size(),12345.25f);
        for(size_t tile=0;tile<tiles;++tile)for(int t=0;t<nr;++t)for(int row=0;row<nc;++row){
            float total=0;
            for(int b=0;b<nb;++b){const auto & w=weights[tile*blocks_per_tile+(row/16)*nb+b];const auto & a=activations[t*nb+b];int dot=0;
                for(int j=0;j<32;++j){const int offset=(j/4)*64+(row%16)*4+j%4;dot+=(int(w.qs[offset])-128)*int(a.qs[j]);}
                const float scale=GGML_CPU_FP16_TO_FP32(w.d[row%16])*GGML_CPU_FP16_TO_FP32(a.d);
                total=std::fma(float(dot),scale,total);
            }
            result[tile*nr*bs+t*bs+row]=total;
        }
        return result;
    }
};
int main(int argc,char ** argv){
    require(argc==3,"Expected gate|bench and output path");ggml_cpu_init();
    Dl_info runtime{};require(dladdr(reinterpret_cast<void*>(ggml_gemv_q8_0_x16_q8_0),&runtime),"Missing CPU binding");
    std::printf("{\"event\":\"library\",\"path\":\"%s\"}\n",runtime.dli_fname);std::fflush(stdout);
    if(std::string(argv[1])=="gate"){
        const size_t sums=sum_checks();size_t cases=0,values=0;FILE * dump=std::fopen(argv[2],"wb");require(dump,"Cannot open output");
        for(int k:{32,128,512,1536,2048,4096,16384,16416,32768})for(int nr:{2,3,4})for(int nc:{16,32,64,128})for(bool padded:{false,true}){
            Data data(k,nr,nc,false,padded);
            for(int probe=0;probe<6;++probe){data.set_data(probe);const auto expected=data.canonical();
                for(float v:expected)require(std::isfinite(v),"Nonfinite canonical value");
                for(auto fn:functions){data.run(fn);exact(expected,data.output);require(std::fwrite(data.output.data(),sizeof(float),data.output.size(),dump)==data.output.size(),"Write failed");values+=data.output.size();}
                ++cases;
            }
        }
        require(std::fclose(dump)==0,"Close failed");std::printf("{\"event\":\"correctness\",\"sum_cases\":%zu,\"matrix_cases\":%zu,\"compared_values\":%zu,\"passed\":true}\n",sums,cases,values);
    }else{
        require(std::string(argv[1])=="bench","Unknown mode");
        for(int k:{512,2048,4096,16384})for(int nr:{2,3,4})for(int nc:{16,64})for(bool rotating:{false,true}){
            Data data(k,nr,nc,rotating,true);data.run(functions[0]);const auto expected=data.output;
            for(auto fn:functions){data.run(fn);exact(expected,data.output);}
            std::vector<double> samples[7];const int passes=rotating?1:std::max<size_t>(1,(2ULL<<20)/data.tile_bytes);
            for(int round=0;round<5;++round)for(int mode:{0,1,2,3,4,5,6,6,5,4,3,2,1,0}){
                const auto start=std::chrono::steady_clock::now();for(int p=0;p<passes;++p)data.run(functions[mode]);
                const auto stop=std::chrono::steady_clock::now();samples[mode].push_back(std::chrono::duration<double,std::micro>(stop-start).count()/(passes*data.tiles));
            }
            exact(expected,data.output);
            for(int mode=0;mode<7;++mode){auto & v=samples[mode];std::sort(v.begin(),v.end());std::printf("{\"event\":\"timing\",\"k\":%d,\"nr\":%d,\"nc\":%d,\"rotating\":%s,\"weight_bytes\":%zu,\"mode\":\"%s\",\"median_us\":%.9f,\"samples\":%zu}\n",k,nr,nc,rotating?"true":"false",data.weights.size()*sizeof(block_q8_0_x16),names[mode],(v[4]+v[5])/2,v.size());}
            std::fflush(stdout);
        }
    }
    std::printf("{\"event\":\"done\",\"passed\":true}\n");
}
