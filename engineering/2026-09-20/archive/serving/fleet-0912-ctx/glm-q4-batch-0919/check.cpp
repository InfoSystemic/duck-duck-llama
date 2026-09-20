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
extern "C" void q4_batch2(int,float * const *,const void *,const void * const *,int);
extern "C" void q4_batch3(int,float * const *,const void *,const void * const *,int);
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
            for(float &v:f)v=i%31==0?0.f:dist(gen)*(i%7==0?32.f:i%7==1?0.001f:1.f);
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

struct Activations {
    std::vector<block_q8_K> quant;
    std::vector<float> storage,reference;
    const void *qp[3];float *out[3],*ref[3];
    int n,nc,nr;
    Activations(int k,int rows,int tokens,int mode):n(k),nc(rows),nr(tokens) {
        const int nb=k/256,stride=rows+19;
        quant.resize((nb+3)*3);
        storage.assign(stride*3,123456.f);reference=storage;
        std::vector<float> values(k);
        std::mt19937 rng(823+mode*13+k+rows);std::normal_distribution<float> dist(0.f,1.f);
        for(int r=0;r<3;++r) {
            for(int j=0;j<k;++j) {
                float x=dist(rng);
                if(mode==1)x=0.f;
                if(mode==2)x=(j%2?1.f:-1.f)*std::pow(10.f,float(j/256%5-2));
                if(mode==3)x=std::sin(j*0.011f)*(r==1?0.0001f:r==2?100.f:1.f);
                values[j]=x;
            }
            qp[r]=quant.data()+(nb+3)*r+1;
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(values.data(),const_cast<void *>(qp[r]),k);
            out[r]=storage.data()+stride*r+3;ref[r]=reference.data()+stride*r+3;
        }
    }
};
static void call(bool candidate,const void *weights,Activations &a) {
    if(candidate&&a.nr==2)q4_batch2(a.n,a.out,weights,a.qp,a.nc);
    else if(candidate&&a.nr==3)q4_batch3(a.n,a.out,weights,a.qp,a.nc);
    else for(int r=0;r<a.nr;++r)ggml_gemv_q4_K_x16_q8_K(a.n,a.out[r],0,weights,a.qp[r],1,a.nc);
}
static uint64_t verify(const void *weights,Activations &a) {
    for(int r=0;r<a.nr;++r)ggml_gemv_q4_K_x16_q8_K(a.n,a.ref[r],0,weights,a.qp[r],1,a.nc);
    call(true,weights,a);
    for(float v:a.storage)require(std::isfinite(v),"finite outputs");
    require(std::memcmp(a.storage.data(),a.reference.data(),a.storage.size()*sizeof(float))==0,"bitwise parity and output guards");
    return hash_bytes(a.storage.data(),a.storage.size()*sizeof(float));
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static void speed(int rows,int banks,int nr) {
    Weights w(GGML_TYPE_Q4_K,4096,rows,banks);
    Activations a(4096,rows,nr,0);
    const size_t stride=size_t(rows/16)*(4096/256)*sizeof(block_q4_K_x16);
    for(int i=0;i<banks;++i)verify(static_cast<const char *>(w.gate->data)+stride*i,a);
    const int repetitions=banks==1?128:banks;
    std::vector<double> base,candidate,ratios;
    auto measure=[&](bool faster,int round){
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<repetitions;++i) {
            const int bank=(i*73+round*19)%banks;
            call(faster,static_cast<const char *>(w.gate->data)+stride*bank,a);
        }
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repetitions;
    };
    for(int r=0;r<4;++r){measure(false,r);measure(true,r);}
    for(int r=0;r<31;++r) {
        double b,c;
        if(r%2){c=measure(true,r);b=measure(false,r);}else{b=measure(false,r);c=measure(true,r);}
        base.push_back(b);candidate.push_back(c);ratios.push_back(b/c);
    }
    std::printf("{\"suite\":\"speed\",\"k\":4096,\"rows\":%d,\"banks\":%d,\"tokens\":%d,\"working_bytes\":%zu,\"rounds\":31,\"repetitions\":%d,\"base_us\":%.6f,\"batch_us\":%.6f,\"median_paired_speedup\":%.6f,\"samples\":[",rows,banks,nr,stride*banks,repetitions,median(base),median(candidate),median(ratios));
    for(int i=0;i<31;++i)std::printf("%s[%.6f,%.6f]",i?",":"",base[i],candidate[i]);
    std::printf("]}\n");std::fflush(stdout);
}
int main(int argc,char **argv) {
    require(argc==2,"usage: check correctness|speed");
    auto backend=ggml_backend_cpu_init();require(backend,"backend");
    Dl_info info{};require(dladdr(reinterpret_cast<void *>(ggml_gemv_q4_K_x16_q8_K),&info),"production symbol");
    std::printf("{\"cpu_library\":\"%s\",\"cpu\":%d,\"q4_size\":%zu,\"q8_size\":%zu}\n",info.dli_fname,sched_getcpu(),sizeof(block_q4_K_x16),sizeof(block_q8_K));
    if(std::string(argv[1])=="correctness") {
        int cases=0;uint64_t digest=0;
        for(int k:{256,512,768,1536,4096})for(int rows:{16,32,48,64,80,512}) {
            Weights w(GGML_TYPE_Q4_K,k,rows,3);
            const size_t stride=size_t(rows/16)*(k/256)*sizeof(block_q4_K_x16);
            for(int nr:{1,2,3})for(int mode:{0,1,2,3})for(int bank=0;bank<3;++bank) {
                Activations a(k,rows,nr,mode);
                digest^=verify(static_cast<const char *>(w.gate->data)+stride*bank,a);++cases;
            }
        }
        std::printf("{\"suite\":\"correctness\",\"cases\":%d,\"bitwise_parity\":true,\"output_guards\":true,\"digest\":\"%016llx\"}\n",cases,(unsigned long long)digest);
    } else {
        require(std::string(argv[1])=="speed","suite name");
        for(int rows:{64,512})for(int banks:{1,rows==64?256:64})for(int nr:{1,2,3})speed(rows,banks,nr);
    }
    ggml_backend_free(backend);
}
