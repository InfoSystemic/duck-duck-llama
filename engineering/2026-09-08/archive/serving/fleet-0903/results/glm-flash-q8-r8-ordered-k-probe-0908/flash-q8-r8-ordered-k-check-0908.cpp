#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "repack.h"
#include "flash-q8-r8-ordered-k-0908.h"

static void require(bool ok, const char * text) {
    if (!ok) { std::fprintf(stderr, "%s\n", text); std::abort(); }
}
static uint64_t random_state = 0xa3d5e721fe673829ULL;
static uint64_t random_word() {
    random_state ^= random_state << 13; random_state ^= random_state >> 7;
    return random_state ^= random_state << 17;
}
static uint64_t hash(const std::vector<float> & values) {
    uint64_t result = 14695981039346656037ULL;
    const auto * bytes = (const unsigned char *) values.data();
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) result = (result ^ bytes[i]) * 1099511628211ULL;
    return result;
}
static void exact(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "Output bytes differ");
}
struct Data {
    int k, nc, nr, nb, tiles, input_stride, output_stride;
    size_t groups_per_tile;
    std::vector<block_q8_0> raw_weights, activation, reference_activation;
    std::vector<block_q8_0x8> weights;
    std::vector<float> input, output;
    std::vector<flash_q8_r8_partial> partial;
    ggml_from_float_t quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    Data(int k_, int nc_, int nr_, int tiles_, bool padded) : k(k_), nc(nc_), nr(nr_), nb(k/32), tiles(tiles_),
        input_stride(k + (padded ? 19 : 0)), output_stride(nc + 7), groups_per_tile(size_t(nc/8)*nb),
        raw_weights(size_t(tiles)*nc*nb), activation(nr*nb), reference_activation(nr*nb),
        weights(tiles*groups_per_tile), input(nr*input_stride), output(size_t(tiles)*nr*output_stride, -123456.0f),
        partial(size_t(nr)*groups_per_tile) {
        const uint16_t scales[] = {0,0x8000,1,0x03ff,0x0400,0x1400,0x3c00,0xbc00,0x7bff,0x3555};
        for (auto & w : raw_weights) {
            w.d = scales[random_word()%10];
            for (auto & q : w.qs) q = int(random_word()%256)-128;
        }
        for (int tile=0; tile<tiles; ++tile) for (int g=0; g<nc/8; ++g) for (int b=0; b<nb; ++b) {
            auto & w = weights[tile*groups_per_tile+g*nb+b];
            for (int r=0; r<8; ++r) {
                const auto & original = raw_weights[(size_t(tile)*nc+g*8+r)*nb+b];
                w.d[r] = original.d;
                for (int j=0; j<32; ++j) {
                    const int offset = (j/16*2+r/4)*64 + (r%4)*16+j%16;
                    ((uint8_t *)w.qs)[offset] = uint8_t(original.qs[j]) ^ 0x80u;
                }
            }
        }
        set_input(1);
    }
    void set_input(int pattern) {
        for (int r=0; r<nr; ++r) for (int j=0; j<input_stride; ++j) {
            float value = float(int(random_word()%1025)-512)/128.0f;
            if (pattern==0) value=0;
            if (pattern==2) value=std::ldexp(j%2 ? -1.0f : 1.0f, (j/32)%41-20);
            input[r*input_stride+j]=value;
        }
        for (int r=0; r<nr; ++r) quantize(input.data()+r*input_stride, reference_activation.data()+r*nb, k);
        activation=reference_activation;
    }
    std::vector<float> canonical() const {
        std::vector<float> result(output.size(), -123456.0f);
        for (int tile=0; tile<tiles; ++tile) for (int r=0; r<nr; ++r) for (int c=0; c<nc; ++c) {
            float sum=0;
            for (int b=0; b<nb; ++b) {
                const auto & w=raw_weights[(size_t(tile)*nc+c)*nb+b];
                const auto & a=reference_activation[r*nb+b];
                int dot=0;
                for (int j=0; j<32; ++j) dot+=int(w.qs[j])*int(a.qs[j]);
                const float scale=GGML_CPU_FP16_TO_FP32(w.d)*GGML_CPU_FP16_TO_FP32(a.d);
                sum=std::fma(float(dot),scale,sum);
            }
            require(std::isfinite(sum), "Nonfinite canonical output");
            result[(size_t(tile)*nr+r)*output_stride+c]=sum;
        }
        return result;
    }
    void serial_library() {
        for (int tile=0; tile<tiles; ++tile)
            ggml_gemv_q8_0_8x8_q8_0(k,output.data()+size_t(tile)*nr*output_stride,output_stride,
                                     weights.data()+tile*groups_per_tile,reference_activation.data(),nr,nc);
    }
    void run(bool ordered, int nth, int passes) {
        cpu_set_t original;
        require(sched_getaffinity(0,sizeof(original),&original)==0,"Cannot read affinity");
        #pragma omp parallel num_threads(nth)
        {
            const int ith=omp_get_thread_num();
            require(omp_get_num_threads()==nth,"Wrong worker count");
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(48+ith,&mask);
            require(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)==0,"Cannot pin worker");
            const int first=ith*nb/nth,last=(ith+1)*nb/nth;
            for (int pass=0; pass<passes; ++pass) for (int tile=0; tile<tiles; ++tile) {
                const auto * w=weights.data()+tile*groups_per_tile;
                float * out=output.data()+size_t(tile)*nr*output_stride;
                if (ordered) {
                    for (int r=0; r<nr; ++r) if (last>first)
                        quantize(input.data()+r*input_stride+first*32,activation.data()+r*nb+first,(last-first)*32);
                    flash_q8_r8_prepare(k,nc,nr,w,activation.data(),first,last,partial.data());
                } else {
                    for (int r=ith; r<nr; r+=nth) quantize(input.data()+r*input_stride,activation.data()+r*nb,k);
                }
                #pragma omp barrier
                if (ordered) flash_q8_r8_finish(k,nc,nr,partial.data(),out,output_stride,ith,nth);
                else for (int g=ith; g<nc/8; g+=nth)
                    ggml_gemv_q8_0_8x8_q8_0(k,out+g*8,output_stride,w+g*nb,activation.data(),nr,8);
                #pragma omp barrier
            }
        }
        require(sched_setaffinity(0,sizeof(original),&original)==0,"Cannot restore affinity");
        require(std::memcmp(activation.data(),reference_activation.data(),activation.size()*sizeof(block_q8_0))==0,"Quantized activation bytes differ");
    }
};

int main(int argc,char ** argv) {
    require(argc==2,"Expected output path");
    ggml_cpu_init();omp_set_dynamic(0);
    require(GGML_CPU_FP16_TO_FP32(0x3c00)==1.0f,"Uninitialized FP16 table");
    Dl_info library{};require(dladdr((void *)ggml_gemv_q8_0_8x8_q8_0,&library),"Missing CPU binding");
    std::printf("{\"event\":\"library\",\"path\":\"%s\"}\n",library.dli_fname);
    FILE * dump=std::fopen(argv[1],"wb");require(dump,"Cannot open output");
    size_t cases=0,values=0;
    for (int k : {32,1024,4096,16384,16416,32768}) for (int nc : {8,24,40}) for (int nr : {1,2,3}) {
        Data data(k,nc,nr,1,nr%2==1);
        for (int pattern : {0,1,2}) {
            data.set_input(pattern);const auto expected=data.canonical();
            data.serial_library();exact(data.output,expected);
            for (int nth : {1,4,15}) {
                for (bool ordered : {false,true}) {
                    data.run(ordered,nth,1);exact(data.output,expected);
                    require(std::fwrite(data.output.data(),sizeof(float),data.output.size(),dump)==data.output.size(),"Write failed");
                    values+=data.output.size();
                }
                ++cases;
            }
        }
    }
    require(std::fclose(dump)==0,"Close failed");
    std::printf("{\"event\":\"correctness\",\"cases\":%zu,\"values\":%zu,\"passed\":true}\n",cases,values);std::fflush(stdout);
    for (int k : {4096,16384}) for (int nr : {1,3}) for (bool rotating : {false,true}) {
        Data data(k,24,nr,rotating?90:1,true);
        data.serial_library();const auto expected=data.output;
        const int passes=rotating?1:90;
        for (bool ordered : {false,true}) {data.run(ordered,15,passes);exact(data.output,expected);}
        std::vector<double> samples[2];
        for (int sample=0;sample<9;++sample) for (int mode : {0,1,1,0}) {
            const auto before=std::chrono::steady_clock::now();
            data.run(mode==1,15,passes);
            const auto after=std::chrono::steady_clock::now();
            samples[mode].push_back(std::chrono::duration<double,std::micro>(after-before).count()/(passes*data.tiles));
            exact(data.output,expected);
        }
        for (int mode : {0,1}) {
            auto & v=samples[mode];std::sort(v.begin(),v.end());
            std::printf("{\"event\":\"timing\",\"k\":%d,\"nr\":%d,\"rotating\":%s,\"weight_bytes\":%zu,\"ordered\":%s,\"median_us\":%.9f,\"samples\":%zu,\"hash\":\"%016llx\"}\n",
                k,nr,rotating?"true":"false",data.weights.size()*sizeof(block_q8_0x8),mode?"true":"false",
                (v[v.size()/2-1]+v[v.size()/2])/2,v.size(),(unsigned long long)hash(expected));
        }
        std::fflush(stdout);
    }
    std::printf("{\"event\":\"done\",\"passed\":true}\n");
}
