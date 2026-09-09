#include "flash-rms-guarded-0908.h"
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint64_t rng = 0x391f012375ab4781ULL;
static uint64_t random_word() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static void require(bool value, const char * what) {
    if (!value) { std::fprintf(stderr, "%s\n", what); std::abort(); }
}
__attribute__((noinline)) static float scalar_mean(const float * x, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += double(x[i] * x[i]);
    return float(sum / n);
}
__attribute__((noinline)) static float guarded_mean(const float * x, int64_t n) {
    float mean;
    return flash_rms_guarded_mean(x,n,&mean) == FLASH_RMS_GUARD_OK ? mean : scalar_mean(x,n);
}
static uint32_t bits(float x) { uint32_t value;std::memcpy(&value,&x,4);return value; }
static size_t counts[FLASH_RMS_GUARD_STATUS_COUNT]{};
static size_t cases = 0, verified_values = 0;
static void check(const float * x, int64_t n, int expected = -1) {
    float fast = 0.0f;
    const auto status = flash_rms_guarded_mean(x,n,&fast);
    ++counts[status];++cases;
    if (expected >= 0) require(int(status) == expected,"Unexpected guard outcome");
    const float reference = scalar_mean(x,n);
    if (status == FLASH_RMS_GUARD_OK) require(bits(fast) == bits(reference),"Accepted mean differs from serial mean");
    const float actual = guarded_mean(x,n);
    require(bits(actual) == bits(reference),"Fallback mean differs");
    verified_values += size_t(n);
}
static void fill(std::vector<float> & x, int pattern) {
    for (size_t i = 0; i < x.size(); ++i) {
        const int sign = random_word() & 1 ? 1 : -1;
        const float mantissa = 1.0f + float(random_word() & 0x7fffff) * 0x1p-23f;
        if (pattern == 0) x[i] = i & 1 ? -0.0f : 0.0f;
        else if (pattern == 1) x[i] = float(sign);
        else if (pattern == 2) x[i] = float(int(random_word() % 257) - 128) / 127.0f;
        else if (pattern == 3) x[i] = std::ldexp(float(sign) * mantissa, int(random_word()%138)-74);
        else if (pattern == 4) x[i] = std::ldexp(float(sign) * mantissa, -74 + int(i%5));
        else if (pattern == 5) x[i] = std::ldexp(float(sign) * mantissa, 58 + int(i%5));
        else if (pattern == 6) x[i] = i % 127 ? 0x1p-25f : 1.0f;
        else x[i] = std::ldexp(float(sign) * mantissa, int(random_word()%31)-15);
    }
}
int main() {
    static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53);
    const unsigned original_csr = _mm_getcsr();
    require((original_csr & _MM_ROUND_MASK) == _MM_ROUND_NEAREST,"Expected nearest rounding");
    for (int n : {1,32,511,1023,1024,1025,2048,4096,8192,16384,16415,32768,65536})
    for (int pattern = 0; pattern < 8; ++pattern)
    for (int offset : {0,1,7}) {
        std::vector<float> values(n + offset);fill(values,pattern);
        check(values.data()+offset,n,n<1024 ? FLASH_RMS_GUARD_LENGTH : -1);
    }
    for (int trial=0;trial<1200;++trial) {
        const int sizes[]={1024,1025,2048,3072,4096,8192,16384,16415,32768};
        const int n=sizes[trial%9];std::vector<float> values(n);
        fill(values,trial%8);check(values.data(),n);
    }
    for (int n : {1024,2048,4096}) for (int exponent : {-60,-10,0,10,50}) {
        std::vector<float> values(n,std::ldexp(1.0f,exponent));
        values[n/2]=std::ldexp(1.0f+float(n)*0x1p-25f,exponent);
        check(values.data(),n,FLASH_RMS_GUARD_BOUNDARY);
        values[n/2]=std::nextafter(values[n/2],0.0f);check(values.data(),n);
        values[n/2]=std::nextafter(std::nextafter(values[n/2],INFINITY),INFINITY);check(values.data(),n);
    }
    for (int exponent : {-74,-50,0,62}) {
        std::vector<float> values((1<<20)+3,std::ldexp(1.0f,exponent));check(values.data(),values.size());
    }
    {
        std::vector<float> values(1<<24,1.0f);check(values.data(),values.size(),FLASH_RMS_GUARD_OK);
        float mean;require(flash_rms_guarded_mean(nullptr,(int64_t(1)<<24)+1,&mean)==FLASH_RMS_GUARD_LENGTH,"Length cap failed");
    }
    for (float special : {INFINITY,-INFINITY,NAN,FLT_MAX}) for (int position : {0,1023,1024}) {
        std::vector<float> values(1025,1.0f);values[position]=special;
        check(values.data(),values.size(),FLASH_RMS_GUARD_NONFINITE);
    }
    for (unsigned rounding : {_MM_ROUND_DOWN,_MM_ROUND_UP,_MM_ROUND_TOWARD_ZERO}) {
        _mm_setcsr((original_csr & ~_MM_ROUND_MASK) | rounding);
        std::vector<float> values(1024,0.125f);check(values.data(),values.size(),FLASH_RMS_GUARD_ENVIRONMENT);
    }
    _mm_setcsr(original_csr);
    for (unsigned mode : {0u,0x40u,0x8000u,0x8040u}) {
        _mm_setcsr((original_csr & ~0x8040u) | mode);
        std::vector<float> values(4096);fill(values,4);check(values.data(),values.size());
    }
    _mm_setcsr(original_csr);
    require(counts[FLASH_RMS_GUARD_OK]>1000 && counts[FLASH_RMS_GUARD_BOUNDARY]>=15,"Missing positive or fallback coverage");
    std::printf("{\"event\":\"correctness\",\"passed\":true,\"cases\":%zu,\"values\":%zu,\"status_counts\":[",cases,verified_values);
    for (int i=0;i<FLASH_RMS_GUARD_STATUS_COUNT;++i) std::printf("%s%zu",i?",":"",counts[i]);
    std::printf("]}\n");std::fflush(stdout);
    volatile float checksum=0;
    for (int n : {1024,4096,16384,65536}) {
        std::vector<float> values(n);fill(values,2);
        require(bits(scalar_mean(values.data(),n))==bits(guarded_mean(values.data(),n)),"Timing outputs differ");
        const int passes=std::max(4,(2<<20)/(n*4));
        std::vector<double> times[2];
        for (int sample=0;sample<21;++sample) for (int mode : {0,1,1,0}) {
            const auto start=std::chrono::steady_clock::now();
            for (int pass=0;pass<passes;++pass) {
                asm volatile("" : : "r"(values.data()) : "memory");
                checksum=mode ? guarded_mean(values.data(),n) : scalar_mean(values.data(),n);
            }
            const auto end=std::chrono::steady_clock::now();
            times[mode].push_back(std::chrono::duration<double,std::micro>(end-start).count()/passes);
        }
        for(int mode=0;mode<2;++mode) {
            auto & t=times[mode];std::sort(t.begin(),t.end());
            std::printf("{\"event\":\"timing\",\"n\":%d,\"mode\":%d,\"median_us\":%.9f,\"samples\":%zu}\n",n,mode,(t[20]+t[21])/2,t.size());
        }
    }
    std::printf("{\"event\":\"done\",\"passed\":true,\"checksum\":%.9g}\n",double(checksum));
}
