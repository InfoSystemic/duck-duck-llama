#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "q8-batch-candidates.h"

static bool glm_q8_batch_fast_enabled() {
    static const bool enabled=[] {
        const char * value=std::getenv("GGML_CPU_Q8_BATCH_FAST");
        return value && std::strcmp(value,"1")==0;
    }();
    if(!enabled)return false;
    static const uint32_t * control=[]()->const uint32_t * {
        const char * path=std::getenv("GGML_CPU_Q8_BATCH_FAST_CONTROL_FILE");
        if(!path || !*path)return nullptr;
        const int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);GGML_ASSERT(fd>=0);
        struct stat info;GGML_ASSERT(fstat(fd,&info)==0 && S_ISREG(info.st_mode) && info.st_size==sizeof(uint32_t));
        void * data=mmap(nullptr,sizeof(uint32_t),PROT_READ,MAP_SHARED,fd,0);close(fd);GGML_ASSERT(data!=MAP_FAILED);
        return static_cast<const uint32_t *>(data);
    }();
    if(!control)return true;
    const uint32_t value=__atomic_load_n(control,__ATOMIC_ACQUIRE);GGML_ASSERT(value<=1);return value==1;
}
static std::atomic<uint64_t> glm_q8_batch_fast_calls[5]{};
extern "C" uint64_t ggml_cpu_q8_batch_fast_count(int nr);
extern "C" uint64_t ggml_cpu_q8_batch_fast_count(int nr) {
    return nr>=2 && nr<=4 ? glm_q8_batch_fast_calls[nr].load(std::memory_order_relaxed):0;
}
static void glm_q8_batch_fast_count(int nr) {
    static const bool probe=[] {
        const char * value=std::getenv("GGML_CPU_Q8_BATCH_FAST_PROBE");
        return value && std::strcmp(value,"1")==0;
    }();
    if(probe)glm_q8_batch_fast_calls[nr].fetch_add(1,std::memory_order_relaxed);
}
