#include "llama-mmap.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

struct advice_call {
    uintptr_t address;
    size_t size;
    int advice;
    int result;
};

static std::vector<advice_call> calls;

extern "C" int __real_posix_madvise(void *, size_t, int);
extern "C" int __wrap_posix_madvise(void * address, size_t size, int advice) {
    const int result = __real_posix_madvise(address, size, advice);
    calls.push_back({reinterpret_cast<uintptr_t>(address), size, advice, result});
    return result;
}

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void * mapping = mmap(nullptr, 4 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(mapping != MAP_FAILED, "mmap");
    auto * bytes = static_cast<unsigned char *>(mapping);
    const uintptr_t base = reinterpret_cast<uintptr_t>(mapping);

    struct example { size_t offset; size_t size; } examples[] = {
        {0, 256}, {256, 256}, {8, 8}, {page - 4, 8},
        {page, page}, {page + 32, 2 * page - 32},
    };
    for (const auto & example : examples) {
        void * address = bytes + example.offset;
        calls.clear();
        llama_prefetch_ranges(&address, &example.size, 1);
        require(calls.size() == 1, "one valid advice call");
        const auto & call = calls.front();
        require(call.result == 0, "real posix_madvise succeeded");
        require(call.address % page == 0, "start is page aligned");
        require(call.address <= base + example.offset, "start covers requested bytes");
        require(call.address + call.size == base + example.offset + example.size, "endpoint is preserved");
        require(call.advice == POSIX_MADV_WILLNEED, "advice is WILLNEED");
    }

    calls.clear();
    void * address = bytes + 8;
    const size_t zero = 0;
    llama_prefetch_ranges(nullptr, nullptr, 0);
    llama_prefetch_ranges(&address, &zero, 1);
    require(calls.empty(), "empty ranges do not issue advice");

    const size_t overflow = SIZE_MAX;
    llama_prefetch_ranges(&address, &overflow, 1);
    address = reinterpret_cast<void *>(UINTPTR_MAX - 3);
    const size_t eight = 8;
    llama_prefetch_ranges(&address, &eight, 1);
    require(calls.empty(), "overflow ranges do not issue advice");

    void * addresses[] = {bytes + 8, bytes + page + 256, bytes + 2 * page - 4};
    const size_t sizes[] = {8, 256, 8};
    llama_prefetch_ranges(addresses, sizes, 3);
    require(calls.size() == 3, "all batch ranges are issued");
    for (const auto & call : calls) {
        require(call.result == 0 && call.address % page == 0, "all batch calls succeed");
    }
    require(munmap(mapping, 4 * page) == 0, "munmap");
    std::printf("PASS: 6 single ranges, 3 batch ranges, empty/overflow guards; real POSIX calls; page=%zu\n", page);
}
