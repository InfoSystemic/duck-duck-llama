#pragma once
// f18 feature switches. GGML_F18_FEATURES is a bit mask (decimal or 0x hex).
// GGML_F18_CONTROL_FILE names a 4-byte file that overrides the mask while the process runs (A/B without reload).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum : uint32_t {
    F18_FA_MQA = 1u << 0, // cell-split MQA flash attention for short query batches
    F18_TOPK   = 1u << 1, // nth_element top-k
};

static uint32_t f18_features() {
    static const uint32_t env_mask = [] {
        const char * v = std::getenv("GGML_F18_FEATURES");
        return v ? (uint32_t) std::strtoul(v, nullptr, 0) : 0u;
    }();
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = std::getenv("GGML_F18_CONTROL_FILE");
        if (!path || !*path) return nullptr;
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        GGML_ASSERT(fd >= 0 && "GGML_F18_CONTROL_FILE must exist");
        struct stat info;
        GGML_ASSERT(fstat(fd, &info) == 0 && info.st_size == (off_t) sizeof(uint32_t));
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        GGML_ASSERT(mapping != MAP_FAILED);
        return (const uint32_t *) mapping;
    }();
    return control ? __atomic_load_n(control, __ATOMIC_ACQUIRE) : env_mask;
}
