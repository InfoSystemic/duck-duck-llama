#pragma once
#include <array>
#include <cstdio>
#include <limits>

static bool glm_pool_cache_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_GLM_POOL_CACHE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return false;
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = std::getenv("GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE");
        if (!path || !*path) return nullptr;
        const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        GGML_ASSERT(fd >= 0);
        struct stat info;
        GGML_ASSERT(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size == sizeof(uint32_t));
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        GGML_ASSERT(mapping != MAP_FAILED);
        return (const uint32_t *) mapping;
    }();
    if (!control) return true;
    const uint32_t mode = __atomic_load_n(control, __ATOMIC_ACQUIRE);
    GGML_ASSERT(mode <= 1);
    return mode == 1;
}

static std::atomic<uint64_t> glm_pool_cache_counts[4]{};
static std::atomic<size_t> glm_pool_cache_reserved{0}, glm_pool_cache_peak{0};
static std::atomic<uint64_t> glm_pool_cache_evictions{0};
extern "C" uint64_t ggml_cpu_glm_pool_cache_stat(int index);
extern "C" uint64_t ggml_cpu_glm_pool_cache_stat(int index) {
    if (index >= 0 && index < 4) return glm_pool_cache_counts[index].load(std::memory_order_relaxed);
    if (index == 4) return glm_pool_cache_reserved.load(std::memory_order_relaxed);
    if (index == 5) return glm_pool_cache_peak.load(std::memory_order_relaxed);
    if (index == 6) return glm_pool_cache_evictions.load(std::memory_order_relaxed);
    return 0;
}
static bool glm_pool_cache_probe() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_GLM_POOL_CACHE_PROBE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static size_t glm_pool_cache_limit() {
    static const size_t limit = [] {
        const char * value = std::getenv("GGML_CPU_GLM_POOL_CACHE_MAX_MIB");
        if (!value || !*value) return size_t(4096) << 20;
        char * end = nullptr;
        const unsigned long mib = std::strtoul(value, &end, 10);
        GGML_ASSERT(end && *end == '\0' && mib >= 1 && mib <= 4096);
        return size_t(mib) << 20;
    }();
    return limit;
}
static bool glm_pool_cache_reserve(size_t bytes) {
    size_t current = glm_pool_cache_reserved.load(std::memory_order_relaxed);
    const size_t limit = glm_pool_cache_limit();
    do {
        if (current > limit || bytes > limit - current) return false;
    } while (!glm_pool_cache_reserved.compare_exchange_weak(current, current + bytes, std::memory_order_relaxed));
    size_t peak = glm_pool_cache_peak.load(std::memory_order_relaxed);
    while (peak < current + bytes && !glm_pool_cache_peak.compare_exchange_weak(peak, current + bytes, std::memory_order_relaxed)) {}
    return true;
}

#if defined(__AVX512F__) && defined(__AVX2__) && defined(__F16C__)
struct glm_pool_cache_record {
    ggml_fp16_t input[4][256];
    float output[128];
    uint64_t pool, epoch;
};
static_assert(sizeof(glm_pool_cache_record) == 2576, "pool record bound");
struct glm_pool_cache_entry {
    const void * kg = nullptr;
    const void * ape = nullptr;
    glm_pool_cache_record * records = nullptr;
    size_t capacity = 0;
    uint64_t used = 0, epoch = 0;
    uint32_t fp_control = 0;
    bool probs_first = false;
    float position[4][128]{};
};
struct glm_pool_cache_worker {
    std::array<glm_pool_cache_entry, 16> entries{};
    size_t bytes = 0;
    uint64_t clock = 0;
    void release(glm_pool_cache_entry & entry) {
        if (entry.records) {
            const size_t size = entry.capacity * sizeof(glm_pool_cache_record);
            std::free(entry.records);
            bytes -= size;
            glm_pool_cache_reserved.fetch_sub(size, std::memory_order_relaxed);
        }
        entry = glm_pool_cache_entry{};
    }
    ~glm_pool_cache_worker() { for (auto & entry : entries) release(entry); }
    glm_pool_cache_entry * get(const ggml_tensor * kg, const ggml_tensor * ape, bool probs_first, size_t count) {
        if (!count) return nullptr;
        glm_pool_cache_entry * entry = nullptr;
        for (auto & item : entries) if (item.kg == kg->data && item.ape == ape->data) {entry = &item; break;}
        if (!entry) {
            entry = &*std::min_element(entries.begin(), entries.end(), [](const auto & a, const auto & b) {return a.used < b.used;});
            if (entry->records) glm_pool_cache_evictions.fetch_add(1, std::memory_order_relaxed);
            release(*entry);
            entry->kg = kg->data; entry->ape = ape->data;
        }
        entry->used = ++clock;
        uint16_t x87_control;
        __asm__ volatile("fnstcw %0" : "=m"(x87_control));
        const uint32_t fp_control = (_mm_getcsr() & 0xffc0u) | (uint32_t(x87_control) << 16);
        bool changed = entry->epoch == 0 || entry->fp_control != fp_control || entry->probs_first != probs_first;
        for (int m = 0; m < 4; ++m) changed |= std::memcmp(entry->position[m], (const char *) ape->data + m * ape->nb[1], sizeof(entry->position[m])) != 0;
        if (changed) {
            GGML_ASSERT(entry->epoch != std::numeric_limits<uint64_t>::max());
            ++entry->epoch; entry->fp_control = fp_control; entry->probs_first = probs_first;
            for (int m = 0; m < 4; ++m) std::memcpy(entry->position[m], (const char *) ape->data + m * ape->nb[1], sizeof(entry->position[m]));
        }
        size_t capacity = 32;
        while (capacity < count && capacity < 2048) capacity *= 2;
        if (capacity > entry->capacity) {
            const size_t size = capacity * sizeof(glm_pool_cache_record);
            while (bytes + size > (size_t(64) << 20)) {
                glm_pool_cache_entry * victim = nullptr;
                for (auto & item : entries) if (&item != entry && item.records && (!victim || item.used < victim->used)) victim = &item;
                if (!victim) break;
                glm_pool_cache_evictions.fetch_add(1, std::memory_order_relaxed); release(*victim);
            }
            if (bytes + size <= (size_t(64) << 20) && glm_pool_cache_reserve(size)) {
                auto * records = (glm_pool_cache_record *) std::malloc(size);
                if (records) {
                    for (size_t i = 0; i < capacity; ++i) records[i].epoch = 0;
                    for (size_t i = 0; i < entry->capacity; ++i) {
                        const auto & old = entry->records[i];
                        if (old.epoch == entry->epoch) records[old.pool % capacity] = old;
                    }
                    const size_t old_size = entry->capacity * sizeof(glm_pool_cache_record);
                    std::free(entry->records);
                    glm_pool_cache_reserved.fetch_sub(old_size, std::memory_order_relaxed);
                    bytes += size - old_size; entry->records = records; entry->capacity = capacity;
                } else {
                    glm_pool_cache_reserved.fetch_sub(size, std::memory_order_relaxed);
                }
            }
        }
        return entry->capacity ? entry : nullptr;
    }
};
static thread_local glm_pool_cache_worker glm_pool_cache_local;
static bool glm_pool_cache_hit(const glm_pool_cache_entry * entry, const glm_pool_cache_record * record,
        const ggml_fp16_t * const cell[4], uint64_t pool) {
    if (record->epoch != entry->epoch || record->pool != pool) return false;
    for (int m = 0; m < 4; ++m) if (std::memcmp(record->input[m], cell[m], sizeof(record->input[m])) != 0) return false;
    return true;
}
static void glm_pool_cache_report(const ggml_compute_params * params, const ggml_tensor * members, uint64_t hits, uint64_t misses, uint64_t bypass) {
    if (!glm_pool_cache_probe()) return;
    glm_pool_cache_counts[0].fetch_add(1, std::memory_order_relaxed);
    glm_pool_cache_counts[1].fetch_add(hits, std::memory_order_relaxed);
    glm_pool_cache_counts[2].fetch_add(misses, std::memory_order_relaxed);
    glm_pool_cache_counts[3].fetch_add(bypass, std::memory_order_relaxed);
    if (params->ith == 0 && hits) {
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            std::fprintf(stderr, "GLM_POOL_CACHE_HIT name=%s hits=%llu misses=%llu bypass=%llu reserved=%zu\n", members->name,
                (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) bypass,
                glm_pool_cache_reserved.load(std::memory_order_relaxed));
        }
    }
}
#endif
