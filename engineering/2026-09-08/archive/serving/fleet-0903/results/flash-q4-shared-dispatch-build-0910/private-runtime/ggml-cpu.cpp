#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <climits>
#    include <cstring>
#    include <linux/mempolicy.h>
#    include <pthread.h>
#    include <sched.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device);

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    if (ggml_backend_cpu_is_repack_buffer_type(buft)) {
        return true;
    }
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

static bool qwen_shared_numa_dispatch_enabled() {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CPU_NUMA_SHARED_DISPATCH");
        return value != nullptr && atoi(value) != 0;
    }();
    return enabled;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation

    // NUMA CPU backends use a persistent per-socket dispatcher for concurrent Meta tensor shards.
    bool                    async_enabled;
    bool                    async_stop;
    bool                    async_pending;
    bool                    async_running;
    ggml_cgraph *           async_graph;
    ggml_status             async_status;
    std::mutex              async_mutex;
    std::condition_variable async_cv;
    std::thread             async_worker;

    bool                owns_threadpool;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    if (cpu_ctx->async_enabled) {
        {
            std::lock_guard<std::mutex> lock(cpu_ctx->async_mutex);
            cpu_ctx->async_stop = true;
        }
        cpu_ctx->async_cv.notify_all();
        if (cpu_ctx->async_worker.joinable()) {
            cpu_ctx->async_worker.join();
        }
    }
    if (cpu_ctx->owns_threadpool && cpu_ctx->threadpool) {
        ggml_threadpool_free(cpu_ctx->threadpool);
        cpu_ctx->threadpool = nullptr;
    }
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute_impl(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    int n_threads = cpu_ctx->n_threads;
    static const char * threads_file = getenv("GGML_CPU_NUMA_THREADS_FILE");
    if (threads_file && cpu_ctx->owns_threadpool) {
        std::ifstream input(threads_file);
        int requested = 0;
        if (input >> requested && requested > 0) {
            n_threads = std::min(n_threads, requested);
        }
    }
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    if (!cpu_ctx->async_enabled) {
        return ggml_backend_cpu_graph_compute_impl(backend, cgraph);
    }

    std::unique_lock<std::mutex> lock(cpu_ctx->async_mutex);
    cpu_ctx->async_cv.wait(lock, [cpu_ctx]() {
        return !cpu_ctx->async_pending && !cpu_ctx->async_running;
    });
    cpu_ctx->async_graph   = cgraph;
    cpu_ctx->async_status  = GGML_STATUS_SUCCESS;
    cpu_ctx->async_pending = true;
    lock.unlock();
    cpu_ctx->async_cv.notify_all();
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cpu_synchronize(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    if (!cpu_ctx->async_enabled) {
        return;
    }

    std::unique_lock<std::mutex> lock(cpu_ctx->async_mutex);
    cpu_ctx->async_cv.wait(lock, [cpu_ctx]() {
        return !cpu_ctx->async_pending && !cpu_ctx->async_running;
    });
    if (cpu_ctx->async_status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: asynchronous CPU graph failed with status %d\n", __func__, cpu_ctx->async_status);
    }
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_cpu_synchronize,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

static ggml_backend_t ggml_backend_cpu_init_device(ggml_backend_dev_t device) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;
    ctx->async_enabled       = false;
    ctx->async_stop          = false;
    ctx->async_pending       = false;
    ctx->async_running       = false;
    ctx->async_graph         = nullptr;
    ctx->async_status        = GGML_STATUS_SUCCESS;
    ctx->owns_threadpool     = false;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ device,
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    return ggml_backend_cpu_init_device(ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0));
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    if (!ctx->owns_threadpool) {
        ctx->n_threads = n_threads;
    }
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

#if defined(__linux__)

struct ggml_backend_cpu_numa_buft_context {
    int         node;
    std::string name;
    bool        huge_pages;
};

struct ggml_backend_cpu_numa_buffer_context {
    void * data;
    size_t mapping_size;
};

static void * ggml_backend_cpu_numa_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_cpu_numa_buffer_context *) buffer->context;
    return ctx->data;
}

static void ggml_backend_cpu_numa_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_cpu_numa_buffer_context *) buffer->context;
    if (ctx) {
        if (ctx->data && ctx->mapping_size) {
            munmap(ctx->data, ctx->mapping_size);
        }
        delete ctx;
    }
}

static void ggml_backend_cpu_numa_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_numa_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_numa_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_numa_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    GGML_UNUSED(buffer);
    return false;
}

static void ggml_backend_cpu_numa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_cpu_numa_buffer_context *) buffer->context;
    memset(ctx->data, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_cpu_numa_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_cpu_numa_buffer_free,
    /* .get_base        = */ ggml_backend_cpu_numa_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_cpu_numa_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_numa_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_numa_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_cpu_numa_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_numa_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_cpu_numa_buft_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_cpu_numa_buft_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cpu_numa_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = (ggml_backend_cpu_numa_buft_context *) buft->context;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        GGML_LOG_ERROR("%s: failed to determine page size\n", __func__);
        return nullptr;
    }
    const size_t requested_size = size;
    const size_t mapping_size = ((std::max<size_t>(size, 1) + page_size - 1) / page_size) * page_size;
    void * data = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        GGML_LOG_ERROR("%s: mmap failed for %zu bytes: %s\n", __func__, mapping_size, strerror(errno));
        return nullptr;
    }

    unsigned long node_mask = 1UL << buft_ctx->node;
    const long mbind_rc = syscall(
        SYS_mbind, data, mapping_size, MPOL_BIND, &node_mask, sizeof(node_mask) * CHAR_BIT,
        MPOL_MF_STRICT | MPOL_MF_MOVE);
    if (mbind_rc != 0) {
        GGML_LOG_ERROR("%s: mbind to NUMA node %d failed for %zu bytes: %s\n",
            __func__, buft_ctx->node, mapping_size, strerror(errno));
        munmap(data, mapping_size);
        return nullptr;
    }

    if (buft_ctx->huge_pages) {
        if (madvise(data, mapping_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_WARN("%s: MADV_HUGEPAGE failed for NUMA node %d: %s\n",
                __func__, buft_ctx->node, strerror(errno));
        }
    }

    auto * buffer_ctx = new ggml_backend_cpu_numa_buffer_context { data, mapping_size };
    return ggml_backend_buffer_init(buft, ggml_backend_cpu_numa_buffer_iface, buffer_ctx, requested_size);
}

static size_t ggml_backend_cpu_numa_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static bool ggml_backend_cpu_numa_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

#endif // __linux__

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string name        = "CPU";
    std::string description = "CPU";
    int         numa_node   = -1;
    int         n_threads   = 0;
    int         poll        = 50;
    std::vector<int> cpus;
    ggml_backend_buffer_type_t buffer_type = nullptr;

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_cpu_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
#if defined(__linux__)
    auto * ctx = (ggml_backend_cpu_device_context *) dev->context;
    if (ctx->numa_node >= 0) {
        std::ifstream file("/sys/devices/system/node/node" + std::to_string(ctx->numa_node) + "/meminfo");
        std::string node_word;
        std::string node_number;
        std::string key;
        std::string unit;
        unsigned long long value_kb       = 0;
        unsigned long long total_kb       = 0;
        unsigned long long free_kb        = 0;
        unsigned long long file_pages_kb  = 0;
        unsigned long long shmem_kb       = 0;
        unsigned long long reclaimable_kb = 0;
        while (file >> node_word >> node_number >> key >> value_kb >> unit) {
            if (key == "MemTotal:") {
                total_kb = value_kb;
            } else if (key == "MemFree:") {
                free_kb = value_kb;
            } else if (key == "FilePages:") {
                file_pages_kb = value_kb;
            } else if (key == "Shmem:") {
                shmem_kb = value_kb;
            } else if (key == "SReclaimable:") {
                reclaimable_kb = value_kb;
            }
        }
        if (total_kb > 0) {
            const unsigned long long file_cache_kb = file_pages_kb > shmem_kb ? file_pages_kb - shmem_kb : 0;
            const unsigned long long available_kb =
                std::min(total_kb, free_kb + file_cache_kb + reclaimable_kb);
            *total = total_kb * 1024;
            *free  = available_kb * 1024;
            return;
        }
    }
#endif
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    auto * ctx = (ggml_backend_cpu_device_context *) dev->context;
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ctx->numa_node >= 0 && !qwen_shared_numa_dispatch_enabled(),
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ ctx->numa_node < 0,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    auto * dev_ctx = (ggml_backend_cpu_device_context *) dev->context;
    ggml_backend_t backend = ggml_backend_cpu_init_device(dev);
    if (!backend || dev_ctx->numa_node < 0) {
        return backend;
    }

    auto * cpu_ctx = (ggml_backend_cpu_context *) backend->context;
    ggml_threadpool_params tpp = ggml_threadpool_params_default(dev_ctx->n_threads);
    std::fill(std::begin(tpp.cpumask), std::end(tpp.cpumask), false);
    for (int cpu : dev_ctx->cpus) {
        if (cpu >= 0 && cpu < GGML_MAX_N_THREADS) {
            tpp.cpumask[cpu] = true;
        }
    }
    tpp.strict_cpu = true;
    tpp.poll       = dev_ctx->poll;

    cpu_ctx->n_threads       = dev_ctx->n_threads;
    cpu_ctx->threadpool      = ggml_threadpool_new(&tpp);
    cpu_ctx->owns_threadpool = true;
    cpu_ctx->async_enabled   = true;

    if (!cpu_ctx->threadpool) {
        ggml_backend_free(backend);
        return nullptr;
    }

    if (qwen_shared_numa_dispatch_enabled()) {
        cpu_ctx->async_enabled = false;
        GGML_LOG_INFO("%s: initialized %s with %d threads (caller dispatch)\n",
            __func__, dev_ctx->name.c_str(), dev_ctx->n_threads);
        GGML_UNUSED(params);
        return backend;
    }

    const std::vector<int> cpus = dev_ctx->cpus;
    cpu_ctx->async_worker = std::thread([backend, cpu_ctx, cpus]() {
#if defined(__linux__)
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        for (int cpu : cpus) {
            if (cpu >= 0 && cpu < CPU_SETSIZE) {
                CPU_SET(cpu, &affinity);
            }
        }
        if (pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity) != 0) {
            GGML_LOG_WARN("%s: failed to bind NUMA dispatcher thread\n", __func__);
        }
#endif
        while (true) {
            ggml_cgraph * graph = nullptr;
            {
                std::unique_lock<std::mutex> lock(cpu_ctx->async_mutex);
                cpu_ctx->async_cv.wait(lock, [cpu_ctx]() {
                    return cpu_ctx->async_stop || cpu_ctx->async_pending;
                });
                if (cpu_ctx->async_stop) {
                    return;
                }
                graph = cpu_ctx->async_graph;
                cpu_ctx->async_pending = false;
                cpu_ctx->async_running = true;
            }

            const ggml_status status = ggml_backend_cpu_graph_compute_impl(backend, graph);
            {
                std::lock_guard<std::mutex> lock(cpu_ctx->async_mutex);
                cpu_ctx->async_status  = status;
                cpu_ctx->async_running = false;
            }
            cpu_ctx->async_cv.notify_all();
        }
    });

    GGML_LOG_INFO("%s: initialized %s with %d threads\n",
        __func__, dev_ctx->name.c_str(), dev_ctx->n_threads);
    GGML_UNUSED(params);
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_cpu_device_context *) dev->context;
    return ctx->buffer_type ? ctx->buffer_type : ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    const auto * ctx = static_cast<const ggml_backend_cpu_device_context *>(device->context);
    if (ctx->numa_node >= 0) {
        static std::mutex mutex;
        static std::map<ggml_backend_dev_t, std::vector<ggml_backend_buffer_type_t>> buffers;
        std::lock_guard<std::mutex> lock(mutex);
        auto found = buffers.find(device);
        if (found == buffers.end()) {
            const auto name = ctx->name + "_REPACK";
            auto repack = ggml_backend_cpu_repack_buffer_type_from(
                ggml_backend_cpu_device_get_buffer_type(device), device, name.c_str());
            found = buffers.emplace(device, std::vector<ggml_backend_buffer_type_t>{repack, nullptr}).first;
        }
        return found->second.data();
    }
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        auto bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();
    return extra_bufts.data();
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_cpu_is_repack_buffer_type(buft)) {
        return ggml_backend_buft_get_device(buft) == dev;
    }
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

#if defined(__linux__)

static bool ggml_backend_cpu_env_flag(const char * name, bool fallback = false) {
    const char * value = getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    return strcmp(value, "0") != 0 && strcasecmp(value, "false") != 0 && strcasecmp(value, "no") != 0;
}

static int ggml_backend_cpu_env_int(const char * name, int fallback) {
    const char * value = getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        GGML_LOG_WARN("%s: ignoring invalid %s=%s\n", __func__, name, value);
        return fallback;
    }
    return (int) parsed;
}

struct ggml_backend_cpu_numa_comm_context {
    std::vector<ggml_backend_t> backends;
    std::mutex                  mutex;
    bool                        logged_first_reduce = false;
};

static void * ggml_backend_cpu_numa_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (!ggml_backend_cpu_env_flag("GGML_CPU_NUMA_DIRECT_ALLREDUCE") || n_backends < 2) {
        return nullptr;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_cpu(backends[i])) {
            return nullptr;
        }
        auto * dev_ctx = (ggml_backend_cpu_device_context *) ggml_backend_get_device(backends[i])->context;
        if (dev_ctx->numa_node < 0) {
            return nullptr;
        }
    }

    auto * ctx = new ggml_backend_cpu_numa_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    GGML_LOG_INFO("%s: enabled direct all-reduce for %zu NUMA CPU backends\n", __func__, n_backends);
    return ctx;
}

static void ggml_backend_cpu_numa_comm_free(void * comm_ctx) {
    delete (ggml_backend_cpu_numa_comm_context *) comm_ctx;
}

static bool ggml_backend_cpu_numa_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
    auto * ctx = (ggml_backend_cpu_numa_comm_context *) comm_ctx;
    if (!ctx || !tensors) {
        return false;
    }

    const size_t n_backends = ctx->backends.size();
    ggml_tensor * first = tensors[0];
    if (!first || !first->buffer || !first->data || first->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(first)) {
        return false;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_tensor * tensor = tensors[i];
        if (!tensor || !tensor->buffer || !tensor->data || tensor->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(tensor) || !ggml_are_same_shape(first, tensor) ||
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) !=
                    ggml_backend_get_device(ctx->backends[i])) {
            return false;
        }
    }

    const int64_t n_elements = ggml_nelements(first);
    if (n_elements == 0) {
        return true;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    for (ggml_backend_t backend : ctx->backends) {
        ggml_backend_synchronize(backend);
    }

    // Empty input slices disable computation but still allocate an output buffer.
    for (size_t i = 0; i < n_backends; ++i) {
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            memset(tensors[i]->data, 0, ggml_nbytes(tensors[i]));
        }
    }

    float * dst = (float *) first->data;
    if (n_backends == 2) {
        const float * src1 = (const float *) tensors[1]->data;
        for (int64_t i = 0; i < n_elements; ++i) {
            dst[i] += src1[i];
        }
    } else if (n_backends == 3) {
        const float * src1 = (const float *) tensors[1]->data;
        const float * src2 = (const float *) tensors[2]->data;
        for (int64_t i = 0; i < n_elements; ++i) {
            dst[i] = (dst[i] + src1[i]) + src2[i];
        }
    } else if (n_backends == 4) {
        const float * src1 = (const float *) tensors[1]->data;
        const float * src2 = (const float *) tensors[2]->data;
        const float * src3 = (const float *) tensors[3]->data;
        for (int64_t i = 0; i < n_elements; ++i) {
            dst[i] = (dst[i] + src1[i]) + (src2[i] + src3[i]);
        }
    } else {
        for (size_t j = 1; j < n_backends; ++j) {
            const float * src = (const float *) tensors[j]->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                dst[i] += src[i];
            }
        }
    }

    const size_t n_bytes = ggml_nbytes(first);
    for (size_t i = 1; i < n_backends; ++i) {
        memcpy(tensors[i]->data, dst, n_bytes);
    }

    if (!ctx->logged_first_reduce) {
        GGML_LOG_INFO("%s: using direct F32 all-reduce for %lld elements\n",
            __func__, (long long) n_elements);
        ctx->logged_first_reduce = true;
    }
    return true;
}

static bool ggml_backend_cpu_read_int(const std::string & path, int & value) {
    std::ifstream file(path);
    return (file >> value).good() || file.eof();
}

static std::vector<int> ggml_backend_cpu_parse_cpu_list(const std::string & path) {
    std::ifstream file(path);
    std::string value;
    if (!std::getline(file, value)) {
        return {};
    }

    std::vector<int> cpus;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const size_t dash = token.find('-');
        int first = 0;
        int last  = 0;
        try {
            first = std::stoi(token.substr(0, dash));
            last  = dash == std::string::npos ? first : std::stoi(token.substr(dash + 1));
        } catch (const std::exception &) {
            return {};
        }
        if (first < 0 || last < first) {
            return {};
        }
        for (int cpu = first; cpu <= last; ++cpu) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

static std::vector<int> ggml_backend_cpu_primary_cpus_for_node(int node) {
    const std::string node_path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
    const std::vector<int> node_cpus = ggml_backend_cpu_parse_cpu_list(node_path);
    if (node_cpus.empty()) {
        return {};
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool have_allowed = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;

    std::set<std::pair<int, int>> cores_seen;
    std::vector<int> primary_cpus;
    for (int cpu : node_cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE || (have_allowed && !CPU_ISSET(cpu, &allowed))) {
            continue;
        }
        const std::string topology = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        int package_id = node;
        int core_id    = cpu;
        ggml_backend_cpu_read_int(topology + "physical_package_id", package_id);
        ggml_backend_cpu_read_int(topology + "core_id", core_id);
        if (cores_seen.emplace(package_id, core_id).second) {
            primary_cpus.push_back(cpu);
        }
    }
    std::sort(primary_cpus.begin(), primary_cpus.end());
    return primary_cpus;
}

#endif // __linux__

struct ggml_backend_cpu_device_registry {
    std::vector<std::unique_ptr<ggml_backend_cpu_device_context>> contexts;
    std::vector<std::unique_ptr<ggml_backend_device>>             devices;
#if defined(__linux__)
    std::vector<std::unique_ptr<ggml_backend_cpu_numa_buft_context>> buft_contexts;
    std::vector<std::unique_ptr<ggml_backend_buffer_type>>           buffer_types;
#endif

    explicit ggml_backend_cpu_device_registry(ggml_backend_reg_t reg) {
        add_device(reg, -1, {}, 0, 0, false);

#if defined(__linux__)
        if (!ggml_backend_cpu_env_flag("GGML_CPU_NUMA_DEVICES")) {
            return;
        }

        const int poll = std::min(100, ggml_backend_cpu_env_int("GGML_CPU_NUMA_POLL", 50));
        const bool huge_pages = ggml_backend_cpu_env_flag("GGML_CPU_NUMA_HUGEPAGES");
        for (int node = 0; node < 64; ++node) {
            std::vector<int> cpus = ggml_backend_cpu_primary_cpus_for_node(node);
            if (cpus.empty()) {
                continue;
            }
            const int default_threads = std::min<int>(10, cpus.size());
            const int n_threads = std::min<int>(
                ggml_backend_cpu_env_int("GGML_CPU_NUMA_THREADS", default_threads), cpus.size());
            cpus.resize(n_threads);
            add_device(reg, node, cpus, n_threads, poll, huge_pages);
        }

        GGML_LOG_INFO("%s: exposed %zu NUMA CPU devices\n", __func__, devices.size() - 1);
#endif
    }

    void add_device(
            ggml_backend_reg_t reg,
            int node,
            std::vector<int> cpus,
            int n_threads,
            int poll,
            bool huge_pages) {
        auto ctx = std::make_unique<ggml_backend_cpu_device_context>();
        ctx->numa_node = node;
        ctx->cpus       = std::move(cpus);
        ctx->n_threads  = n_threads;
        ctx->poll       = poll;
        if (node >= 0) {
            ctx->name = "CPU-NUMA" + std::to_string(node);
            ctx->description += " / NUMA node " + std::to_string(node) + " / " +
                std::to_string(n_threads) + " physical cores";
        }

        auto device = std::make_unique<ggml_backend_device>(ggml_backend_device {
            /* .iface   = */ ggml_backend_cpu_device_i,
            /* .reg     = */ reg,
            /* .context = */ ctx.get(),
        });

#if defined(__linux__)
        if (node >= 0) {
            auto buft_ctx = std::make_unique<ggml_backend_cpu_numa_buft_context>(
                ggml_backend_cpu_numa_buft_context { node, "CPU_NUMA" + std::to_string(node), huge_pages });
            auto buft = std::make_unique<ggml_backend_buffer_type>(ggml_backend_buffer_type {
                /* .iface   = */ {
                    /* .get_name       = */ ggml_backend_cpu_numa_buft_get_name,
                    /* .alloc_buffer   = */ ggml_backend_cpu_numa_buft_alloc_buffer,
                    /* .get_alignment  = */ ggml_backend_cpu_numa_buft_get_alignment,
                    /* .get_max_size   = */ nullptr,
                    /* .get_alloc_size = */ nullptr,
                    /* .is_host        = */ ggml_backend_cpu_numa_buft_is_host,
                },
                /* .device  = */ device.get(),
                /* .context = */ buft_ctx.get(),
            });
            ctx->buffer_type = buft.get();
            buft_contexts.push_back(std::move(buft_ctx));
            buffer_types.push_back(std::move(buft));
        }
#else
        GGML_UNUSED(huge_pages);
#endif

        contexts.push_back(std::move(ctx));
        devices.push_back(std::move(device));
    }
};

static ggml_backend_cpu_device_registry & ggml_backend_cpu_devices(ggml_backend_reg_t reg) {
    static ggml_backend_cpu_device_registry registry(reg);
    return registry;
}

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_cpu_devices(reg).devices.size();
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto & registry = ggml_backend_cpu_devices(reg);
    GGML_ASSERT(index < registry.devices.size());
    return registry.devices[index].get();
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
#if defined(__linux__)
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) ggml_backend_cpu_numa_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) ggml_backend_cpu_numa_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) ggml_backend_cpu_numa_comm_allreduce_tensor;
    }
#endif
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
