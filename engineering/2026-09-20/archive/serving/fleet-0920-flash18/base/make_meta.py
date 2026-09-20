#!/usr/bin/env python3
"""make_meta.py -- ggml-backend-meta.f18.cpp over the production glm-fix/ggml-backend-meta.cpp (libggml-base).

Two host-side costs of the Meta (tensor-parallel) backend on CPU-NUMA devices, found with strace and /proc sampling on 2026-09-20:

1. set_tensor on a CPU-NUMA buffer group starts one std::thread per device for EVERY call. That is right for loading weights and
   wrong for graph inputs: a decode cycle uploads ~8 small inputs for each of its 3 graphs, i.e. ~93 thread creations and exits per
   cycle (clone, 8 MB stack madvise, arena mmap, TLB shootdowns to the 60 busy worker CPUs, ~324 page faults per cycle on the main
   thread). Bit 1: uploads below GGML_META_F18_SET_TENSOR_PAR_MIN bytes (default 1 MiB) run in the calling thread.
2. The main thread and the per-device dispatch threads wait for each other by spinning: 300 us of pause, then sched_yield() for up
   to 20 ms, before they block. They are not pinned, so they spin on the hyperthread siblings of worker cores, and every barrier of
   the running graph waits for the slowed worker. On the 8-layer proxy the trunk graph is 22.95 ms with the default, 22.17 ms when
   both waits block at once. Bit 0: block immediately.

GGML_META_F18=<mask> sets the bits; GGML_META_F18_CONTROL_FILE maps a 4-byte word that overrides them in a running process (A/B).
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix/ggml-backend-meta.cpp')
s = SRC.read_text()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:70]); s = s.replace(old, new)

rep('''static bool ggml_backend_meta_buffer_is_cpu_numa_group(ggml_backend_buffer_t buffer) {''', '''// ---- f18 switches (see make_meta.py): bit 0 = dispatch waits block at once, bit 1 = small set_tensor runs in the caller
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
static uint32_t ggml_backend_meta_f18() {
    static const uint32_t env_mask = [] {
        const char * v = getenv("GGML_META_F18");
        return v ? (uint32_t) strtoul(v, nullptr, 0) : 0u;
    }();
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = getenv("GGML_META_F18_CONTROL_FILE");
        if (!path || !*path) {
            return nullptr;
        }
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return nullptr;
        }
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        return mapping == MAP_FAILED ? nullptr : (const uint32_t *) mapping;
    }();
    return control ? __atomic_load_n(control, __ATOMIC_ACQUIRE) : env_mask;
}
static size_t ggml_backend_meta_f18_set_tensor_par_min() {
    static const size_t v = [] {
        const char * e = getenv("GGML_META_F18_SET_TENSOR_PAR_MIN");
        return e ? (size_t) strtoull(e, nullptr, 0) : (size_t) 1 << 20;
    }();
    return v;
}

static bool ggml_backend_meta_buffer_is_cpu_numa_group(ggml_backend_buffer_t buffer) {''')

rep('''    auto for_each_buffer = [&](const auto & fn) {
        if (!ggml_backend_meta_buffer_is_cpu_numa_group(buffer)) {
            for (size_t j = 0; j < n_bufs; ++j) {
                fn(j);
            }
            return;
        }
''', '''    auto for_each_buffer = [&](const auto & fn) {
        // one thread per device pays off for weights, not for the handful of small graph inputs of every decode
        const bool small = (ggml_backend_meta_f18() & 2u) && size < ggml_backend_meta_f18_set_tensor_par_min();
        if (small || !ggml_backend_meta_buffer_is_cpu_numa_group(buffer)) {
            for (size_t j = 0; j < n_bufs; ++j) {
                fn(j);
            }
            return;
        }
''')

# main thread: collect
rep('''            while (dispatch_pending.load(std::memory_order_acquire) != 0) {
                if (yielding) {
                    sched_yield();
                } else {
                    dispatch_relax();
                }
                if ((++spins & 255) == 0 && !yielding && ggml_time_us() - t_start > dispatch_hard_spin_us) {
                    yielding = true;
                }
                if ((spins & 255) == 0 && ggml_time_us() - t_start > dispatch_spin_us) {''', '''            const bool block_now = (ggml_backend_meta_f18() & 1u) != 0;
            while (dispatch_pending.load(std::memory_order_acquire) != 0) {
                if (yielding) {
                    sched_yield();
                } else {
                    dispatch_relax();
                }
                if ((++spins & 255) == 0 && !yielding && ggml_time_us() - t_start > dispatch_hard_spin_us) {
                    yielding = true;
                }
                if (block_now || ((spins & 255) == 0 && ggml_time_us() - t_start > dispatch_spin_us)) {''')

# dispatch threads: wait for the next epoch
rep('''                        while (dispatch_epoch.load(std::memory_order_acquire) == observed_epoch &&
                               !dispatch_stop.load(std::memory_order_relaxed)) {
                            if (yielding) {
                                sched_yield();
                            } else {
                                dispatch_relax();
                            }
                            if ((++spins & 255) == 0) {
                                const int64_t el = ggml_time_us() - t_start;
                                if (!yielding && el > dispatch_hard_spin_us) {
                                    yielding = true;
                                }
                            }
                            if ((spins & 255) == 0 && ggml_time_us() - t_start > dispatch_spin_us) {''', '''                        const bool block_now = (ggml_backend_meta_f18() & 1u) != 0;
                        while (dispatch_epoch.load(std::memory_order_acquire) == observed_epoch &&
                               !dispatch_stop.load(std::memory_order_relaxed)) {
                            if (yielding) {
                                sched_yield();
                            } else {
                                dispatch_relax();
                            }
                            if ((++spins & 255) == 0) {
                                const int64_t el = ggml_time_us() - t_start;
                                if (!yielding && el > dispatch_hard_spin_us) {
                                    yielding = true;
                                }
                            }
                            if (block_now || ((spins & 255) == 0 && ggml_time_us() - t_start > dispatch_spin_us)) {''')
(HERE/'ggml-backend-meta.f18.cpp').write_text(s)
print('wrote ggml-backend-meta.f18.cpp')
