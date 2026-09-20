#!/usr/bin/env python3
"""make_dispatch.py -- ggml-cpu.f18.cpp: one worker team per NUMA device, shared by every backend instance (GGML_CPU_NUMA_SHARED_TEAM=1).

Every llama_context creates its own backend per CPU-NUMA device, and every such backend owns a dispatcher thread that is the
master of its own OpenMP team. The server has two contexts (trunk and MTP draft), so each core carries TWO pinned workers, one
per team, and the teams alternate: trunk graph, two draft graphs, trunk graph. A team that has just finished keeps spinning
at libgomp's dock (GOMP_SPINCOUNT, default 300000 x pause = ~15 ms here) on the very CPUs the other team now needs.
Measured on the 8-layer proxy: the first draft graph 8.3 ms against 5.9, the trunk graph 26.5 ms against 23.1.

With this switch all backends of a device hand their graphs to one dispatcher thread, so one team of 15 workers serves both
contexts: no second set of threads, nothing to hand over, and the idle spin keeps the workers hot for the next graph of
either context. Graphs of one device then run strictly in submission order, which is what the single-threaded server loop
does anyway; two threads driving two contexts concurrently could deadlock on cross-device collectives, hence opt-in.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/ggml-cpu.cpp')
s = SRC.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70]); s = s.replace(old, new)

rep('''struct ggml_backend_cpu_context {
    int                 n_threads;''', '''struct ggml_backend_cpu_shared_dispatcher;

struct ggml_backend_cpu_context {
    int                 n_threads;''')
rep('''    std::thread             async_worker;

    bool                owns_threadpool;
};
''', '''    std::thread             async_worker;

    // f18: when set, graphs go to the device-wide dispatcher instead of async_worker (GGML_CPU_NUMA_SHARED_TEAM=1)
    ggml_backend_cpu_shared_dispatcher * shared;

    bool                owns_threadpool;
};

// One dispatcher thread per NUMA device = one OpenMP team of pinned workers for all backends (contexts) of that device.
struct ggml_backend_cpu_shared_dispatcher {
    std::mutex                                        mutex;
    std::condition_variable                           cv;
    std::deque<std::pair<ggml_backend_t, ggml_cgraph *>> queue;
    std::thread                                       worker; // lives for the process
};
''')
rep('#include <condition_variable>\n', '#include <condition_variable>\n#include <deque>\n')

# free: a shared backend has no thread of its own; wait for its last graph
rep('''    if (cpu_ctx->async_enabled) {
        {
            std::lock_guard<std::mutex> lock(cpu_ctx->async_mutex);
            cpu_ctx->async_stop = true;
        }
        cpu_ctx->async_cv.notify_all();
        if (cpu_ctx->async_worker.joinable()) {
            cpu_ctx->async_worker.join();
        }
    }''', '''    if (cpu_ctx->async_enabled && cpu_ctx->shared) {
        std::unique_lock<std::mutex> lock(cpu_ctx->async_mutex);
        cpu_ctx->async_cv.wait(lock, [cpu_ctx]() {
            return !cpu_ctx->async_pending && !cpu_ctx->async_running;
        });
    } else if (cpu_ctx->async_enabled) {
        {
            std::lock_guard<std::mutex> lock(cpu_ctx->async_mutex);
            cpu_ctx->async_stop = true;
        }
        cpu_ctx->async_cv.notify_all();
        if (cpu_ctx->async_worker.joinable()) {
            cpu_ctx->async_worker.join();
        }
    }''')

# submit
rep('''    cpu_ctx->async_graph   = cgraph;
    cpu_ctx->async_status  = GGML_STATUS_SUCCESS;
    cpu_ctx->async_pending = true;
    lock.unlock();
    cpu_ctx->async_cv.notify_all();
    return GGML_STATUS_SUCCESS;
}''', '''    cpu_ctx->async_graph   = cgraph;
    cpu_ctx->async_status  = GGML_STATUS_SUCCESS;
    cpu_ctx->async_pending = true;
    lock.unlock();
    if (cpu_ctx->shared) {
        {
            std::lock_guard<std::mutex> shared_lock(cpu_ctx->shared->mutex);
            cpu_ctx->shared->queue.emplace_back(backend, cgraph);
        }
        cpu_ctx->shared->cv.notify_one();
        return GGML_STATUS_SUCCESS;
    }
    cpu_ctx->async_cv.notify_all();
    return GGML_STATUS_SUCCESS;
}''')

rep('''    ctx->async_status        = GGML_STATUS_SUCCESS;
''', '''    ctx->async_status        = GGML_STATUS_SUCCESS;
    ctx->shared              = nullptr;
''')

# init: shared dispatcher per device
rep('''    const std::vector<int> cpus = dev_ctx->cpus;
    cpu_ctx->async_worker = std::thread([backend, cpu_ctx, cpus]() {''', '''    const std::vector<int> cpus = dev_ctx->cpus;

    const char * shared_team_env = getenv("GGML_CPU_NUMA_SHARED_TEAM");
    if (shared_team_env && atoi(shared_team_env) != 0) {
        static std::mutex registry_mutex;
        static std::map<ggml_backend_dev_t, ggml_backend_cpu_shared_dispatcher *> registry;
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        auto & shared = registry[dev];
        if (!shared) {
            shared = new ggml_backend_cpu_shared_dispatcher;
            shared->worker = std::thread([shared, cpus]() {
#if defined(__linux__)
                cpu_set_t affinity;
                CPU_ZERO(&affinity);
                for (int cpu : cpus) {
                    if (cpu >= 0 && cpu < CPU_SETSIZE) {
                        CPU_SET(cpu, &affinity);
                    }
                }
                if (pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity) != 0) {
                    GGML_LOG_WARN("%s: failed to bind shared NUMA dispatcher thread\\n", __func__);
                }
#endif
                while (true) {
                    std::pair<ggml_backend_t, ggml_cgraph *> job;
                    {
                        std::unique_lock<std::mutex> lock(shared->mutex);
                        shared->cv.wait(lock, [shared]() { return !shared->queue.empty(); });
                        job = shared->queue.front();
                        shared->queue.pop_front();
                    }
                    auto * job_ctx = (ggml_backend_cpu_context *) job.first->context;
                    {
                        std::lock_guard<std::mutex> lock(job_ctx->async_mutex);
                        job_ctx->async_pending = false;
                        job_ctx->async_running = true;
                    }
                    const ggml_status status = ggml_backend_cpu_graph_compute_impl(job.first, job.second);
                    {
                        std::lock_guard<std::mutex> lock(job_ctx->async_mutex);
                        job_ctx->async_status  = status;
                        job_ctx->async_running = false;
                    }
                    job_ctx->async_cv.notify_all();
                }
            });
            shared->worker.detach();
            GGML_LOG_INFO("%s: %s uses one shared worker team for all contexts\\n", __func__, dev_ctx->name.c_str());
        }
        cpu_ctx->shared = shared;
        GGML_LOG_INFO("%s: initialized %s with %d threads (shared team)\\n", __func__, dev_ctx->name.c_str(), dev_ctx->n_threads);
        GGML_UNUSED(params);
        return backend;
    }

    cpu_ctx->async_worker = std::thread([backend, cpu_ctx, cpus]() {''')
(HERE/'ggml-cpu.f18.cpp').write_text(s)
print('wrote ggml-cpu.f18.cpp')
