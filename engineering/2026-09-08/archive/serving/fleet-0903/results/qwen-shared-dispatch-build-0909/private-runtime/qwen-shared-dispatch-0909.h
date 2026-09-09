#pragma once

class qwen_shared_numa_dispatch {
    std::mutex submit_mutex;
    std::mutex wait_mutex;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> pending{0};
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_cgraph *> graphs;
    std::vector<ggml_status> statuses;
    std::vector<std::thread> workers;
    int64_t spin_us = 20000;
    int64_t hard_spin_us = 300;
    uint64_t group_id;

    static void relax() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(wait_mutex);
            stop.store(true, std::memory_order_relaxed);
        }
        start_cv.notify_all();
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void work(size_t rank) {
        uint64_t observed = 0;
        while (true) {
            const int64_t begun = ggml_time_us();
            int spins = 0;
            bool yielding = false;
            while (epoch.load(std::memory_order_acquire) == observed && !stop.load(std::memory_order_relaxed)) {
                if (yielding) {
                    sched_yield();
                } else {
                    relax();
                }
                if ((++spins & 255) == 0) {
                    const int64_t elapsed = ggml_time_us() - begun;
                    yielding = yielding || elapsed > hard_spin_us;
                    if (elapsed > spin_us) {
                        std::unique_lock<std::mutex> lock(wait_mutex);
                        start_cv.wait(lock, [this, observed]() {
                            return stop.load(std::memory_order_relaxed) || epoch.load(std::memory_order_acquire) != observed;
                        });
                        break;
                    }
                }
            }
            if (stop.load(std::memory_order_relaxed)) {
                return;
            }
            observed = epoch.load(std::memory_order_acquire);
            statuses[rank] = graphs[rank] != nullptr
                ? ggml_backend_graph_compute(backends[rank], graphs[rank]) : GGML_STATUS_SUCCESS;
            if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                {
                    std::lock_guard<std::mutex> lock(wait_mutex);
                }
                done_cv.notify_one();
            }
        }
    }

public:
    explicit qwen_shared_numa_dispatch(size_t count) :
        backends(count, nullptr), graphs(count, nullptr), statuses(count, GGML_STATUS_SUCCESS) {
        static std::atomic<uint64_t> next_id{1};
        group_id = next_id.fetch_add(1, std::memory_order_relaxed);
        const char * spin = getenv("GGML_CPU_NUMA_DISPATCH_SPIN_US");
        const char * hard = getenv("GGML_CPU_NUMA_DISPATCH_HARD_SPIN_US");
        if (spin != nullptr) {
            spin_us = atoll(spin);
        }
        if (hard != nullptr) {
            hard_spin_us = atoll(hard);
        }
        GGML_ASSERT(count > 1);
        workers.reserve(count);
        try {
            for (size_t rank = 0; rank < count; ++rank) {
                workers.emplace_back([this, rank]() { work(rank); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
        GGML_LOG_INFO("SHARED_NUMA_DISPATCH create group=%" PRIu64 " ranks=%zu\n", group_id, count);
    }

    ~qwen_shared_numa_dispatch() {
        shutdown();
        GGML_LOG_INFO("SHARED_NUMA_DISPATCH destroy group=%" PRIu64 "\n", group_id);
    }

    uint64_t id() const { return group_id; }

    ggml_status run(const std::vector<ggml_backend_t> & job_backends, const std::vector<ggml_cgraph *> & job_graphs) {
        // Keep all collective ranks in the same order across submitting contexts.
        std::lock_guard<std::mutex> submission(submit_mutex);
        GGML_ASSERT(job_backends.size() == workers.size() && job_graphs.size() == workers.size());
        GGML_ASSERT(pending.load(std::memory_order_acquire) == 0);
        backends = job_backends;
        graphs = job_graphs;
        std::fill(statuses.begin(), statuses.end(), GGML_STATUS_SUCCESS);
        pending.store((int) workers.size(), std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(wait_mutex);
        }
        start_cv.notify_all();
        const int64_t begun = ggml_time_us();
        int spins = 0;
        bool yielding = false;
        while (pending.load(std::memory_order_acquire) != 0) {
            if (yielding) {
                sched_yield();
            } else {
                relax();
            }
            if ((++spins & 255) == 0) {
                const int64_t elapsed = ggml_time_us() - begun;
                yielding = yielding || elapsed > hard_spin_us;
                if (elapsed > spin_us) {
                    std::unique_lock<std::mutex> lock(wait_mutex);
                    done_cv.wait(lock, [this]() { return pending.load(std::memory_order_acquire) == 0; });
                    break;
                }
            }
        }
        for (ggml_status status : statuses) {
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    static std::shared_ptr<qwen_shared_numa_dispatch> obtain(const std::vector<ggml_backend_dev_t> & devices) {
        static std::mutex registry_mutex;
        static std::map<std::vector<uintptr_t>, std::weak_ptr<qwen_shared_numa_dispatch>> registry;
        std::vector<uintptr_t> key;
        for (ggml_backend_dev_t device : devices) {
            key.push_back(reinterpret_cast<uintptr_t>(device));
        }
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto & entry = registry[key];
        auto group = entry.lock();
        if (!group) {
            group = std::make_shared<qwen_shared_numa_dispatch>(devices.size());
            entry = group;
        }
        return group;
    }
};
