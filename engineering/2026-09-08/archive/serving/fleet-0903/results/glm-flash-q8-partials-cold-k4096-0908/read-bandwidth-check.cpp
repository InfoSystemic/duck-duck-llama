// Local DRAM calibration, independent of any model or production runtime.
// The timed loop reads every byte and checks an exact integer checksum.
#include <immintrin.h>
#include <linux/mempolicy.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef COLD_X16
#include "cold-x16-workload.h"
#endif

static double now() {
    timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) std::abort();
    return t.tv_sec + t.tv_nsec * 1e-9;
}

[[noreturn]] static void fail(const char * message) {
    std::perror(message);
    std::exit(1);
}

template<bool streaming>
__attribute__((noinline)) static uint64_t scan(const uint64_t * data, size_t count) {
    // Prevent a repeated scan from being hoisted or replaced with an old sum.
    asm volatile("" : : "r"(data) : "memory");
    __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
    __m512i s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    auto load = [](const uint64_t * p) {
        if constexpr (streaming) return _mm512_stream_load_si512((void *) p);
        else return _mm512_load_si512(p);
    };
    for (size_t i = 0; i < count; i += 64) {
        s0 = _mm512_add_epi64(s0, load(data + i));
        s1 = _mm512_add_epi64(s1, load(data + i + 8));
        s2 = _mm512_add_epi64(s2, load(data + i + 16));
        s3 = _mm512_add_epi64(s3, load(data + i + 24));
        s4 = _mm512_add_epi64(s4, load(data + i + 32));
        s5 = _mm512_add_epi64(s5, load(data + i + 40));
        s6 = _mm512_add_epi64(s6, load(data + i + 48));
        s7 = _mm512_add_epi64(s7, load(data + i + 56));
    }
    const auto s = _mm512_add_epi64(
        _mm512_add_epi64(_mm512_add_epi64(s0, s1), _mm512_add_epi64(s2, s3)),
        _mm512_add_epi64(_mm512_add_epi64(s4, s5), _mm512_add_epi64(s6, s7)));
    alignas(64) uint64_t lanes[8];
    _mm512_store_si512(lanes, s);
    uint64_t sum = 0;
    for (auto x : lanes) sum += x;
    return sum;
}

struct alignas(64) Worker {
    int cpu = 0, node = 0;
    uint64_t * data = nullptr;
    uint64_t passes = 0, checksum = 0, expected = 0;
    size_t pass_bytes = 0;
    double start = 0, end = 0;
};

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: read-bandwidth-check cached|stream workers_per_socket MiB_per_worker seconds\n");
        return 2;
    }
    const std::string mode = argv[1];
    const int per_socket = std::stoi(argv[2]), mib = std::stoi(argv[3]);
    const double seconds = std::stod(argv[4]);
#ifdef COLD_X16
    cold_x16::configure();
    cold_x16::function(mode);
#else
    if (mode != "cached" && mode != "stream") return 2;
#endif
    if (per_socket < 1 || per_socket > 15 ||
        mib < 16 || mib > 64 || seconds < 3 || seconds > 30) return 2;
    const size_t bytes = size_t(mib) * 1024 * 1024, count = bytes / sizeof(uint64_t);
    const size_t page_size = size_t(sysconf(_SC_PAGESIZE));
    std::vector<Worker> workers(4 * per_socket);
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable ready_cv, go_cv;
    size_t ready = 0;
    bool go = false;
    std::atomic<bool> stop{false};
    for (size_t i = 0; i < workers.size(); ++i) {
        auto & w = workers[i];
        w.node = int(i) / per_socket;
        w.cpu = w.node * 16 + int(i) % per_socket;
        threads.emplace_back([&, i]() {
            auto & item = workers[i];
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(item.cpu, &mask);
            if (sched_setaffinity(0, sizeof(mask), &mask)) fail("worker affinity");
            item.data = (uint64_t *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (item.data == MAP_FAILED) fail("mmap");
            unsigned long nodes = 1UL << item.node;
            if (syscall(SYS_mbind, item.data, bytes, MPOL_BIND, &nodes, 8 * sizeof(nodes), 0)) fail("mbind");
            if (madvise(item.data, bytes, MADV_NOHUGEPAGE)) fail("madvise");
            const uint64_t seed = 101 + i;
#ifdef COLD_X16
            auto workload = std::make_unique<cold_x16::Workload>(item.data, bytes, seed, mode);
            item.expected = workload->scan();
            item.pass_bytes = workload->bytes_per_pass();
#else
            for (size_t j = 0; j < count; ++j) item.data[j] = j + seed;
            item.expected = uint64_t(count) * (count - 1) / 2 + uint64_t(count) * seed;
            item.pass_bytes = bytes;
#endif
            // Query sampled physical page locations, not just the binding policy.
            void * pages[64];
            int locations[64];
            for (size_t j = 0; j < 64; ++j) pages[j] = (char *) item.data +
                ((bytes / page_size - 1) * j / 63) * page_size;
            if (syscall(SYS_move_pages, 0, 64, pages, nullptr, locations, 0)) fail("query page placement");
            for (int location : locations) if (location != item.node) {
                std::fprintf(stderr, "Wrong physical node for worker %zu: %d != %d\n", i, location, item.node);
                std::exit(1);
            }
#ifndef COLD_X16
            if (scan<false>(item.data, count) != item.expected ||
                scan<true>(item.data, count) != item.expected) {
                std::fprintf(stderr, "Initial checksum failed\n");
                std::exit(1);
            }
#endif
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++ready;
                ready_cv.notify_one();
                go_cv.wait(lock, [&] { return go; });
            }
            item.start = now();
#ifdef COLD_X16
            while (!stop.load(std::memory_order_relaxed)) {
                item.checksum += workload->scan();
                ++item.passes;
            }
#else
            if (mode == "cached") {
                while (!stop.load(std::memory_order_relaxed)) {
                    item.checksum += scan<false>(item.data, count);
                    ++item.passes;
                }
            } else {
                while (!stop.load(std::memory_order_relaxed)) {
                    item.checksum += scan<true>(item.data, count);
                    ++item.passes;
                }
            }
#endif
            item.end = now();
        });
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        ready_cv.wait(lock, [&] { return ready == workers.size(); });
    }
    std::printf("{\"event\":\"ready\",\"pid\":%d,\"mode\":\"%s\",\"workers\":%zu,\"bytes_per_worker\":%zu,\"verified_page_samples\":%zu}\n",
                getpid(), mode.c_str(), workers.size(), bytes, workers.size() * 64);
    std::fflush(stdout);
    char command[16];
    if (!std::fgets(command, sizeof(command), stdin) || std::strcmp(command, "go\n")) return 2;
    const double start = now();
    {
        std::lock_guard<std::mutex> lock(mutex);
        go = true;
    }
    go_cv.notify_all();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true, std::memory_order_relaxed);
    for (auto & thread : threads) thread.join();
    const double end = now();
    uint64_t total_bytes = 0;
    bool exact = true;
    std::printf("{\"event\":\"done\",\"start\":%.9f,\"end\":%.9f,\"threads\":[", start, end);
    for (size_t i = 0; i < workers.size(); ++i) {
        const auto & w = workers[i];
        exact &= w.passes > 0 && w.checksum == w.expected * w.passes;
        total_bytes += w.passes * w.pass_bytes;
        std::printf("%s{\"cpu\":%d,\"node\":%d,\"passes\":%llu,\"start\":%.9f,\"end\":%.9f,\"checksum\":\"%016llx\"}",
                    i ? "," : "", w.cpu, w.node, (unsigned long long) w.passes, w.start, w.end,
                    (unsigned long long) w.checksum);
    }
    std::printf("],\"bytes\":%llu,\"logical_gb_s\":%.9f,\"checksums_exact\":%s}\n",
                (unsigned long long) total_bytes, total_bytes / (end - start) / 1e9, exact ? "true" : "false");
    std::fflush(stdout);
    if (!std::fgets(command, sizeof(command), stdin) || std::strcmp(command, "exit\n")) return 2;
    for (auto & w : workers) if (munmap(w.data, bytes)) fail("munmap");
    return exact ? 0 : 1;
}
