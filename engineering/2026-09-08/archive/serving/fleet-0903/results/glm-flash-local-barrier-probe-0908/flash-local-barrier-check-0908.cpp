#include "flash-local-barrier-0908.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <omp.h>
#include <sched.h>
#include <thread>
#include <vector>

static_assert(sizeof(flash_local_flag) == 64, "Separate notification cache lines");
static const char * mode_names[] = {"openmp", "central", "dissemination", "tree"};

struct central_barrier {
    alignas(64) std::atomic<int> arrived{0};
    alignas(64) std::atomic<int> passed{0};
    void init() { arrived.store(0); passed.store(0); }
    void wait(int nth) {
        if (nth == 1) return;
        const int generation = passed.load(std::memory_order_relaxed);
        if (arrived.fetch_add(1, std::memory_order_seq_cst) == nth - 1) {
            arrived.store(0, std::memory_order_relaxed);
            passed.fetch_add(1, std::memory_order_seq_cst);
        } else {
            while (passed.load(std::memory_order_relaxed) == generation) _mm_pause();
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
    }
};

struct barriers {
    flash_local_barrier local;
    central_barrier central;
    void init(int nth) { flash_local_barrier_init(&local, nth); central.init(); }
};

template<int mode> static inline void wait_at(barriers & b, int ith, int nth, flash_local_cursor & c) {
    if constexpr (mode == 0) {
        if (nth > 1) {
            #pragma omp barrier
        }
    } else if constexpr (mode == 1) {
        b.central.wait(nth);
    } else if constexpr (mode == 2) {
        flash_dissemination_wait(&b.local, ith, &c);
    } else {
        flash_tree_wait(&b.local, ith, &c);
    }
}

static void pin(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) { perror("affinity"); abort(); }
}

static uint64_t pattern(int iteration, int worker, int lane) {
    return uint64_t(iteration + 1) * UINT64_C(0x9e3779b97f4a7c15) ^
           uint64_t(worker + 3) * UINT64_C(0x94d049bb133111eb) ^
           uint64_t(lane + 7) * UINT64_C(0xbf58476d1ce4e5b9);
}

struct alignas(64) published_row { uint64_t values[8]; };

template<int mode> static uint64_t correctness(barriers & b, int nth, int iteration_count) {
    b.init(nth);
    std::array<published_row, FLASH_LOCAL_MAX_THREADS> data{};
    std::array<uint64_t, FLASH_LOCAL_MAX_THREADS> hashes{};
    std::atomic<int> failures{0};
    #pragma omp parallel num_threads(nth)
    {
        const int ith = omp_get_thread_num();
        if (omp_get_num_threads() != nth) abort();
        pin(ith);
        flash_local_cursor cursor{0, 1};
        uint64_t hash = 0;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            for (int delay = 0; delay < (iteration * 11 + ith * 3) % 57; ++delay) _mm_pause();
            for (int lane = 0; lane < 8; ++lane) data[ith].values[lane] = pattern(iteration, ith, lane);
            wait_at<mode>(b, ith, nth, cursor);
            for (int worker = 0; worker < nth; ++worker) {
                for (int lane = 0; lane < 8; ++lane) {
                    const uint64_t observed = data[worker].values[lane];
                    if (observed != pattern(iteration, worker, lane)) failures.fetch_add(1);
                    hash ^= observed + uint64_t(iteration * 3 + ith);
                }
            }
            // Readers finish before any worker overwrites its publication.
            wait_at<mode>(b, ith, nth, cursor);
            if (iteration % 3 == 0) wait_at<mode>(b, ith, nth, cursor);
        }
        hashes[ith] = hash;
    }
    pin(127);
    if (failures.load() != 0) abort();
    uint64_t hash = 0;
    for (int i = 0; i < nth; ++i) hash = hash * UINT64_C(0x9e3779b97f4a7c15) + hashes[i];
    return hash;
}

struct alignas(64) final_value { uint64_t value; };
struct timing_result { double ns; uint64_t hash; };

template<int mode> static timing_result timing(barriers & b, int base_cpu, int nth, int workload, int iterations) {
    b.init(nth);
    std::array<final_value, FLASH_LOCAL_MAX_THREADS> outputs{};
    double start = 0, end = 0;
    #pragma omp parallel num_threads(nth) shared(start, end)
    {
        const int ith = omp_get_thread_num();
        if (omp_get_num_threads() != nth) abort();
        pin(base_cpu + ith);
        flash_local_cursor cursor{0, 1};
        for (int warmup = 0; warmup < 100; ++warmup) wait_at<mode>(b, ith, nth, cursor);
        #pragma omp barrier
        if (ith == 0) start = omp_get_wtime();
        #pragma omp barrier
        uint64_t hash = 0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (workload) {
                __m512 value = _mm512_set1_ps(float((iteration % 11) + ith + 1));
                const __m512 multiplier = _mm512_set1_ps(.99999f);
                const __m512 addition = _mm512_set1_ps(.0001f);
                const int work = workload == 1 || ith == iteration % nth ? 256 : 4;
                for (int step = 0; step < work; ++step) value = _mm512_fmadd_ps(value, multiplier, addition);
                const float first = _mm_cvtss_f32(_mm512_castps512_ps128(value));
                uint32_t bits; memcpy(&bits, &first, sizeof(bits));
                hash = hash * UINT64_C(0x9e3779b97f4a7c15) + bits;
            }
            wait_at<mode>(b, ith, nth, cursor);
        }
        #pragma omp barrier
        if (ith == 0) end = omp_get_wtime();
        outputs[ith].value = hash;
    }
    pin(base_cpu);
    uint64_t hash = 0;
    for (int i = 0; i < nth; ++i) hash = hash * UINT64_C(0xbf58476d1ce4e5b9) + outputs[i].value;
    return {(end - start) * 1e9 / iterations, hash};
}

static timing_result run_timing(int mode, barriers & b, int base_cpu, int nth, int workload, int iterations) {
    switch (mode) {
        case 0: return timing<0>(b, base_cpu, nth, workload, iterations);
        case 1: return timing<1>(b, base_cpu, nth, workload, iterations);
        case 2: return timing<2>(b, base_cpu, nth, workload, iterations);
        case 3: return timing<3>(b, base_cpu, nth, workload, iterations);
        default: abort();
    }
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    omp_set_dynamic(0);
    pin(127);
    auto b = std::make_unique<barriers>();
    int cases = 0;
    for (int nth : {1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 15, 4, 1, 15}) {
        const uint64_t reference = correctness<0>(*b, nth, 400);
        for (int mode = 1; mode < 4; ++mode) {
            const uint64_t hash = mode == 1 ? correctness<1>(*b, nth, 400) :
                                  mode == 2 ? correctness<2>(*b, nth, 400) : correctness<3>(*b, nth, 400);
            if (hash != reference) abort();
            ++cases;
        }
    }
    printf("{\"event\":\"correctness\",\"passed\":true,\"comparisons\":%d,\"iterations_per_case\":400}\n", cases);
    // Four independent teams model four socket-local pools. Team leaders
    // coordinate outside timed regions so each measured arm overlaps its peers.
    for (int teams : {1, 4}) {
        central_barrier host_gate;
        std::array<std::vector<double>, 4 * 3 * 4> times;
        std::array<uint64_t, 4 * 3 * 4> hashes{};
        std::vector<std::thread> hosts;
        for (int team = 0; team < teams; ++team) hosts.emplace_back([&, team] {
            const int base_cpu = teams == 1 ? 48 : 16 * team;
            pin(base_cpu);
            auto pool = std::make_unique<barriers>();
            for (int workload = 0; workload < 3; ++workload) {
                for (int cycle = 0; cycle < 7; ++cycle) {
                    for (int mode : {0, 2, 3, 1, 1, 3, 2, 0}) {
                        host_gate.wait(teams);
                        auto sample = run_timing(mode, *pool, base_cpu, 15, workload, 4000);
                        const int index = (team * 3 + workload) * 4 + mode;
                        if (!times[index].empty() && hashes[index] != sample.hash) abort();
                        hashes[index] = sample.hash;
                        times[index].push_back(sample.ns);
                        host_gate.wait(teams);
                    }
                }
            }
        });
        for (auto & host : hosts) host.join();
        for (int team = 0; team < teams; ++team) for (int workload = 0; workload < 3; ++workload) {
            const uint64_t reference = hashes[(team * 3 + workload) * 4];
            for (int mode = 0; mode < 4; ++mode) {
                const int index = (team * 3 + workload) * 4 + mode;
                auto & values = times[index];
                if (hashes[index] != reference || values.size() != 14) abort();
                std::sort(values.begin(), values.end());
                printf("{\"event\":\"timing\",\"teams\":%d,\"team\":%d,\"threads\":15,\"workload\":%d,\"mode\":\"%s\",\"samples\":14,\"median_ns\":%.6f,\"p25_ns\":%.6f,\"p75_ns\":%.6f,\"hash\":\"%016lx\"}\n",
                       teams, team, workload, mode_names[mode], (values[6] + values[7]) / 2, values[3], values[10], (unsigned long) hashes[index]);
            }
        }
    }
    printf("{\"event\":\"done\",\"passed\":true}\n");
}
