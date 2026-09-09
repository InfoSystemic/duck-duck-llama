#pragma once

// Load/store barrier experiments following the dissemination and scalable-tree
// algorithms at https://www.cs.rochester.edu/research/synchronization/pseudocode/ss.html
// Acquire/release operations provide the C/C++ memory-order edges. This header
// contains no model arithmetic and is not enabled in the current runtime.
#ifdef __cplusplus
#include <atomic>
typedef std::atomic<int> flash_barrier_atomic;
#define FLASH_B_LOAD(p) (p)->load(std::memory_order_acquire)
#define FLASH_B_STORE(p, v) (p)->store((v), std::memory_order_release)
#define FLASH_B_INIT(p, v) (p)->store((v), std::memory_order_relaxed)
#else
#include <stdatomic.h>
typedef atomic_int flash_barrier_atomic;
#define FLASH_B_LOAD(p) atomic_load_explicit((p), memory_order_acquire)
#define FLASH_B_STORE(p, v) atomic_store_explicit((p), (v), memory_order_release)
#define FLASH_B_INIT(p, v) atomic_store_explicit((p), (v), memory_order_relaxed)
#endif

#include <assert.h>
#include <immintrin.h>

#define FLASH_LOCAL_MAX_THREADS 64
#define FLASH_LOCAL_ROUNDS 6

struct __attribute__((aligned(64))) flash_local_flag {
    flash_barrier_atomic value;
};

struct flash_local_node {
    struct flash_local_flag disseminate[2][FLASH_LOCAL_ROUNDS];
    struct flash_local_flag children[4];
    struct flash_local_flag wake;
};

struct flash_local_barrier {
    struct flash_local_node nodes[FLASH_LOCAL_MAX_THREADS];
    int threads;
    int rounds;
};

struct flash_local_cursor {
    int parity;
    int sense;
};

// The owner initializes only outside the parallel region, after the previous
// region has joined. Every participant starts that region at parity=0,sense=1.
static inline void flash_local_barrier_init(struct flash_local_barrier * b, int nth) {
    assert(nth >= 1 && nth <= FLASH_LOCAL_MAX_THREADS);
    b->threads = nth;
    b->rounds = 0;
    while ((1 << b->rounds) < nth) ++b->rounds;
    for (int i = 0; i < nth; ++i) {
        for (int parity = 0; parity < 2; ++parity) {
            for (int round = 0; round < b->rounds; ++round) {
                FLASH_B_INIT(&b->nodes[i].disseminate[parity][round].value, 0);
            }
        }
        for (int child = 0; child < 4; ++child) {
            FLASH_B_INIT(&b->nodes[i].children[child].value, 0);
        }
        FLASH_B_INIT(&b->nodes[i].wake.value, 0);
    }
}

static inline void flash_dissemination_wait(struct flash_local_barrier * b, int ith,
                                          struct flash_local_cursor * c) {
    for (int round = 0; round < b->rounds; ++round) {
        int partner = ith + (1 << round);
        if (partner >= b->threads) partner -= b->threads;
        FLASH_B_STORE(&b->nodes[partner].disseminate[c->parity][round].value, c->sense);
        while (FLASH_B_LOAD(&b->nodes[ith].disseminate[c->parity][round].value) != c->sense) {
            _mm_pause();
        }
    }
    if (c->parity) c->sense = !c->sense;
    c->parity = !c->parity;
}

static inline void flash_tree_wait(struct flash_local_barrier * b, int ith,
                                  struct flash_local_cursor * c) {
    // Four-way arrival tree. Every parent acquires all child publications
    // before forwarding its own arrival, making pre-barrier writes transitive.
    for (int child = 0; child < 4; ++child) {
        if (4 * ith + child + 1 >= b->threads) break;
        while (FLASH_B_LOAD(&b->nodes[ith].children[child].value) != c->sense) {
            _mm_pause();
        }
    }
    if (ith != 0) {
        FLASH_B_STORE(&b->nodes[(ith - 1) / 4].children[(ith - 1) % 4].value, c->sense);
        while (FLASH_B_LOAD(&b->nodes[ith].wake.value) != c->sense) {
            _mm_pause();
        }
    }
    // Binary wakeup tree. Wakeup reaches a child only after the root has
    // acquired every arrival. Sense reverses between completed barriers.
    for (int child = 2 * ith + 1; child <= 2 * ith + 2 && child < b->threads; ++child) {
        FLASH_B_STORE(&b->nodes[child].wake.value, c->sense);
    }
    c->sense = !c->sense;
}
