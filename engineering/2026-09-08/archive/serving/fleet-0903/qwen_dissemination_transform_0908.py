"""Add an opt-in load/store graph barrier to the private Qwen CPU runtime."""


def transform(original):
    text = original
    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old[:100]
        text = text.replace(old, new)

    replace('#include <omp.h>', '#include <omp.h>\n#include "flash-local-barrier-0908.h"')
    replace('    // synchronization primitives\n', '''#ifdef GGML_USE_OPENMP
    struct flash_local_barrier * dissemination;
#endif

    // synchronization primitives
''')
    replace('    struct ggml_threadpool * threadpool;\n    int ith;', '''    struct ggml_threadpool * threadpool;
    int ith;
#ifdef GGML_USE_OPENMP
    struct flash_local_cursor dissemination_cursor;
#endif''')
    replace('static struct ggml_state g_state = {0};', '''static struct ggml_state g_state = {0};

#ifdef GGML_USE_OPENMP
static bool qwen_dissemination_enabled = false;
static bool qwen_dissemination_audit = false;
static atomic_uint_fast64_t qwen_dissemination_calls;
GGML_BACKEND_API uint64_t ggml_cpu_dissemination_count(int ignored);
uint64_t ggml_cpu_dissemination_count(int ignored) {
    GGML_UNUSED(ignored);
    return atomic_load_explicit(&qwen_dissemination_calls, memory_order_relaxed);
}
#endif''')
    replace('#ifdef GGML_USE_OPENMP\n    #pragma omp barrier\n#else', '''#ifdef GGML_USE_OPENMP
    if (qwen_dissemination_enabled && n_threads <= FLASH_LOCAL_MAX_THREADS) {
        const int ith = omp_get_thread_num();
        GGML_ASSERT(tp->dissemination != NULL && tp->dissemination->threads == n_threads);
        GGML_ASSERT(ith < n_threads && omp_get_num_threads() == n_threads);
        if (qwen_dissemination_audit && ith == 0) {
            atomic_fetch_add_explicit(&qwen_dissemination_calls, 1, memory_order_relaxed);
        }
        flash_dissemination_wait(tp->dissemination, ith, &tp->workers[ith].dissemination_cursor);
        return;
    }
    #pragma omp barrier
#else''')
    replace('    ggml_aligned_free(threadpool->workers, workers_size);', '''#ifdef GGML_USE_OPENMP
    if (threadpool->dissemination != NULL) {
        ggml_aligned_free(threadpool->dissemination, sizeof(struct flash_local_barrier));
    }
#endif
    ggml_aligned_free(threadpool->workers, workers_size);''')
    replace('    struct ggml_threadpool    * tp    = state->threadpool;', '''    struct ggml_threadpool    * tp    = state->threadpool;
#ifdef GGML_USE_OPENMP
    state->dissemination_cursor.parity = 0;
    state->dissemination_cursor.sense = 1;
#endif''')
    replace('        threadpool->cgraph           = cgraph;\n        threadpool->cplan            = cplan;\n        threadpool->n_graph          = 0;', '''        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
#ifdef GGML_USE_OPENMP
        threadpool->dissemination    = NULL;
#endif
        threadpool->n_graph          = 0;''')
    replace('                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);', '''                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
                if (qwen_dissemination_enabled && n_threads <= FLASH_LOCAL_MAX_THREADS) {
                    if (threadpool->dissemination == NULL) {
                        threadpool->dissemination = ggml_aligned_malloc(sizeof(struct flash_local_barrier));
                    }
                    flash_local_barrier_init(threadpool->dissemination, n_threads);
                }''')
    marker = '        {\n            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
    replace(marker, '''#ifdef GGML_USE_OPENMP
        {
            const char * env = getenv("GGML_CPU_DISSEMINATION_BARRIER");
            qwen_dissemination_enabled = (env != NULL && strcmp(env, "1") == 0);
            env = getenv("GGML_CPU_DISSEMINATION_AUDIT");
            qwen_dissemination_audit = (env != NULL && strcmp(env, "1") == 0);
        }
#endif
''' + marker)
    return text
