"""Transform the private CPU graph executor without changing tensor arithmetic."""


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
    replace('static bool ggml_cpu_omp_simple_barrier = false;', '''static bool ggml_cpu_omp_simple_barrier = false;
static bool flash_dissemination_enabled = false;
static bool flash_dissemination_audit = false;
static atomic_uint_fast64_t flash_dissemination_calls;
GGML_BACKEND_API uint64_t ggml_cpu_dissemination_count(int ignored);
uint64_t ggml_cpu_dissemination_count(int ignored) {
    GGML_UNUSED(ignored);
    return atomic_load_explicit(&flash_dissemination_calls, memory_order_relaxed);
}''')
    replace('    if (!ggml_cpu_omp_simple_barrier) {', '''    if (flash_dissemination_enabled && n_threads <= FLASH_LOCAL_MAX_THREADS) {
        const int ith = omp_get_thread_num();
        GGML_ASSERT(tp->dissemination != NULL && tp->dissemination->threads == n_threads);
        GGML_ASSERT(ith < n_threads && omp_get_num_threads() == n_threads);
        if (flash_dissemination_audit && ith == 0) {
            atomic_fetch_add_explicit(&flash_dissemination_calls, 1, memory_order_relaxed);
        }
        flash_dissemination_wait(tp->dissemination, ith, &tp->workers[ith].dissemination_cursor);
        return;
    }
    if (!ggml_cpu_omp_simple_barrier) {''')
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
                // The preceding region has joined; this existing omp-single
                // barrier publishes fresh flags to the actual next team.
                if (flash_dissemination_enabled && n_threads <= FLASH_LOCAL_MAX_THREADS) {
                    if (threadpool->dissemination == NULL) {
                        threadpool->dissemination = ggml_aligned_malloc(sizeof(struct flash_local_barrier));
                    }
                    flash_local_barrier_init(threadpool->dissemination, n_threads);
                }''')
    replace('            ggml_cpu_omp_simple_barrier = (env != NULL && atoi(env) == 1);', '''            ggml_cpu_omp_simple_barrier = (env != NULL && atoi(env) == 1);
            env = getenv("GGML_CPU_DISSEMINATION_BARRIER");
            flash_dissemination_enabled = (env != NULL && strcmp(env, "1") == 0);
            env = getenv("GGML_CPU_DISSEMINATION_AUDIT");
            flash_dissemination_audit = (env != NULL && strcmp(env, "1") == 0);''')
    return text


def fixture_audit(source, require_calls=True):
    includes = '#include "ggml-cpu.h"\n#include <dlfcn.h>\n'
    marker = 'int main(int argc, char ** argv) {'
    assert source.count(marker) == 1
    extra = '''
    Dl_info barrier_runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &barrier_runtime)) std::abort();
    std::printf("BARRIER_CPU_LIBRARY %s\\n", barrier_runtime.dli_fname);
    std::atexit([] {
        using counter_fn = uint64_t (*)(int);
        const auto counter = reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_dissemination_count"));
        const uint64_t count = counter ? counter(0) : 0;
        const char * flag = std::getenv("GGML_CPU_DISSEMINATION_BARRIER");
        const char * audit = std::getenv("GGML_CPU_DISSEMINATION_AUDIT");
        const bool enabled = flag && flag[0] == '1';
        const bool auditing = audit && audit[0] == '1';
        std::printf("DISSEMINATION_CALLS %llu\\n", (unsigned long long) count);
        if (auditing && ((!enabled && count != 0) || (enabled && REQUIRE_CALLS && count == 0))) std::abort();
    });
'''.replace('REQUIRE_CALLS', 'true' if require_calls else 'false')
    return includes + source.replace(marker, marker + extra)
