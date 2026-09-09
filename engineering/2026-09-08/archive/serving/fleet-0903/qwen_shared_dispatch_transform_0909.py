"""Prepare default-off shared dispatch without modifying the original checkout."""


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def cpu_source(text):
    text = replace_once(text, '// CPU backend - backend (stream)', '''static bool qwen_shared_numa_dispatch_enabled() {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CPU_NUMA_SHARED_DISPATCH");
        return value != nullptr && atoi(value) != 0;
    }();
    return enabled;
}

// CPU backend - backend (stream)''')
    text = replace_once(text, '/* .async                 = */ ctx->numa_node >= 0,',
                        '/* .async                 = */ ctx->numa_node >= 0 && !qwen_shared_numa_dispatch_enabled(),')
    text = replace_once(text, '    const std::vector<int> cpus = dev_ctx->cpus;\n    cpu_ctx->async_worker = std::thread', '''    if (qwen_shared_numa_dispatch_enabled()) {
        cpu_ctx->async_enabled = false;
        GGML_LOG_INFO("%s: initialized %s with %d threads (caller dispatch)\\n",
            __func__, dev_ctx->name.c_str(), dev_ctx->n_threads);
        GGML_UNUSED(params);
        return backend;
    }

    const std::vector<int> cpus = dev_ctx->cpus;
    cpu_ctx->async_worker = std::thread''')
    return text


def meta_source(text):
    text = replace_once(text, '#include <vector>\n', '#include <vector>\n\n#include "qwen-shared-dispatch-0909.h"\n')
    text = replace_once(text, '    bool                     parallel_cpu_numa = false;', '''    std::shared_ptr<qwen_shared_numa_dispatch> shared_cpu_numa_dispatch;
    std::vector<ggml_backend_t> shared_cpu_numa_backends;
    bool                     parallel_cpu_numa = false;''')
    text = replace_once(text, '    ggml_status dispatch_cpu_numa(const std::vector<ggml_cgraph *> & graphs) {', '''    ggml_status dispatch_cpu_numa(const std::vector<ggml_cgraph *> & graphs) {
        if (shared_cpu_numa_dispatch) {
            return shared_cpu_numa_dispatch->run(shared_cpu_numa_backends, graphs);
        }''')
    text = replace_once(text, '        dispatch_graphs.resize(n_devs, nullptr);', '''        const char * shared_env = getenv("GGML_CPU_NUMA_SHARED_DISPATCH");
        if (shared_env != nullptr && atoi(shared_env) != 0) {
            std::vector<ggml_backend_dev_t> devices;
            for (const backend_config & bc : backend_configs) {
                devices.push_back(ggml_backend_get_device(bc.backend));
                shared_cpu_numa_backends.push_back(bc.backend);
            }
            shared_cpu_numa_dispatch = qwen_shared_numa_dispatch::obtain(devices);
            GGML_LOG_INFO("SHARED_NUMA_DISPATCH attach group=%" PRIu64 " ranks=%zu\\n",
                shared_cpu_numa_dispatch->id(), n_devs);
            return;
        }

        dispatch_graphs.resize(n_devs, nullptr);''')
    text = replace_once(text, '    void stop_cpu_numa_dispatch() {', '''    void stop_cpu_numa_dispatch() {
        if (shared_cpu_numa_dispatch) {
            shared_cpu_numa_dispatch.reset();
            shared_cpu_numa_backends.clear();
            return;
        }''')
    return text
