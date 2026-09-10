"""Extend the existing opt-in graph profiler with monotonic stage timestamps."""


def transform(source):
    old = '    int64_t * profile_node_us = profile_ops ? calloc(cgraph->n_nodes, sizeof(int64_t)) : NULL;\n'
    assert source.count(old) == 1
    source = source.replace(old, old + '''    int64_t * profile_node_start = profile_ops ? calloc(cgraph->n_nodes, sizeof(int64_t)) : NULL;
    int64_t * profile_node_work_end = profile_ops ? calloc(cgraph->n_nodes, sizeof(int64_t)) : NULL;
    int64_t * profile_node_end = profile_ops ? calloc(cgraph->n_nodes, sizeof(int64_t)) : NULL;
    GGML_ASSERT(!profile_ops || (profile_node_us && profile_node_start && profile_node_work_end && profile_node_end));
    const int64_t profile_graph_start = profile_ops ? ggml_time_us() : 0;
''')
    old = '        if (state->ith == 0 && cplan->abort_callback &&\n'
    assert source.count(old) == 1
    source = source.replace(old, '''        if (profile_ops) {
            profile_node_start[measured_node_n] = profile_start_us;
            profile_node_work_end[measured_node_n] = ggml_time_us();
        }

''' + old)
    old = '            profile_node_us[measured_node_n] = ggml_time_us() - profile_start_us;'
    assert source.count(old) == 1
    source = source.replace(old, '''            profile_node_end[measured_node_n] = ggml_time_us();
            profile_node_us[measured_node_n] = profile_node_end[measured_node_n] - profile_start_us;''')
    old = '    if (profile_ops) {\n        int64_t profile_total_us = 0;'
    assert source.count(old) == 1
    source = source.replace(old, '''    if (profile_ops) {
        const int64_t profile_graph_end = ggml_time_us();
        GGML_LOG_WARN("CPU_OP_TIMELINE index=%d cpu=%d graph=%p threads=%d start_us=%" PRId64 " end_us=%" PRId64 "\\n",
            profile_graph_index, profile_cpu, (const void *) cgraph, n_threads_graph, profile_graph_start, profile_graph_end);
        int64_t profile_total_us = 0;''')
    old = '            if (profile_node_us[node_n] < 5) {'
    assert source.count(old) == 1
    source = source.replace(old, '            if (profile_node_start[node_n] == 0) {')
    old = '                "src0_type=%s src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_name=\'%s\'\\n",'
    assert source.count(old) == 1
    source = source.replace(old, '''                "src0_type=%s src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_name='%s' "
                "start_us=%" PRId64 " work_end_us=%" PRId64 " end_us=%" PRId64 " "
                "dst_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\\n",''')
    old = '                src0 != NULL ? src0->name : "");'
    assert source.count(old) == 1
    source = source.replace(old, '''                src0 != NULL ? src0->name : "",
                profile_node_start[node_n], profile_node_work_end[node_n], profile_node_end[node_n],
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);''')
    old = '        free(profile_node_us);'
    assert source.count(old) == 1
    source = source.replace(old, '''        free(profile_node_us);
        free(profile_node_start);
        free(profile_node_work_end);
        free(profile_node_end);''')
    return source
