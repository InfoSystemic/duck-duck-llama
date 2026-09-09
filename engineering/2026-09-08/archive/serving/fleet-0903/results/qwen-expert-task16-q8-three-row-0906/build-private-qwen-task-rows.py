#!/usr/bin/env python3
"""Build an opt-in IQ expert scheduler in a private CPU library."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(source):
    start = source.index('    void forward_mul_mat_id_swiglu(\n')
    end = source.index('    bool compute_forward_mul_mat_id_weighted_sum(', start)
    old = source[start:end]
    assert 'GGML_CPU_IQ_MOE_TASK_ROWS' not in source
    barrier = '''        ggml_barrier(params->threadpool);

        int active_nth = nth;
        if (n_tokens == 1) {
            const int requested = moe_single_token_threads();
            if (requested > 0) {
                active_nth = std::min(active_nth, requested);
            }
        }
'''
    replacement = '''        int active_nth = nth;
        if (n_tokens == 1) {
            const int requested = moe_single_token_threads();
            if (requested > 0) {
                active_nth = std::min(active_nth, requested);
            }
        }
        static const int requested_task_rows = [] {
            const char * value = getenv("GGML_CPU_IQ_MOE_TASK_ROWS");
            const int rows = value ? atoi(value) : 0;
            return rows == 16 || rows == 32 || rows == 64 ? rows : 0;
        }();
        const int task_rows = expanded_iq && NB_COLS == 16 && k == 2560 &&
            n_out > 0 && n_out <= 256 && n_out % 16 == 0 && n_tokens <= 3 ? requested_task_rows : 0;
        if (ith == 0 && task_rows) {
            ggml_threadpool_chunk_set(params->threadpool, active_nth);
            static std::atomic<bool> logged { false };
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                GGML_LOG_INFO("IQ_MOE_TASK_ROWS rows=%d n_out=%" PRId64 " n_tokens=%" PRId64 "\\n", task_rows, n_out, n_tokens);
            }
        }
        ggml_barrier(params->threadpool);
'''
    assert old.count(barrier) == 1
    changed = old.replace(barrier, replacement)
    loop = '''        for (int64_t expert = 0; expert < n_experts; ++expert) {
            const int64_t n_rows = matrix_row_counts[expert];
            if (n_rows == 0) {
                continue;
            }

'''
    begin = changed.index('        int64_t row_start = (ith * n_out) / active_nth;')
    loop_start = changed.index(loop, begin)
    assert changed.endswith('        }\n    }\n\n')
    body = changed[loop_start + len(loop):-len('        }\n    }\n\n')]
    assert body.count('                    continue;') == 1
    body = body.replace('                    continue;', '                    return;')
    scheduling = '''        auto compute_rows = [&](int64_t expert, int64_t row_start, int64_t row_end) {
            const int64_t n_rows = matrix_row_counts[expert];
''' + body + '''        };

        if (task_rows) {
            const int64_t tiles = (n_out + task_rows - 1) / task_rows;
            int64_t current = ith;
            int64_t base = 0;
            for (int64_t expert = 0; expert < n_experts; ++expert) {
                if (matrix_row_counts[expert] == 0) {
                    continue;
                }
                const int64_t end = base + tiles;
                while (current < end) {
                    GGML_ASSERT(current >= base);
                    const int64_t row_start = (current - base) * task_rows;
                    compute_rows(expert, row_start, std::min(row_start + task_rows, n_out));
                    current = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
                base = end;
            }
            return;
        }

        int64_t row_start = GGML_PAD((ith * n_out) / active_nth, NB_COLS);
        int64_t row_end = std::min<int64_t>(GGML_PAD(((ith + 1) * n_out) / active_nth, NB_COLS), n_out);
        if (row_start < row_end) {
            for (int64_t expert = 0; expert < n_experts; ++expert) {
                if (matrix_row_counts[expert] > 0) {
                    compute_rows(expert, row_start, row_end);
                }
            }
        }
    }

'''
    changed = changed[:begin] + scheduling
    return source[:start] + changed + source[end:]


def build(engine, destination, run):
    engine, destination = Path(engine), Path(destination)
    destination.mkdir(exist_ok=False)
    pinned = engine / 'validated-iq-batch3-bin'
    source_path = engine / 'ggml/src/ggml-cpu/repack.cpp'
    source = source_path.read_text()
    changed = transform(source)
    private_source = destination / 'repack.cpp'
    private_source.write_text(changed)
    (destination / 'iq-expert-task-rows.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='a/ggml/src/ggml-cpu/repack.cpp', tofile='b/ggml/src/ggml-cpu/repack.cpp')))
    commands_path = engine / 'build-goal/compile_commands.json'
    entries = [e for e in json.loads(commands_path.read_text()) if e['file'] == str(source_path)]
    assert len(entries) == 1
    entry = entries[0]
    cwd = Path(entry['directory'])
    command = shlex.split(entry['command'])
    output_index = command.index('-o') + 1
    old_object = (cwd / command[output_index]).resolve()
    assert old_object.name == 'repack.cpp.o' and command.count(str(source_path)) == 1
    link_path = cwd / 'CMakeFiles/ggml-cpu.dir/link.txt'
    link = shlex.split(link_path.read_text())
    link_output_index = link.index('-o') + 1
    library_name = Path(link[link_output_index]).name
    assert library_name == 'libggml-cpu.so.0.22.0'
    objects = [(cwd / a).resolve() for a in link if a.endswith('.o')]
    assert objects.count(old_object) == 1
    originals = [source_path, commands_path, link_path] + objects
    for folder in (engine / 'ggml/src', engine / 'ggml/include'):
        originals += list(folder.rglob('*.h')) + list(folder.rglob('*.hpp'))
    originals += [p for p in pinned.glob('lib*.so.*') if p.is_file()]
    info = dict(input_sha256={str(p): digest(p) for p in originals},
                private_source_sha256=digest(private_source),
                scope='Private CPU library; no production source, object, library, or service is changed.')
    manifest = destination / 'manifest.json'
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    obj = destination / 'repack.cpp.o'
    library = destination / library_name
    command[output_index] = str(obj)
    command[command.index(str(source_path))] = str(private_source)
    assert command.count('-o') == 1 and Path(command[output_index]).resolve() == obj.resolve()
    assert obj.parent.resolve() == destination.resolve() and not obj.exists()
    run(command, cwd, 'task-rows-compile.log')
    link[link_output_index] = str(library)
    for index, arg in enumerate(link):
        if index == link_output_index:
            continue
        if arg.endswith('.o'):
            original = (cwd / arg).resolve()
            link[index] = str(obj if original == old_object else original)
        elif arg.startswith('-Wl,-rpath,'):
            link[index] = '-Wl,-rpath,' + str(pinned)
        elif '.so.' in arg and not arg.startswith('-') and '/build-goal/' in str((cwd / arg).resolve()):
            link[index] = str(pinned / Path(arg).name)
    assert link.count('-o') == 1 and Path(link[link_output_index]).resolve() == library.resolve()
    assert library.parent.resolve() == destination.resolve() and not library.exists()
    run(link, cwd, 'task-rows-link.log')
    (destination / 'libggml-cpu.so.0').symlink_to(library.name)
    (destination / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
    info.update(library=str(library), library_sha256=digest(library),
                object_sha256=digest(obj), compile_command=command, link_command=link)
    assert all(digest(p) == value for p, value in info['input_sha256'].items())
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    return info
