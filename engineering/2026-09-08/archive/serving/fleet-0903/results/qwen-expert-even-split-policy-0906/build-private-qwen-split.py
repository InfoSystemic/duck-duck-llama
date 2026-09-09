#!/usr/bin/env python3
"""Stage an opt-in even Qwen expert split in a private libllama."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(engine, destination, run):
    engine, destination = Path(engine), Path(destination)
    destination.mkdir(exist_ok=False)
    pinned = engine / 'validated-iq-batch3-bin'
    source_path = engine / 'src/llama-model.cpp'
    source = source_path.read_text()
    assert 'GGML_Q4E_EXPERT_EVEN_SPLIT' not in source
    anchor = '''            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};'''
    replacement = '''            static const bool q4e_expert_even_split = [] {
                const char * value = getenv("GGML_Q4E_EXPERT_EVEN_SPLIT");
                return value && atoi(value) == 1;
            }();
            static const std::regex q4e_expert_weight("blk\\\\.\\\\d+\\\\.ffn_(gate|up|down)_exps\\\\.weight");
            const bool even_expert = q4e_expert_even_split && ud->model->arch == LLM_ARCH_QWEN4EXP &&
                ud->n_devices == 4 && hparams.n_ff_exp == 640 && blck_size == 32 &&
                std::regex_match(tensor_name, q4e_expert_weight);
            const int64_t blck_size_perf = std::lcm(blck_size, even_expert ? 16 : 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};'''
    assert source.count(anchor) == 1
    changed = source.replace(anchor, replacement)
    private_source = destination / 'llama-model.cpp'
    private_source.write_text(changed)
    (destination / 'qwen-expert-even-split.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='a/src/llama-model.cpp', tofile='b/src/llama-model.cpp')))
    commands_path = engine / 'build-goal/compile_commands.json'
    entries = [e for e in json.loads(commands_path.read_text()) if e['file'] == str(source_path)]
    assert len(entries) == 1
    entry = entries[0]
    cwd = Path(entry['directory'])
    command = shlex.split(entry['command'])
    command = [a.replace('\\"', '"') if a.startswith(('-DLLAMA_COMMIT=', '-DLLAMA_VERSION=')) else a for a in command]
    output_index = command.index('-o') + 1
    old_object = (cwd / command[output_index]).resolve()
    assert old_object.name == 'llama-model.cpp.o' and command.count(str(source_path)) == 1
    link_path = cwd / 'CMakeFiles/llama.dir/link.txt'
    link = shlex.split(link_path.read_text())
    link_output_index = link.index('-o') + 1
    library_name = Path(link[link_output_index]).name
    assert library_name == 'libllama.so.0.3.0'
    objects = [(cwd / a).resolve() for a in link if a.endswith('.o')]
    assert objects.count(old_object) == 1
    originals = [source_path, commands_path, link_path] + objects
    for folder in (engine / 'src', engine / 'include', engine / 'ggml/include'):
        originals += list(folder.glob('*.h')) + list(folder.glob('*.hpp'))
    originals += [p for p in pinned.glob('lib*.so.*') if p.is_file()]
    info = dict(input_sha256={str(p): digest(p) for p in originals},
                private_source_sha256=digest(private_source),
                scope='Private opt-in split policy; no production source, object, library, or service is modified.')
    manifest = destination / 'manifest.json'
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    obj = destination / 'llama-model.cpp.o'
    library = destination / library_name
    command[output_index] = str(obj)
    command[command.index(str(source_path))] = str(private_source)
    run(command, cwd, 'split-policy-compile.log')
    link[link_output_index] = str(library)
    for index, arg in enumerate(link):
        if arg.endswith('.o'):
            original = (cwd / arg).resolve()
            link[index] = str(obj if original == old_object else original)
        elif arg.startswith('-Wl,-rpath,'):
            link[index] = '-Wl,-rpath,' + str(pinned)
        elif '.so.' in arg and not arg.startswith('-'):
            link[index] = str(pinned / Path(arg).name)
    run(link, cwd, 'split-policy-link.log')
    (destination / 'libllama.so.0').symlink_to(library.name)
    (destination / 'libllama.so').symlink_to('libllama.so.0')
    info.update(library=str(library), library_sha256=digest(library),
                object_sha256=digest(obj), compile_command=command, link_command=link)
    assert all(digest(p) == value for p, value in info['input_sha256'].items())
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    return info
