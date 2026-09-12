#!/usr/bin/env python3
"""Build a private split-callback regression from real GLM Full GGUF metadata."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
ENGINE = ROOT / 'engines/llama.cpp-sr950-glm'
SOURCE = ENGINE / 'src/llama-model.cpp'
sys.path.insert(0, str(ENGINE / 'gguf-py'))
from gguf import GGUFReader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = SOURCE.read_text()
    marker = '    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {\n'
    assert original.count(marker) == 1
    addition = '''        if (ud->model->arch == LLM_ARCH_GLM_DSA &&
                (std::regex_match(tensor_name, pattern_attn_q_a_weight) ||
                 std::regex_match(tensor_name, pattern_attn_kv_a_weight) ||
                 std::regex_match(tensor_name, pattern_ffn_shexp_gate_up_weight) ||
                 std::regex_match(tensor_name, pattern_ffn_shexp_down_weight))) {
            // Match shared gate/up to down, including K-quant repacking blocks.
            GGML_ASSERT(segments.size() == 1);
            return {std::lcm(blck_size, int64_t(256))};
        }

'''
    patched = original.replace(marker, marker + addition)
    parent = BASE / 'llama-model.parent.cpp'
    candidate = BASE / 'llama-model.patched.cpp'
    parent.write_text(original)
    candidate.write_text(patched)
    (BASE / 'llama-model.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile='a/src/llama-model.cpp', tofile='b/src/llama-model.cpp')))

    first = Path('/models/GLM-5.3-GGUF/UD-Q4_K_XL/GLM-5.3-UD-Q4_K_XL-00001-of-00011.gguf')
    tensors, metadata, files = [], {}, []
    for index in range(1, 12):
        path = first.with_name(first.name.replace('00001-of', f'{index:05d}-of'))
        reader = GGUFReader(path, 'r')
        for name, field in reader.fields.items():
            if name.startswith('glm-dsa.') or name == 'general.architecture':
                metadata.setdefault(name, field.contents())
        for tensor in reader.tensors:
            tensors.append(dict(name=tensor.name, type=tensor.tensor_type.name,
                                type_id=int(tensor.tensor_type), shape=[int(n) for n in tensor.shape]))
        with path.open('rb') as stream:
            header_digest = hashlib.sha256(stream.read(reader.data_offset)).hexdigest()
        files.append(dict(path=str(path), size=path.stat().st_size,
                          header_bytes=reader.data_offset, header_sha256=header_digest))
    assert metadata['general.architecture'] == 'glm-dsa'
    assert len({row['name'] for row in tensors}) == len(tensors)
    (BASE / 'model-metadata.json').write_text(json.dumps(dict(metadata=metadata, tensors=tensors, files=files), indent=2) + '\n')
    with (BASE / 'tensor-metadata.tsv').open('w') as out:
        values = [metadata['glm-dsa.' + key] for key in (
            'block_count', 'embedding_length', 'feed_forward_length', 'expert_feed_forward_length',
            'attention.head_count', 'attention.head_count_kv', 'attention.key_length', 'attention.value_length')]
        values[0] -= metadata.get('glm-dsa.nextn_predict_layers', 0)
        out.write(' '.join(str(v) for v in values) + '\n')
        for row in tensors:
            dims = row['shape'] + [1] * (4 - len(row['shape']))
            out.write(' '.join(map(str, [row['name'], row['type_id'], *dims])) + '\n')

    begin = original.index('struct ggml_backend_meta_split_state llama_meta_device_get_split_state(')
    end = original.index('\nconst char * llm_type_name', begin)
    start_patched = patched.index('struct ggml_backend_meta_split_state llama_meta_device_get_split_state(')
    end_patched = patched.index('\nconst char * llm_type_name', start_patched)
    callbacks = original[begin:end].replace('llama_meta_device_get_split_state(', 'split_parent(', 1)
    callbacks += patched[start_patched:end_patched].replace('llama_meta_device_get_split_state(', 'split_candidate(', 1)
    cpp = BASE / 'split-callbacks.generated.inc'
    cpp.write_text(callbacks)
    binary = BASE / 'check-split-metadata'
    command = ['c++', '-std=c++17', '-O1', '-g0', '-Wall', '-Wextra',
               '-I' + str(ENGINE / 'ggml/include'), str(BASE / 'check_split_metadata.cpp'),
               '-L' + str(ENGINE / 'build-dev2/bin'), '-Wl,-rpath,' + str(ENGINE / 'build-dev2/bin'),
               '-lggml-base', '-o', str(binary)]
    subprocess.run(command, check=True)
    syntax_command = ['c++', '-std=c++17', '-fsyntax-only', '-O0',
                      '-DGGML_BACKEND_SHARED', '-DGGML_SHARED', '-DGGML_USE_CPU',
                      '-DLLAMA_BUILD', '-DLLAMA_SHARED', '-Dllama_EXPORTS',
                      '-I' + str(ENGINE / 'src'), '-I' + str(ENGINE / 'include'),
                      '-I' + str(ENGINE / 'ggml/include'), str(candidate)]
    subprocess.run(syntax_command, check=True)
    output = subprocess.check_output([str(binary), str(BASE / 'tensor-metadata.tsv')], text=True)
    result = json.loads(output)
    assert result['passed'] and result['parent_unequal_invalid_axis0'] > 0
    result.update(source_sha256={str(path): sha(path) for path in (
        SOURCE, parent, candidate, BASE / 'llama-model.patch', BASE / 'check_split_metadata.cpp',
        cpp, BASE / 'tensor-metadata.tsv', BASE / 'model-metadata.json', Path(__file__), binary,
        ENGINE / 'ggml/include/ggml.h', ENGINE / 'ggml/include/ggml-backend.h',
        ENGINE / 'build-dev2/bin/libggml-base.so')},
                  compile_command=command, syntax_command=syntax_command, full_translation_unit_syntax_passed=True,
                  checkpoint_payload_loaded=False, service_modified=False,
                  runtime_built=False, original_source_unchanged=sha(SOURCE) == sha(parent),
                  scope='Actual complete split callbacks compiled against real GGML structs/block sizes with metadata-only model scaffolding; all Full GGUF tensor shapes/types. No graph or model execution.')
    (BASE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output, end='')


if __name__ == '__main__':
    main()
