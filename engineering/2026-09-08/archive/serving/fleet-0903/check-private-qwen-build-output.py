#!/usr/bin/env python3
"""Check real build-recipe rewriting with a writer restricted to a temp directory."""
import importlib.util
from pathlib import Path
import tempfile


def main():
    base = Path(__file__).resolve().parent
    helper = base / 'build-private-qwen-split.py'
    spec = importlib.util.spec_from_file_location('private_qwen_split', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = base.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
    with tempfile.TemporaryDirectory(prefix='qwen-private-output-check-') as temp:
        destination = Path(temp) / 'private'
        outputs = []
        def isolated_writer(command, cwd, label):
            assert command.count('-o') == 1
            output = Path(command[command.index('-o') + 1])
            output = output if output.is_absolute() else Path(cwd) / output
            output = output.resolve()
            assert output.parent == destination.resolve(), (label, output)
            assert not output.exists()
            outputs.append(output.name)
            output.write_bytes(b'private output check; not an executable\n')
        result = module.build(engine, destination, isolated_writer)
        assert outputs == ['llama-model.cpp.o', 'libllama.so.0.3.0']
        assert Path(result['library']).resolve() == destination / outputs[-1]
        assert (destination / 'libllama.so').resolve() == Path(result['library'])
        print('PASS: compiler and linker outputs stay private; original engine inputs retain their hashes')


if __name__ == '__main__':
    main()
