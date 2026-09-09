#!/usr/bin/env python3
"""Behaviorally check a completed measured merge function in an isolated process."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from qwen_split_trial import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurement', type=Path)
    args = parser.parse_args()
    source = args.measurement.resolve()
    result_path = source.parent / 'generated-code-check.json'
    assert source.name == 'result.json' and not result_path.exists()
    measurement = json.loads(source.read_text())
    assert measurement['input_integrity_verified'] and not measurement.get('error')
    entry = next(x for x in measurement['measurements'] if x['kind'] == 'code')
    assert entry['completed_answer'], 'The measured sample did not complete an answer'
    chunks_path = source.parent / f'code-draft{entry["draft_n"]}' / 'chunks.json'
    chunks = json.loads(chunks_path.read_text())
    content = ''.join(choice.get('delta', {}).get('content') or '' for chunk in chunks for choice in chunk.get('choices', []))
    result = dict(started=time.time(), passed=False, measurement=str(source), measurement_sha256=sha256(source),
                  chunks_sha256=sha256(chunks_path), checker_sha256=sha256(__file__),
                  scope='Eight merge-function cases; not a broad model-quality comparison')
    try:
        candidates = []
        for block in re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, re.S):
            try:
                candidates += [node for node in ast.parse(block).body if isinstance(node, ast.FunctionDef) and len(node.args.args) == 2]
            except SyntaxError:
                pass
        assert candidates, 'No complete two-argument Python function was found'
        function = candidates[0]
        assert not function.decorator_list and not function.args.defaults and not function.args.kwonlyargs
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        for node in ast.walk(function):
            assert not isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef))
            if isinstance(node, ast.Name):
                assert not node.id.startswith('__')
            if isinstance(node, ast.Attribute):
                assert node.attr in ('append', 'extend', 'copy')
            if isinstance(node, ast.Call):
                assert isinstance(node.func, (ast.Name, ast.Attribute))
                if isinstance(node.func, ast.Name):
                    assert node.func.id in ('len', 'range', 'list', 'sorted')
        generated = ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])))
        cases = [([], []), ([], [1, 2]), ([1, 2], []), ([1, 3, 5], [2, 4, 6]),
                 ([1, 1, 3], [1, 2, 2]), ([-9, -3, 0], [-8, -2, 4]),
                 ([0], list(range(1, 30))), ([4, 4], [4, 4, 4])]
        script = ('import json,resource\nresource.setrlimit(resource.RLIMIT_CPU,(3,3))\n'
                  'resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))\n'
                  'namespace={"__builtins__":{"len":len,"range":range,"list":list,"sorted":sorted}}\n'
                  + 'exec(' + repr(generated) + ',namespace)\n'
                  + 'function=namespace[' + repr(function.name) + ']\n'
                  + 'cases=' + repr(cases) + '\n'
                  + 'for left,right in cases:\n a,b=left.copy(),right.copy()\n'
                    ' assert function(a,b)==sorted(left+right)\n assert (a,b)==(left,right)\n'
                    'print(json.dumps({"passed":True,"cases":len(cases)}))\n')
        completed = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, timeout=10)
        result.update(function=function.name, exit_code=completed.returncode, output=completed.stdout, stderr=completed.stderr)
        assert completed.returncode == 0, 'The generated function failed a behavioral check'
        result.update(passed=True, cases=len(cases))
        print(json.dumps(dict(passed=True, cases=len(cases), function=function.name)))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        result_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
