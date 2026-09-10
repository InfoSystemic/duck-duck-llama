#!/usr/bin/env python3
"""Export the checked scheduling source as an optional, disabled-by-default patch."""
import difflib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1] / 'engines/llama-llama-duck/engineering/2026-09-08'


def main():
    result_path = BASE / 'results/qwen-decode-scheduling-build-0909b/result.json'
    result = json.loads(result_path.read_text())
    assert result['passed'] and result['finished'] and result['expert_bit_exact'] and result['route_guard_only']
    for name in ('input_sha256', 'private_source_sha256'):
        assert all(sha256(path) == digest for path, digest in result[name].items())
    original = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/repack.cpp'
    private = result_path.parent / 'private-cpu'
    assert sha256(original) == '62ec23388c0a33d10bd1105aa8a17034d487bd42fd2c3cd6c430a1a8ef28d116'
    patch_path = ROOT / 'patches/qwen-decode-scheduling-ten-routes.patch'
    metadata_path = ROOT / 'experimental-overlays.json'
    assert not patch_path.exists() and not metadata_path.exists()
    records, parts = [], []
    for name in ('repack.cpp', 'qwen-q8-hc-ordered-k-0909.h'):
        target = 'ggml/src/ggml-cpu/' + name
        before = original.read_text() if name == 'repack.cpp' else ''
        after = (private / name).read_text()
        parts.append(f'diff --git a/{target} b/{target}\n')
        if not before:
            parts.append('new file mode 100644\n')
        parts.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile='a/'+target if before else '/dev/null', tofile='b/'+target))
        records.append(dict(path=target, before_sha256=sha256(original) if before else None,
                            after_sha256=sha256(private / name),
                            archived_source='archive/serving/fleet-0903/results/qwen-decode-scheduling-build-0909b/private-cpu/'+name))
    patch_path.write_text(''.join(parts))
    with tempfile.TemporaryDirectory(prefix='qwen-scheduling-source-0909b-') as temporary:
        tree = Path(temporary)
        parent = tree / 'ggml/src/ggml-cpu'
        parent.mkdir(parents=True)
        (parent / original.name).write_bytes(original.read_bytes())
        subprocess.run(['git', 'apply', '--check', str(patch_path)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(patch_path)], cwd=tree, check=True)
        assert all(sha256(tree / row['path']) == row['after_sha256'] for row in records)
    metadata = dict(name='qwen-decode-scheduling-ten-routes', patch=str(patch_path.relative_to(ROOT)),
                    patch_sha256=sha256(patch_path), files=records,
                    parent='After qwen-flash-next-goal-source.patch and qwen-q6-selected-overlay.patch; compatible with the separate gather correction.',
                    apply_check=True, reconstructed_files_identical=True, temporary_source_tree_only=True,
                    defaults_unchanged=True, runtime_promoted=False,
                    component_evidence='archive/serving/fleet-0903/results/qwen-decode-scheduling-build-0909b/result.json',
                    flags=dict(GGML_CPU_QWEN_HC_ORDERED_K='1', GGML_CPU_QWEN_HC_ROW_SPLIT='1', GGML_CPU_QWEN_Q6_MOE_TILE_ROWS='16'),
                    scope='Opt-in Q8 HC scheduling for K10240/nc64 and Q6 expert tiles for K2560/nc160/512 experts/10 routes. Exact reconstruction of these two source files, not a rebuild or full reconstruction of the experimental shared-dispatch/HC-normalization model runtime. Separate private overlays and evidence remain in the archive.')
    metadata_path.write_text(json.dumps([metadata], indent=2)+'\n')
    out = BASE / 'results/qwen-decode-scheduling-source-publication-0909b.json'
    with out.open('x') as handle:
        json.dump(dict(time=time.time(), passed=True, controller_sha256=sha256(__file__),
                       input_build_sha256=sha256(result_path), metadata=metadata), handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(passed=True, patch=metadata['patch'], patch_sha256=metadata['patch_sha256'],
                          reconstructed_files_identical=True, files=len(records)), indent=2))


if __name__ == '__main__':
    main()
