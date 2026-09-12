#!/usr/bin/env python3
"""Export source changes and filtered tuning evidence; never operate a model."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

SOURCE_SUFFIXES = {'.py', '.cpp', '.c', '.h', '.hpp', '.sh', '.patch', '.md',
                   '.cmake', '.jinja', '.env', '.sha256', '.inc', '.tsv'}
OMIT_KEYS = {
    'original_environment', 'original_env', 'environment', 'environ', 'env',
    'server_command', 'command', 'original_command', 'previous_command',
    'protected_server', 'protected', 'protected_before', 'protected_after',
    'peer', 'peer_service_after', 'current', 'before_state', 'after_state',
    'processes', 'process_details', 'process_snapshot', 'other_inference',
    'cmdline', 'cmd', 'argv', 'maps', 'numa_maps', 'smaps', 'status_text',
    'request', 'payload', 'messages', 'prompt', 'content', 'reasoning_content',
    'response', 'response_text', 'text', 'raw', 'stdout', 'stderr',
    'systemctl', 'journal', 'journalctl', 'cgroup', 'cgroups', 'proc',
    'api_key', 'authorization', 'password', 'access_token', 'secret',
    'host_before', 'host_after', 'inference_before', 'inference_after',
}
SAFE_ENV = re.compile(r'^(?:GGML_|LLAMA_|OMP_|GOMP_|KMP_|LD_LIBRARY_PATH$)')
ENGINE_ROOTS = {'common', 'conversion', 'docs', 'examples', 'ggml', 'gguf-py',
                'include', 'src', 'tests', 'tools', 'cmake'}
PUBLIC_BASES = {
    # Include the committed forward port as well as the uncommitted tuning.
    'llama.cpp-dspark-current': '876a4321163249c43ca4e986818fab5ab081f282',
}
PUBLIC_BASE_REPOSITORIES = {
    'llama.cpp-sr950-glm': 'https://github.com/ggml-org/llama.cpp.git',
    'llama.cpp-sr950-numa': 'https://github.com/ggml-org/llama.cpp.git',
    'llama.cpp-dspark-current': 'https://github.com/ggml-org/llama.cpp.git',
    'llama.cpp-dspark-official': 'https://github.com/ggml-org/llama.cpp.git',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean(value, counts):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key.lower() in OMIT_KEYS or key.lower().endswith(('cmdline', 'environ')):
                counts['removed_fields'] += 1
                continue
            if 'env' in key.lower() and isinstance(child, dict):
                out[key] = {k: v for k, v in child.items() if SAFE_ENV.match(k)}
            else:
                out[key] = clean(child, counts)
        return out
    if isinstance(value, list):
        if len(value) > 1500:
            counts['summarized_arrays'] += 1
            return {'publication_omitted_items': len(value),
                    'original_array_sha256': sha(json.dumps(value, sort_keys=True).encode())}
        return [clean(child, counts) for child in value]
    if isinstance(value, str) and len(value) > 4000:
        counts['summarized_strings'] += 1
        return {'publication_omitted_characters': len(value), 'original_string_sha256': sha(value.encode())}
    return value


def stable_bytes(path):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed while reading: {path}')
    return data


def git(root, *args, env=None):
    return subprocess.check_output(['git', '-C', str(root), *args], env=env)


def bundle(ai, out, tree_name, git_name, base):
    source, repository = ai / 'engines' / tree_name, ai / 'engines' / git_name
    cmd = ['git', '-C', str(repository), '--work-tree=' + str(source)]
    head = git(repository, 'rev-parse', 'HEAD').decode().strip()
    if base is None:
        base = PUBLIC_BASES.get(tree_name, head)
    with tempfile.TemporaryDirectory(prefix='llama-snapshot-index-') as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / 'index'))
        def run(*args):
            return subprocess.check_output([*cmd, *args], env=env)
        run('read-tree', base)
        changed = run('diff', '--name-only', '-z', base, '--').split(b'\0')
        extra = run('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
        paths = []
        for raw in sorted(set(changed + extra)):
            if not raw:
                continue
            name = raw.decode()
            p = Path(name)
            if any(x in name for x in ('.private.', '.before', '.bak', '.orig', '__pycache__')):
                continue
            if (p.parts[0] in ENGINE_ROOTS and p.suffix in SOURCE_SUFFIXES | {'.txt', '.json', '.yml', '.yaml'}) or name in {'CMakeLists.txt', 'Makefile', '.gitignore'}:
                paths.append(name)
        if paths:
            run('add', '-A', '--', *paths)
        patch = run('diff', '--cached', '--binary', '--full-index', base, '--')
        if not patch:
            return None
        expected = run('write-tree').decode().strip()
        changed_paths = [p.decode() for p in run('diff', '--cached', '--name-only', '-z', base).split(b'\0') if p]
        files = {name: sha(stable_bytes(source / name)) if (source / name).is_file() else None for name in changed_paths}
        dest = out / 'patches' / (tree_name + '.patch')
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(patch)
        run('read-tree', base)
        run('apply', '--cached', '--check', str(dest))
        run('apply', '--cached', '--whitespace=nowarn', str(dest))
        if run('write-tree').decode().strip() != expected:
            raise RuntimeError('Patch did not reconstruct ' + tree_name)
        conflicts = [name for name in changed_paths if (source / name).is_file()
                     and re.search(rb'^<<<<<<< ', (source / name).read_bytes(), re.MULTILINE)]
    remotes = git(repository, 'remote', '-v').decode().splitlines()
    public_remotes = sorted({line.split()[1] for line in remotes if 'https://github.com/' in line})
    if tree_name in PUBLIC_BASE_REPOSITORIES:
        public_remotes = sorted(set(public_remotes) | {PUBLIC_BASE_REPOSITORIES[tree_name]})
    unmerged = []
    for row in git(repository, 'ls-files', '-u', '-z').split(b'\0'):
        if not row:
            continue
        mode_sha_stage, raw_name = row.split(b'\t', 1)
        _, object_id, stage = mode_sha_stage.decode().split()
        name = raw_name.decode()
        raw = git(repository, 'cat-file', 'blob', object_id)
        raw.decode('utf-8')
        relative = Path('archive/engines') / tree_name / ('unmerged-stage-' + stage) / name
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        unmerged.append({'path': relative.as_posix(), 'category': 'unresolved-merge-stage',
                         'source_path': name, 'stage': int(stage), 'object_id': object_id,
                         'original_sha256': sha(raw), 'sha256': sha(raw), 'bytes': len(raw), 'filtered': False})
    result = {'name': tree_name, 'source_tree': tree_name, 'base_commit': base,
              'source_head': head,
              'patch': str(dest.relative_to(out)), 'patch_sha256': sha(patch),
              'bytes': len(patch), 'files': files, 'git_tree': expected,
              'apply_check': True, 'reconstructed_tree_identical': True,
              'public_remotes': public_remotes, 'unresolved_conflict_files': conflicts,
              'unmerged_stages': unmerged,
              'runtime_validation': 'Historical source snapshot; no new full-model validation implied.'}
    print(json.dumps({'bundle': tree_name, 'files': len(files), 'bytes': len(patch), 'conflicts': conflicts}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ai-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    args = p.parse_args()
    ai, out, prior = args.ai_root.resolve(), args.output.resolve(), args.prior.resolve()
    out.mkdir(parents=True, exist_ok=True)
    previous = {entry['path']: entry for entry in json.loads((prior / 'archive-manifest.json').read_text())['files']}
    inventory, inherited, excluded, counts = [], [], collections.Counter(), collections.Counter()
    invalid_evidence = []
    roots = [ai / 'serving' / name for name in ('fleet-0903', 'fleet-0911', 'fleet-0912',
             'glm-sr950', 'glm53-flash', 'qwen38-flash-next', 'flash-models', 'dsv4-sr950')]
    roots += [ai / 'host-setup']
    inputs = []
    for root in roots:
        inputs += list(root.rglob('*')) if root.exists() else []
    inputs += list(ai.glob('*.md'))
    for source in sorted(inputs):
        if source.is_symlink() or not source.is_file():
            continue
        relative = source.relative_to(ai)
        if any(part in {'__pycache__', 'portal-assistant', 'build-xrdp', 'build'} for part in relative.parts) or '.private.' in str(relative):
            excluded['private_context_or_build_output'] += 1
            continue
        if any(v in source.name for v in ('close_busy_chrome', 'close_hc_control_chrome', '.bak')):
            excluded['ephemeral_helper_or_backup'] += 1
            continue
        is_source_json = (source.name in {'profile.json', 'link_command.json', 'this-host-baselines.json'}
                          and 'results' not in relative.parts)
        is_json = source.suffix == '.json' and source.stat().st_size <= 4_000_000
        is_text = (source.suffix in SOURCE_SUFFIXES or source.name in {'SHA256SUMS', 'kernel-oom.txt'}
                   or ('fleet-0911' in relative.parts and source.name in {
                       'restore-argv.txt', 'restore-env.txt', 'sweep1.txt', 'sweep2.txt', 'sweep3.txt'}))
        if not (is_text or is_json):
            excluded['weights_binaries_traces_or_generated_arrays'] += 1
            continue
        raw = stable_bytes(source)
        raw.decode('utf-8')
        target_name = 'archive/' + relative.as_posix()
        if target_name in previous and previous[target_name]['original_sha256'] == sha(raw):
            inherited.append({'path': target_name, 'original_sha256': sha(raw), 'prior_path': '../' + prior.name + '/' + target_name})
            continue
        filtered = is_json and not is_source_json
        if filtered:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                excluded['incomplete_or_invalid_json'] += 1
                invalid_evidence.append({'source': relative.as_posix(), 'sha256': sha(raw), 'error': str(error)})
                continue
            data = (json.dumps(clean(parsed, counts), indent=2) + '\n').encode()
        else:
            data = raw
        target = out / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if source.suffix in {'.py', '.sh'} and os.access(source, os.X_OK) else 0o644)
        inventory.append({'path': target_name, 'category': 'filtered-evidence' if filtered else 'engineering-source-or-notes',
                          'original_sha256': sha(raw), 'sha256': sha(data), 'bytes': len(data), 'filtered': filtered})
    old_bundles = json.loads((prior / 'source-bundles.json').read_text())
    aliases = {'llama.cpp-q4e-goal-0904': 'llama.cpp-qwen4exp-current',
               'llama.cpp-glm5n-goal-0904': 'llama.cpp-glm53-flash'}
    configs = [(b['source_tree'], aliases.get(b['source_tree'], b['source_tree']), b['base_commit']) for b in old_bundles]
    configs += [(name, name, None) for name in ('llama.cpp-dspark-combined', 'llama.cpp-dspark-current',
                'llama.cpp-dspark-dsv4', 'llama.cpp-dspark-official', 'llama.cpp-deepseek41-jigsaw-0912')]
    bundles = []
    for tree, repo, base in configs:
        result = bundle(ai, out, tree, repo, base)
        if result:
            bundles.append(result)
            inventory.extend(result['unmerged_stages'])
    manifest = {'date': '2026-09-12', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope': 'GLM-5.3 Full/Flash, Qwen3.8 Flash Next, DeepSeek V4/V4.1 CPU engineering through September 12',
                'policy': 'Source/configuration text is preserved. Evidence JSON removes private host context, environment dumps and response text. Prior unchanged sources are referenced. No weights, compiled artifacts or raw process/perf traces.',
                'files': inventory, 'inherited_files': inherited, 'excluded_counts': dict(excluded),
                'invalid_evidence': invalid_evidence, 'filter_counts': dict(counts)}
    (out / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'source-bundles.json').write_text(json.dumps(bundles, indent=2) + '\n')
    for name in ('selected-overlays.json', 'correctness-overlays.json', 'experimental-overlays.json'):
        (out / name).write_text('[]\n')
    print(json.dumps({'exported_files': len(inventory), 'inherited_files': len(inherited),
                      'exported_bytes': sum(row['bytes'] for row in inventory), 'bundles': len(bundles),
                      'excluded': dict(excluded), 'filter': dict(counts)}), flush=True)


if __name__ == '__main__':
    main()
