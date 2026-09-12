#!/usr/bin/env python3
"""Verify the published engineering inventory without loading or contacting a model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    inventory = json.loads((root / 'archive-manifest.json').read_text())
    report = {'archive_files': 0, 'inherited_files': 0, 'patches': 0,
              'python_files': 0, 'shell_files': 0, 'errors': []}
    for entry in inventory['files']:
        path = root / entry['path']
        if not path.is_file() or digest(path) != entry['sha256']:
            report['errors'].append('Missing or changed archive file: ' + entry['path'])
            continue
        report['archive_files'] += 1
        if not entry['filtered'] and entry['sha256'] != entry['original_sha256']:
            report['errors'].append('Original-source hash mismatch: ' + entry['path'])
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(), filename=entry['path'])
                report['python_files'] += 1
            except SyntaxError as error:
                report['errors'].append(f'{entry["path"]}:{error.lineno}: {error.msg}')
        elif path.suffix == '.sh':
            checked = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if checked.returncode:
                report['errors'].append(entry['path'] + ': ' + checked.stderr.strip())
            else:
                report['shell_files'] += 1
    prior_manifests = {}
    for entry in inventory.get('inherited_files', []):
        path = (root / entry['prior_path']).resolve()
        prior_root = path
        for _ in Path(entry['path']).parts:
            prior_root = prior_root.parent
        try:
            if prior_root not in prior_manifests:
                prior_inventory = json.loads((prior_root / 'archive-manifest.json').read_text())
                prior_manifests[prior_root] = {row['path']: row for row in prior_inventory['files']}
            prior_entry = prior_manifests[prior_root][entry['path']]
            if prior_entry['original_sha256'] != entry['original_sha256']:
                raise ValueError('Original-source hash differs from prior inventory')
            if not path.is_file() or digest(path) != prior_entry['sha256']:
                raise ValueError('Prior published file is missing or changed')
            report['inherited_files'] += 1
        except (OSError, KeyError, ValueError) as error:
            report['errors'].append(f'Invalid inherited file: {entry["prior_path"]}: {error}')
    for name in ['source-bundles.json', 'selected-overlays.json', 'correctness-overlays.json', 'experimental-overlays.json']:
        if name == 'experimental-overlays.json' and not (root / name).exists():
            continue
        for entry in json.loads((root / name).read_text()):
            path = root / entry['patch']
            if not path.is_file() or digest(path) != entry['patch_sha256']:
                report['errors'].append('Missing or changed patch: ' + entry['patch'])
            else:
                report['patches'] += 1
            if 'archived_source' in entry:
                source = root / entry['archived_source']
                if not source.is_file() or digest(source) != entry['after_sha256']:
                    report['errors'].append('Corrected source mismatch: ' + entry['archived_source'])
            if name == 'experimental-overlays.json':
                for record in entry['files']:
                    source = root / record['archived_source']
                    if not source.is_file() or digest(source) != record['after_sha256']:
                        report['errors'].append('Experimental source mismatch: ' + record['archived_source'])
    report['passed'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).resolve().parents[1] / 'engineering/2026-09-08')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    report = verify(args.snapshot)
    text = json.dumps(report, indent=2) + '\n'
    if args.json:
        args.json.write_text(text)
    print(text, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
