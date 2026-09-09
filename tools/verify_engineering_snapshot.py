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
    report = {'archive_files': 0, 'patches': 0, 'python_files': 0, 'shell_files': 0, 'errors': []}
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
    for name in ['source-bundles.json', 'selected-overlays.json', 'correctness-overlays.json']:
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
