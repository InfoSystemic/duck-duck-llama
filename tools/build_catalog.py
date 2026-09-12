#!/usr/bin/env python3
"""Index the latest logical source artifacts and the dated engine patches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'engineering/catalog.json'


def kind_of(entry):
    suffix = Path(entry['path']).suffix.lower()
    if suffix == '.md':
        return 'notes'
    if suffix == '.patch':
        return 'patch'
    return 'evidence' if entry.get('filtered') else 'source'


def build_catalog():
    sources, patches, snapshots = {}, {}, []
    for manifest_path in sorted((ROOT / 'engineering').glob('*/archive-manifest.json')):
        snapshot = manifest_path.parent
        snapshots.append(snapshot.name)
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest['files']:
            sources[entry['path']] = {
                'path': (snapshot / entry['path']).relative_to(ROOT).as_posix(),
                'kind': kind_of(entry), 'snapshot': snapshot.name,
                'sha256': entry['sha256'], 'filtered': entry['filtered'],
            }
        for name in ('source-bundles.json', 'selected-overlays.json',
                     'correctness-overlays.json', 'experimental-overlays.json'):
            index = snapshot / name
            if not index.exists():
                continue
            for entry in json.loads(index.read_text()):
                path = (snapshot / entry['patch']).relative_to(ROOT).as_posix()
                patches[path] = {'path': path, 'kind': 'patch', 'snapshot': snapshot.name,
                                 'sha256': entry['patch_sha256'], 'filtered': False}
    for path in sorted((ROOT / 'patches').glob('*.patch')):
        relative = path.relative_to(ROOT).as_posix()
        patches[relative] = {'path': relative, 'kind': 'patch', 'snapshot': 'legacy',
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'filtered': False}
    artifacts = sorted([*sources.values(), *patches.values()], key=lambda row: row['path'])
    return {'schema': 1, 'snapshots': snapshots,
            'scope': 'Latest published version of each logical archive path, plus dated engine and legacy integration patches. Prior versions remain in snapshots and Git history.',
            'counts': dict(sorted(Counter(row['kind'] for row in artifacts).items())),
            'artifacts': artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    expected = build_catalog()
    if args.write:
        CATALOG.write_text(json.dumps(expected, indent=2) + '\n')
    elif not CATALOG.exists() or json.loads(CATALOG.read_text()) != expected:
        parser.exit(1, 'Catalog is stale; run python3 tools/build_catalog.py --write\n')
    print(json.dumps({'artifacts': len(expected['artifacts']), 'counts': expected['counts'], 'passed': True}))


if __name__ == '__main__':
    main()
