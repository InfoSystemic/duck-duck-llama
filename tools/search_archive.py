#!/usr/bin/env python3
"""Search published artifact paths without reading model data or contacting a server."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', nargs='?', default='', help='Space-separated terms; all must match the path')
    parser.add_argument('--kind', choices=('source', 'notes', 'evidence', 'patch'))
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / 'engineering/catalog.json').read_text())
    terms = args.query.casefold().split()
    matches = [row for row in catalog['artifacts']
               if (args.kind is None or row['kind'] == args.kind)
               and all(term in row['path'].casefold() for term in terms)]
    result = {'matches': len(matches), 'shown': min(len(matches), args.limit), 'artifacts': matches[:args.limit]}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f'{result["matches"]} matches; showing {result["shown"]}')
        for row in result['artifacts']:
            print(f'{row["kind"]:8} {row["path"]}')
    return 0 if matches else 1


if __name__ == '__main__':
    raise SystemExit(main())
