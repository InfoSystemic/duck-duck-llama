#!/usr/bin/env python3
"""Check publication integrity and portable tests; never launch or contact a model."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from build_catalog import build_catalog, CATALOG, ROOT
from verify_engineering_snapshot import verify


def curated_documents():
    paths = list(ROOT.glob('*.md'))
    paths += list((ROOT / 'docs').rglob('*.md'))
    paths += [ROOT / directory / 'README.md' for directory in ('tools', 'engineering', 'profiles', 'benchmarks')]
    paths += list((ROOT / 'engineering').glob('*/README.md'))
    paths += list((ROOT / 'profiles').glob('*/README.md'))
    paths += [ROOT / 'patches' / 'README.md']
    return sorted(set(path for path in paths if path.is_file()))


def check_links():
    errors, count = [], 0
    for path in curated_documents():
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for target in re.findall(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)', text):
            target = target.strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            count += 1
            if not destination.is_relative_to(ROOT) or not destination.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing local link {target}')
    return {'documents': len(curated_documents()), 'local_links': count, 'errors': errors, 'passed': not errors}


def portable_tests():
    suites = [('tools', 'test_quality_and_decode.py'),
              ('engineering/2026-09-12/archive/serving/fleet-0912', 'test_ab_profiles.py')]
    reports = []
    for directory, pattern in suites:
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', directory, '-p', pattern, '-v']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = result.stdout + result.stderr
        match = re.search(r'Ran (\d+) tests?', output)
        count = int(match.group(1)) if match else 0
        report = {'directory': directory, 'pattern': pattern, 'tests': count,
                  'passed': result.returncode == 0 and count > 0}
        if not report['passed']:
            report['output'] = output
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    snapshots = {path.parent.name: verify(path.parent)
                 for path in sorted((ROOT / 'engineering').glob('*/archive-manifest.json'))}
    syntax_errors = []
    for path in sorted((ROOT / 'tools').glob('*.py')):
        try:
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except SyntaxError as error:
            syntax_errors.append(f'{path.relative_to(ROOT)}:{error.lineno}: {error.msg}')
    catalog_ok = CATALOG.exists() and json.loads(CATALOG.read_text()) == build_catalog()
    links = check_links()
    tests = portable_tests()
    report = {'scope': 'Published artifact integrity, source syntax, curated local documentation links, catalog consistency, and portable probe/controller tests. No full-model or hardware validation.',
              'snapshots': snapshots, 'tool_syntax_errors': syntax_errors,
              'catalog_passed': catalog_ok, 'documentation': links, 'portable_tests': tests}
    report['passed'] = (bool(snapshots) and all(row['passed'] for row in snapshots.values())
                        and not syntax_errors and catalog_ok and links['passed']
                        and all(row['passed'] for row in tests))
    output = json.dumps(report, indent=2) + '\n'
    if args.json:
        args.json.write_text(output)
    print(output, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
