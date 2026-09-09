#!/usr/bin/env python3
"""Render a parameterized SR950 profile as a quoted shell command; do not launch it."""
import argparse
import json
from pathlib import Path
import shlex
from string import Template


def render(profile, values):
    bindings = dict(profile.get('defaults', {}))
    allowed = set(profile['required']) | set(bindings)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError('Unknown parameters: ' + ', '.join(sorted(unknown)))
    bindings.update(values)
    missing = [name for name in profile['required'] if not bindings.get(name)]
    if missing:
        raise ValueError('Missing parameters: ' + ', '.join(missing))
    substitute = lambda value: Template(value).substitute(bindings)
    environment = [name + '=' + substitute(value) for name, value in sorted(profile['environment'].items())]
    command = ['env', *environment, substitute(profile['executable']), *map(substitute, profile['args'])]
    return shlex.join(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE')
    args = parser.parse_args()
    values = {}
    for binding in args.set:
        name, separator, value = binding.partition('=')
        if not separator or name in values:
            parser.error('Use each parameter once as NAME=VALUE')
        values[name] = value
    try:
        print(render(json.loads(args.profile.read_text()), values))
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
