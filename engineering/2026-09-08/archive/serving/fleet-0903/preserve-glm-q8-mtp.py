#!/usr/bin/env python3
"""Preserve the verified RAM-staged GLM draft in the model cache without overwriting files."""
import hashlib
import json
import os
from pathlib import Path
import shutil

base = Path(__file__).resolve().parent
manifest = json.loads((base / 'results/glm-mtp-q8-artifact.json').read_text())
assert manifest['verified']
source = Path(manifest['output'])
destination = Path('/models/gguf/GLM-5.3-Flash/MTP/GLM-5.3-Flash-MTP-Q8_0-goal-0904.gguf')
expected = manifest['output_sha256']
expected_bytes = manifest['output_bytes']

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

if destination.exists():
    assert destination.stat().st_size == expected_bytes and digest(destination) == expected
else:
    assert source.stat().st_size == expected_bytes
    assert shutil.disk_usage(destination.parent).free > expected_bytes + 1024**3, 'Insufficient model-cache space'
    temporary = destination.with_suffix('.gguf.partial')
    created = False
    try:
        with temporary.open('xb') as output:
            created = True
            value = hashlib.sha256()
            with source.open('rb') as input_file:
                while chunk := input_file.read(16 * 1024 * 1024):
                    value.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        assert value.hexdigest() == expected, 'RAM artifact differs from verified manifest'
        assert temporary.stat().st_size == expected_bytes and digest(temporary) == expected
        os.link(temporary, destination)
        temporary.unlink()
        created = False
    finally:
        if created:
            temporary.unlink(missing_ok=True)
result = dict(source=str(source), destination=str(destination), bytes=expected_bytes,
              sha256=expected, source_manifest=str(base / 'results/glm-mtp-q8-artifact.json'), verified=True)
(base / 'results/glm-mtp-q8-persistent.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
