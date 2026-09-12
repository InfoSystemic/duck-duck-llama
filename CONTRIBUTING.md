# Contributing

Useful contributions make a model work better and make the evidence easier to inspect. A failed optimization with a well-controlled explanation is also valuable.

## Describe the change

State the concrete bottleneck or incorrect behavior, the changed implementation, and the resulting behavior. Identify the model/checkpoint, quantization, source base, loaded libraries, topology, and relevant runtime settings. Distinguish original engineering from imported upstream changes.

## Match validation to the claim

- For numerical code, define the intended arithmetic and include appropriate reference/edge-case checks.
- For a performance claim, retain matched controls, prompts, token counts, cache state, outputs, and measurement scope.
- For serving behavior, cover identity, reset/rollback, failure, and cleanup paths that the change affects.
- For source publication, preserve exact hashes and document filtering or relocation.

A component speedup alone is not a model speedup. A build pass is not a quality result. See [methodology](docs/methodology.md) for the project's measurement conventions.

## Repository checks

```bash
python3 tools/build_catalog.py --write
python3 tools/check_repository.py
```

Keep the curated guides current when the demonstrated behavior changes. Preserve historical source artifacts and unsuccessful experiments; use a new snapshot or an explicit update with provenance rather than silently rewriting their conclusions.

Do not include model weights, compiled libraries, private environment/process dumps, credentials, or unrelated user data. Use the [archive contract](engineering/README.md#archive-contract), retain upstream licensing notices, and describe full-model checks that could not be run.
