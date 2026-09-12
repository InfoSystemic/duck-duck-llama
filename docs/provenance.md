# Provenance and attribution

Duck Duck Llama is maintained by InfoSystemic. The new repository preserves the preceding engineering repository's Git history, dated source snapshots, unsuccessful experiments, and licensing notices.

## Local engineering and upstream work

Local work includes CPU-NUMA integration and tuning, quantized kernel/layout experiments, model-specific split corrections, speculative-serving state work, benchmark and lifecycle controllers, native CPU operators and lookup integration, cache behavior, and validation tooling.

Model support also incorporates public work from llama.cpp/GGML, Unsloth's integration line, named Qwen/MTP contributors, DeepSeek's published reference implementation, YanissAmz's DSpark work, and JigSawPT's DeepSeek port. The repository does not claim authorship of those upstream contributions.

- [Patch attribution](../patches/ATTRIBUTION.md) identifies named contributions and source revisions.
- [Earlier source bundles](../engineering/2026-09-08/source-bundles.json) and [latest source bundles](../engineering/2026-09-12/source-bundles.json) pin the engine bases and patch identities.
- [DeepSeek port assessment](../engineering/2026-09-12/archive/serving/fleet-0912/upstream/DEEPSEEK-V41-UPSTREAM-20260912.md) records the inspected upstream implementations and remaining differences.
- [Native lookup report](../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md) explains the fixture/real-row distinction and links the retained official notice.

## What the hashes mean

Archive entries distinguish original-source hashes from published hashes. Source text and executable configuration are preserved; filtered evidence can intentionally differ from its original JSON. References to unchanged older files retain their prior published artifact.

Patch reconstruction compares the applied tree with the exported tree. It proves fidelity to the recorded source, not that every archived branch compiles or every candidate is suitable for serving. In particular, the legacy DSpark DSV4 branch retains an unresolved index merge; its three merge-stage blobs are preserved explicitly.

## Licensing and authorship

The [MIT license](../LICENSE) covers original project material under its retained notices. Imported source retains the applicable upstream notices, and model weights have separate publisher licenses. Model weights and compiled model runtimes are not distributed in this repository.

The research was developed with AI-assisted engineering tools. Commit history, source attribution, and exact validation records make that assistance and the resulting work inspectable. [Citation metadata](../CITATION.cff) is provided for reuse.
