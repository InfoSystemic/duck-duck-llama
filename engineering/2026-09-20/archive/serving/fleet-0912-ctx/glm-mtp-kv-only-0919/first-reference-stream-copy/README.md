# Reference-only separate-stream copy failure

The unmodified deployed libllama (e77a573d...) returned SIGSEGV (-11) during the first decode after copying a full sequence from KV stream 0 to stream 1. Its log ends with `update: copying KV buffer: stream 0 to stream 1`. The unified-cache cases completed before this failure. The normal GLM server uses a unified cache.

The failure occurred before any candidate-on arm. The standalone process peaked at 18.48 GiB RSS, below its 32 GiB bound, and production remained healthy, idle and unchanged. This is retained as an unresolved reference-runtime issue, not attributed to the MTP candidate.

The subsequent gate retains full-sequence copying on unified caches. Separate-stream cases prefill their second sequence directly, retaining independent streams, mixed batches, rollback and checkpoint-restore tests without triggering this unrelated copy path.
