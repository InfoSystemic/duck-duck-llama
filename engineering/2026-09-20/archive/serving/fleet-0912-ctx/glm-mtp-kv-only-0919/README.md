# GLM MTP cache-only catch-up candidate

Status: deployed and independently verified on the normal endpoint (:18131), PID 3717968. Standalone and full-model exact gates passed. The isolated A/B measured +2.51% on the standard actual Codex fixture. Final cold/cached deployment smoke rates were 15.67/15.52 tok/s; these are not a workload-wide throughput floor. See the result record below.

## Change

The MTP driver calls `process()` with no output rows to refresh its KV cache from target hidden states. The original graph builds query projection and attention despite requesting no logits or hidden rows. This candidate retains token/hidden normalization, their combined projection, attention-input normalization, KV projection and normalization, and the same `cpy_k` cache write. It returns before the unused query/attention and output path.

The branch is default off (`GGML_GLM5N_MTP_KV_ONLY=1` enables it) and only applies with zero outputs, a single MTP layer, disabled ordinary embeddings, enabled masked next-token embeddings, no backend samplers, and no requested intermediate layer embeddings. Other cases retain the full graph. No floating-point arithmetic in the retained KV path changes.

A private input class supplies only cache destination indices. It intentionally uses the input base class's no-reuse policy; each cache-only graph is rebuilt so no cache-context pointer can become stale. The optional read-only four-byte mapped control is changed only between idle requests. A graph input checks the mode during reuse and rejects reuse after a mode change, including repeated equal-shaped zero-output calls. `GGML_GLM5N_MTP_KV_ONLY_PROBE=1` is an optional standalone diagnostic and is absent from the live throughput test.

## Exact parent and build

The preserved GLM source recompiles to the exact frozen object. Relinking it with the bounded KV rollback object reconstructs the current production libllama byte for byte (`e77a573df69456745857677e71e4e9cc876b8294384fa7798629f3ebe7c3306a`). Only `models/glm5next.cpp.o` changes in the new candidate. No engine source or object is overwritten. See `reference-build.json`, `build_candidate.py`, `candidate-build.json` and `mtp-kv-only.patch`.

Candidate SHA256: `65670d4a19178bbdffcfbc79ee953e5319083875309ed52aa4de214393e20ce3`.

## Standalone correctness gate

`test_mtp_kv_only.cpp` loads the real 9.26 GB Q8 MTP sidecar on the same four NUMA devices, without loading another target model. It uses synthetic hidden inputs to test execution and cache equivalence, not language-model quality. Production, candidate off, candidate on, and alternating mapped control all pass at both 1 and 15 workers per socket.

Each arm executes 46 calls across unified and separate-stream contexts, including 24 zero-output calls and 30 full output rows. All 170 records / 28,400,396 bytes match the production reference at the same worker count. Records contain serialized sequence metadata plus actual KV tensor bytes, full vocabulary logits, and next-token hidden outputs. Cases include repeated equal-shaped calls, 1/3/4/6/8/12/32/64/128-token batches, mixed output masks, rollback and overwrite, checkpoint save/restore, shared unified sequences and two-sequence batches. The cache-only graph has 15 nodes for unified cache and 16 for separate streams.

The private process is low priority, limited to 64 GiB virtual address space, aborted above 32 GiB RSS or when production becomes busy, and has a ten-minute timeout. Observed peak RSS remained below 20 GiB. Production runtime identity and idle health were verified before and after each worker-count gate. See `validation-w1.json` and `validation-w15.json`.

## Preserved reference-runtime failure

The first reference-only fixture encountered SIGSEGV when a separate KV stream copied a full sequence to another stream, at the first subsequent decode. This occurred with the deployed library, before any candidate arm. The reproduction, original test binary/source, output prefix and log are preserved in `first-reference-stream-copy/`. The normal server uses unified KV. The final gate retains unified sequence copying and directly prefills independent separate streams to avoid conflating this unrelated failing path with the candidate. This reference-runtime issue remains unresolved.

## Full-model gate

The private-port controller is [mtp_kv_window_r2.py](/home/user/sr950-strategy/codex-bench-0919/mtp_kv_window_r2.py). It ran both modes in one loaded model on :18141, with native and stateful regression gates, actual Codex cold pairs, eight balanced cached requests, and separate phase traces. It restored the exact prior production runtime on :18131. The raw controller report preserves a final trace-parser assertion: the expected new 15-node graph was saved but rejected by an old strict allowed-node list. Saved traces passed offline validation after the parser gained an explicit mode-specific node set; a final restored production control passed. No performance A/B was repeated. The original failed record remains unchanged alongside the derived validation report and its source hashes.

Primary cached Codex result: **15.43298 -> 15.82093 tok/s (+2.51377%)**, with +2.69088% and +2.33489% balanced blocks, identical output and verification work, and clean request-owned counters. Both profiles reconcile 105 verifications / 422 graphs; total draft work falls 21.269 -> 18.152 ms per steady cycle. All six native, fourteen Codex and nine stateful regression outputs match their references. No new long-context speed measurement or consistent 17+ claim is made.

An earlier shared-port attempt stopped before enabling the optimization because another request was active. It restored production; its contaminated rates are excluded.

Full result: [MTP-KV-RESULT.md](/home/user/sr950-strategy/codex-bench-0919/MTP-KV-RESULT.md). Deployment has its own pinned manifest, identity/output gates and rollback controller; it is not automatic in the A/B script.

The pre-existing cached/fresh consistency issue is unchanged; standalone state parity is not a fix for it. The measured gain above applies only to the matched short fixture.

Production override: `~/.config/systemd/user/glm53-flash-production.service.d/90-glm-mtp-kv-only-0919.conf`. It enables the cache-only path and retains wider pooling and bounded KV rollback. Benchmark control/probe flags and all profiler arm files are absent. Independent final audit: [mtp-kv-only-final-audit.json](/home/user/sr950-strategy/codex-bench-0919/mtp-kv-only-final-audit.json).
