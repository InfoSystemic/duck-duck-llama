# Native DeepSeek-V4.1 context candidate

Status: isolated, unpromoted engineering candidate. Four context tests and five interruption-recovery tests pass without checkpoint loading or network calls. **No 4K/16K full-model generation or logit comparison has been run.** The September 12 published snapshot preserves the earlier version; this working candidate now includes the cache-recovery correction.

## What prevents extending the existing endpoint

- `fleet-0903/deepseek_v41_checkpoint_0910.py:174` explicitly constructs 256-token buffers. The official model itself defaults to 4096 and has parameterized compressed-KV, index-key, RoPE and Engram-history buffers.
- `deepseek_v41_server_0910.py` uses the real config for request bounds but hardcodes 256 in health/ready messages. Output is separately capped at 128 tokens; the candidate retains that limit.
- `deepseek_v41_serving_store_0910.py:96` asserts a lifetime maximum of 100,000 unique Engram rows, with no eviction. Two Engram layers can touch 48 rows/token: 4K novel tokens can exceed the cache limit, even within one request. The same limit can eventually fail many short requests.
- A full 16K prefill materializes large attention and hyper-connection activations. The official attention and compressor nonzero-position branches accept one token, not arbitrary prompt chunks. Merely looping over 256-token chunks at increasing positions is incorrect.
- The selected `NativeSparse` caps selected positions at 2048, not absolute context position. The actual window128 plus index-topk512 totals640, so that particular bound does not rule out16K. Full-model validation is nevertheless missing beyond the old256 context.

## What this candidate changes

`context_server.py` subclasses the existing Goal2 runtime. During its isolated process initialization it replaces the model factory and store class, then restores those module-level factories. It retains native GEMM, quantization, sparse attention and exact16 configuration checks. Nothing is monkey-patched in a running selected process.

`context_policy.resize_context()` rebuilds only derived context buffers and RoPE tables using the official equations. It updates the config/model limits, clears source cache state and indexer RoPE references, and leaves parameters unchanged. Supported evaluation sizes are256,4096,16384.

`BoundedPrefill` uses at most256 tokens for the initial position-zero forward, then submits remaining prompt tokens one at a time through the official decode path. This keeps prompt activation sizes bounded while preserving the model's native sequential semantics. It does **not** promise equal logits to a differently batched full prefill; that comparison remains a promotion requirement. Long prompts can be very slow with this policy.

`bounded_engram.py` replaces the fatal monotonic row-cache limit with an8192-row memory LRU and a separate owned persistent directory capped at8192 row files by default. It reads existing selected row files without editing them, batches at most128 missing rows at a time, and retains the original raw264-byte representation, SHA-256 validation and FP8/E8M0-to-BF16 decoding. Once the new persistent cache fills, new rows remain memory-cached but are not persisted; they may be downloaded again after eviction. Existing legacy files are never deleted. The limit is a file-count bound, not an exact filesystem-byte budget.

The candidate has a distinct port default18174, dynamic context metadata, an experimental-context marker and per-request context/row-cache metrics. It still requires the existing fleet lifecycle-lock descriptor and explicit Goal2 environment flags. No launch command was executed, no cache directory initialized outside tests, and no weights downloaded.

## Persistent cache and cold-fetch costs

The existing common/expert store cap remains90GiB and is separate from the selected64GiB packed-expert cache. Existing native tensor eviction uses expert groups, LRU timestamps and at least4GiB reclamation when possible. Resident/packed eviction hooks remain inherited. Engram rows and external Engram projection files are not included in `stored_bytes`, so that metric is not total resident or filesystem use.

`CoalescedStore._fetch` globally spaces requests by0.14 seconds, with eight connections and retries/backoff. Missing Engram rows require two tiny HTTP range requests each, so48 novel rows can incur96 scheduled requests, about13.44 seconds/token of spacing before additional latency/retries. A16K fully novel prompt could therefore take days of cold Engram fetching. Common/expert ranges are coalesced up to32MiB, but the row path does not combine random tiny ranges. This wrapper preserves that transport policy; it does not claim to solve cold-checkpoint performance. Long-context usability still needs locally available checkpoint rows/experts or a separately validated transport/storage plan.

The context wrapper does not retain conversational KV across HTTP requests. As before, each request encodes and prefills the supplied conversation from position zero; only weight/row caches survive between requests.

## Verification and remaining work

The interrupted-write quota defect is corrected. A partially written row reserves a persistent slot before its write completes. Recovering a fully manifested partial updates the committed-row count without allocating a second slot. Uncommitted rows with missing or torn manifests are discarded only in the owned context directory and fetched again; committed corruption still fails integrity checks. A lifetime directory lock prevents a second cache instance from bypassing the quota. Calls within one runtime remain serialized by the existing HTTP model lock. Explicit `close()` releases the directory lock.

`test_cache_recovery.py` reproduced three failures and two errors in the published parent. All five tests pass after the fix, covering recovery after restart and within the same process, missing/torn manifests, and competing cache writers. They compare the recovered native BF16 rows and check actual persistent-file counts. `recovery-verification.json` records the before/after evidence and source hashes.

Run with the existing CPU-reference Python environment:

```sh
taskset -c 15 /home/kwebb/InfoSystemic/AI-Server/tools/deepseek-v41-cpu-reference-0910/venv/bin/python serving/fleet-0912/deepseek-context/test_context.py
taskset -c 15 /home/kwebb/InfoSystemic/AI-Server/tools/deepseek-v41-cpu-reference-0910/venv/bin/python serving/fleet-0912/deepseek-context/test_cache_recovery.py
```

`test-result.json` and `test.log` record four passing tests:

1. Bounded prefill sequencing, reset and bounds through16384 positions using a deterministic scheduling oracle.
2. Native Engram row equality across memory eviction, bounded new persistence, restart hits and deliberate disk corruption.
3. The unchanged official Engram hash at positions255,256,4095,4096,16383 and reset after a longer request.
4. Derived cache resizing and official RoPE generation for4096/16384, including ratio1/ratio2 source ownership.

The existing official compressor cadence oracle is in `../deepseek/results/compressor-oracle/`. Neither that oracle nor these tests establish full-model extended-context correctness. Before promotion: compare cached attention/logits and text using real native weights across window, index-topk and context boundaries; run interrupted-request reset tests; verify long-request memory and fetch costs; and measure a representative4K/16K workload. The candidate leaves vision, tools, prefix-KV reuse and DSpark unchanged and unimplemented.
