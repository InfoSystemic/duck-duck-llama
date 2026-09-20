# GLM bounded KV rollback kernel

Deployed and verified on the normal GLM endpoint after exact cache-state gates and a completed full-model A/B. The immutable production override is `deploy-r2/libllama.so.0.3.0`. See [the measured result](/home/user/sr950-strategy/codex-bench-0919/KV-RANGE-RESULT.md).

The normal GLM service allocates 1,048,576 KV cells. The original rollback scanned the entire position array, even with only a short occupied prefix. The candidate snapshots `cells.used_max_p1()` and scans only to that bound. Empty cells after that index cannot match the normalized nonnegative removal range. The used bound is captured before removals change the set. Sequence handling, position comparisons, mutations, head updates and return values are retained.

Activation is `LLAMA_KV_SEQ_RM_USED_PREFIX=1`, default off. The flag-off branch retains the original source body. An optional `LLAMA_KV_SEQ_RM_CONTROL_FILE` maps one regular four-byte file read-only; benchmark controllers change its aligned 0/1 word only between idle requests. `LLAMA_KV_SEQ_RM_PROBE=1` logs mode changes for engagement checks. Neither control file nor probe is intended for normal deployment.

## Exact deployed parent recovered

The earlier GLM KV port was blocked because the build tree's library differed from the pinned runtime. This reconstruction freezes 183 object files, then recompiles three sources with the deployed revisions:

- `llama-model.cpp`: remove the later DeepSeek-specific output tensor-parallel branch, retaining GLM MLA and opt-in NUMA repacking.
- `models/deepseek4.cpp`: use the earlier GLM source snapshot before the DeepSeek V4.1 compression-loader edits.
- `llama-kv-cache.cpp`: use the earlier source snapshot before the empty-cache rotation change.

The original include lookup and macro source-path spelling are preserved. The resulting parent library is byte-for-byte identical to the actual deployed `libllama.so.0.3.0`:

`8c794722eccdcc27aac88def7f1a1d549926a0d6aa3dd00586d6fdca5c8eb08a`

See `parent-inputs.json`, `parent-rebuild.json`, `parent-final-objects.json` and `rebuild_parent.py`. The candidate changes only `llama-kv-cache.cpp.o` relative to that exact parent; see `candidate-build.json` and `rebuild_candidate.py`. No current engine source or object is overwritten. The completed temporary test used the immutable corrected copy in `deploy-r2/`, which is now deployed while retaining the wider-pool CPU library.

## Standalone correctness gate

`test_kv_range.cpp` creates real engine cache objects with no model weight tensors. It exercises 16 configurations: capacities 1, 32, 128, 1K, 4K, 32K, 100K and 1M, each with unified and separate sequence streams. Across 5,650 scripted steps it records 2,800 states, including serialized metadata, physical cell indices, positions, shifts, extended positions, sequence membership, occupied bounds and the next allocation selected by the engine.

Coverage includes empty/full and sparse high-index cells, removal of the highest used cell, specific/all sequences, negative/open/empty ranges, shared cells, copy, keep, shifts, division, checkpoint save/restore, and shared-cache rollback no-ops. Separate-stream copies respect the engine's full-range requirement. An initial invalid partial-copy fixture failed on the unmodified production library and was corrected; its logs are retained under `logs/first-attempt-*`.

Production, candidate flag-off, candidate flag-on and alternating in-process modes all produce exactly the same 2,826,776 recorded bytes. The mode probe confirms both paths execute during the switching test. Tests use a 4 GiB virtual-address limit and observed less than 206 MiB RSS. Production remained healthy and unchanged during the standalone gate. See `validation.json` and its before/after runtime snapshots.

## Microbenchmark interpretation

With 1M cells allocated, the final candidate's bounded path measures about 2.3 microseconds per no-op rollback with 4K cells occupied, 17.9 microseconds with 32K, and 52.4 microseconds with 100K. The deployed control measured about 0.84-0.87 ms; the retained-source flag-off branch measured about 0.48-0.50 ms. Adding the branch changes compiler optimization even for the original source loop, so the whole production-to-candidate difference must not be attributed solely to the bound. These are one-thread metadata microbenchmarks, not end-to-end inference rates.

The first simpler candidate and its measurements are retained in `first-candidate/`. The final candidate adds the original-source control branch and the mapped switch so a full-model comparison can run within one loaded model.

The live controller is [kv_range_window_r2.py](/home/user/sr950-strategy/codex-bench-0919/kv_range_window_r2.py). It compares short native and actual Codex requests, runs the existing stateful regression suite, records separate coarse graph-phase traces, and restores the exact original production configuration. The temporary service has a 30-minute maximum and an `ExecStopPost` production restart. It does not promote the candidate.

This optimization does not address the pre-existing cached/fresh output consistency failure. That issue remains part of the full-model regression record.

The first live window stopped because the INFO-level probe was filtered; production was restored exactly. The r2 candidate raises only that opt-in probe to WARN. `diagnostic-only-diff.json` verifies one changed executable byte and a changed build ID. The corrected hash is `e77a573df69456745857677e71e4e9cc876b8294384fa7798629f3ebe7c3306a`. Its complete standalone gate passed again before the r2 window.

## Completed A/B and deployment

Eight actual Codex cached runs in ABBA/BAAB order at 3,998 input tokens improved aggregate decode from 15.2387 to 15.6078 tok/s (+2.42%). Both blocks improved (+3.19% and +1.66%). Input/cache counts, generated text, generated tokens, drafted tokens, accepted drafts and verification steps all match. The cold pair measured 15.09 to 15.70 tok/s. All six native outputs, fourteen Codex texts including profiles and bracketing controls, and nine stateful regression outputs pass their reference gates. The existing cache-consistency issue remains unresolved.

The successful temporary window restored the exact prior configuration after 19.79 minutes. The separate promotion installed `80-glm-kv-range-0919.conf`; production PID 1561764 was independently audited healthy, idle and unchanged from the deployment snapshot. Native parity and actual Codex cold/cached checks passed again (15.58/15.71 tok/s). Benchmark control/probe flags are absent. The optional coarse profile trigger `/dev/shm/glm-graph-phase.arm` and CPU op trigger `/dev/shm/flash-optrace.arm` are absent.

This measures only the short Codex fixture; it does not establish 18+ tok/s or a new 30K-context rate. Full evidence is linked in the result above, including the exact runtime audit and rollback-ready deployment record.
