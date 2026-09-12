# GLM-5.3-Flash candidate profile

This separate, unpromoted candidate retains the exact pinned Q4, MTP2/p_min=0, private runtime and environment, with the September 11 parallel UNARY/SCALE CPU library prepended. Fresh end-to-end A/B validation is still required.

```
python3 launch.py --arm baseline --port 18131 --dry-run
python3 launch.py --arm tuned --port 18131 --dry-run
python3 launch.py --arm tuned --port 18131
```

The launcher runs in the foreground with exec, leaving service lifecycle to its caller. It never stops another process. Preflight rejects occupied ports, missing model shards or libraries, and an unexpected candidate CPU library resolution. Run it in the host network namespace because the restricted sandbox cannot bind even for preflight. Dry-run prints the full command and fresh fixed environment. Context and concurrency can be set with `--ctx-size` and `--parallel`. Each arm disables prompt caching and starts with CPUs 0-127 available; runtime NUMA pools retain the pinned 15 threads per socket.

Historical fleet-0911 evidence measured 15.39 to 15.71 decode tokens/s for the three-prompt parallel UNARY/SCALE A/B, with byte-identical greedy output. The new path has not yet been live-benchmarked. Keep plain `draft-mtp`: the ngram composite measured 13.65 tokens/s on those prompts and should not be transferred from Qwen. Benchmark both arms on the same prompts with server decode timing and full request wall time reported separately, then repeat the baseline.

Rebuild serially with `taskset -c 15 bash rebuild.sh`. The source snapshots and recipe are copied from fleet-0911/parallel-unary-0911; output is under `build/`. The link recipe preserves the original private object provenance, including the separate RMS guard and Q8 repack layers. `source-manifest.json` records the original measurements and verification; `build-manifest.json` records the new artifacts. Embedded source paths can change binary hashes, so numerical validation remains necessary.
