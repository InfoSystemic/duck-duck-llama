# Qwen3.8-Flash-Next candidate profile

This is a separate, unpromoted candidate using the pinned Q6 model and private library stack. It adds the September 11 parallel UNARY/SCALE CPU changes and Qwen's measured ngram/MTP composite. Fresh end-to-end A/B validation is still required.

```
python3 launch.py --arm baseline --port 18131 --dry-run
python3 launch.py --arm unary --port 18131 --dry-run
python3 launch.py --arm tuned --port 18131 --dry-run
python3 launch.py --arm tuned --port 18131
```

The foreground process replaces the launcher and keeps its PID. The caller owns stopping and starting services. The launcher never stops another process. It rejects occupied ports, missing shards, missing libraries and an unexpected CPU-library resolution. Run in the host network namespace; the restricted sandbox cannot bind a port even for preflight. Dry-run prints the complete command and exact fresh environment. Optional context and concurrency controls are `--ctx-size` and `--parallel`; total context must allow at least 512 tokens per slot.

`baseline` uses the original MTP4/p_min=0.3 recipe. `unary` adds only the parallel CPU changes. `tuned` also changes speculation to `ngram-mod,draft-mtp`. Every arm disables prompt caching and widens initial affinity to CPUs 0-127. The original library search path is preserved after the candidate CPU directory; the expert even-split private libllama remains first for that soname. No environment settings are inherited from the caller.

Historical evidence in fleet-0911/results: `resp-qwen-pinned-mtp4-{1,2,3}.json` averaged 19.90 decode tokens/s, `resp-qmax1-{1,2,3}.json` averaged 20.32, and `resp-qngram-{1,2,3}.json` averaged 20.80. These are workload-dependent prior measurements. Ngram state persists across requests despite `--no-cache-prompt`: compare arms using fresh servers and the same distinct prompts, and repeat the control after the candidate. Measure server decode timing separately from wall-clock throughput and require correct text.

Rebuild serially with `taskset -c 15 bash rebuild.sh`. Sources were copied unchanged from fleet-0911/parallel-unary-q4e-0911; the build and relink scripts place outputs under `build/`. The rebuild checks that the baseline relink matches the pinned CPU library byte-for-byte. `source-manifest.json` describes the original evidence; it is not a manifest of the new path. `build-manifest.json` records the new files.
