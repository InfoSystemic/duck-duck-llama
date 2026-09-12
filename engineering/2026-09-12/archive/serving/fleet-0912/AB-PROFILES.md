# Fresh-server A/B runner

After the shared port is vacant, run from the host network namespace:

```
python3 ab_profiles.py --model glmflash --dry-run
python3 ab_profiles.py --model glmflash
python3 ab_profiles.py --model qwen
```

Each run starts baseline, tuned, then baseline again, with a fresh foreground server for each arm. It uses one pass through the three historical prompts per server; Qwen ngram history therefore cannot warm a repeated benchmark prompt. The runner stops only its own child server, using recorded PID start time, executable/model identity and Linux pidfds. It never stops an existing listener. Normal completion leaves the shared port vacant.

The default load and benchmark observation windows are each 3600 seconds per arm. An expired observation window saves the precise live server and benchmark identities and returns exit code 3 without killing or replacing them. Resume those same processes with:

```
python3 ab_profiles.py --resume /absolute/path/to/results/ab-qwen-TIMESTAMP
```

`--load-seconds`, `--benchmark-seconds`, and `--request-seconds` can adjust the respective deadlines. The request deadline defaults to 600 seconds. Ctrl-C also preserves live processes and the state can be resumed. A failed quality/comparison run does not resume into later arms; inspect its raw responses before starting a new run. A live benchmark is preserved if another failure prevents safe cleanup.

Every stage retains launch preflight, server stdout/stderr, benchmark stdout/stderr, raw requests/responses, and benchmark runtime metadata. `state.json` records child lifecycle events atomically. The runner verifies that the benchmarked PID owns the listening socket, that executable/model/draft paths and alias match the selected profile, and that the process identity remains stable after inference. It rejects quality failures, truncated throughput samples, prompt cache reuse, changed request parameters, and changed greedy output before starting another arm.

`comparison.json` reports per-prompt baseline-before/tuned/baseline-after rates, control drift and candidate change relative to the two controls. It does not automatically promote the candidate. Three simple quality checks and byte-identical benchmark output are useful gates, not a comprehensive model-quality evaluation. One sample per prompt is insufficient to establish statistical significance.

`python3 -m unittest test_ab_profiles -v` runs bounded fixture tests for the quality, measurement, parity and process-identity gates without starting a model or signalling a process.
