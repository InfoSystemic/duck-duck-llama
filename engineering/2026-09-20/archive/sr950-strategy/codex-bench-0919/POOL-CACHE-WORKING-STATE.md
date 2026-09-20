# Active pooled-result-cache experiment

Final standalone candidate: /home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-pool-cache-r2-0920/deploy/libggml-cpu.so.0.22.0
SHA256: 5e3ef3b3de7fbe37d24a04afa7d3bf8b66308f259deb6533aa0b6e64c82927a2

All build, exact-output and four-NUMA gates pass. This is not deployed. Production PID 3717968 is unchanged and healthy on MTP cache-only + bounded KV rollback + wider pooling. The Q8 candidate remains undeployed.

Next: preflight then run pool_cache_window.py --production-pid 3717968 --context short --execute. The window uses private port 18141, checks exact native/stateful/Codex output and request-owned token counts, runs eight balanced warm comparisons with diagnostics disabled, then excluded profiles with explicit cache-hit logging. A 1,050-second controller alarm and 1,080-second transient-service lifetime trigger restoration. A separate long window can run only after the short validation report passes. Do not combine both into a longer outage.

No model throughput gain has been measured for this cache yet. NUMA pooling-only speedups and cold-fill cost are documented in the candidate README. Existing cache consistency discrepancy remains unresolved.
