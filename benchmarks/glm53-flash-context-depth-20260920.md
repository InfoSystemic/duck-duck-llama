# GLM-5.3-Flash on four sockets: decode and prefill against context depth (2026-09-20)

Production revision f of the [Paseo decode campaign](glm53-flash-paseo-decode-20260920.md), measured in ONE append-only session on the
production server (1,048,576-token context, two unified slots, prompt cache and context checkpoints on), 8K tokens of new context
per turn up to 128K. Probe: [tools/depth_curve.py](../tools/depth_curve.py); raw results in
[engineering/2026-09-20/depth-curve/](../engineering/2026-09-20/depth-curve/).

## Method

Every turn appends one chunk of open-source C/C++ (this repository's llama.cpp tree) and the same 30-token instruction, then generates
192 tokens twice from the identical prompt: with the server's default sampling (temperature 1.0, top-k 40, top-p 0.95, min-p 0.05: what
Codex traffic gets) and greedy. The assistant turns kept in the history are a fixed string, so the prompt cache carries everything but
the new chunk, as it does for a harness. Reported per depth: tokens prefilled and their rate, and for decode the ms per verify cycle
(the cost that depends on depth), tokens per cycle (which depends on the text) and tok/s.

The probe yields: a watcher polls `/slots` and drops the probe's own request when a second slot starts processing, then waits for a long
idle gap and retries the same turn. In a deliberate collision test the foreign request took 22 s instead of ~4 s, the cost of one
in-flight prefill batch; the run itself had no preemptions. The host was not penned (other workloads ran), which shows in three sampled
rows (33K, 82K, 128K) whose cycle time is well above the greedy pass at the same depth.

## Results

| depth (tokens) | prefill of the new chunk, tok/s | greedy tok/s | greedy ms per cycle | sampled tok/s | sampled ms per cycle |
|---:|---:|---:|---:|---:|---:|
| 8,072 | 63.9 | 16.84 | 131.0 | 16.47 | 135.6 |
| 16,686 | 46.7 | 19.12 | 130.8 | 18.80 | 146.3 |
| 25,054 | 35.7 | 19.59 | 127.6 | 19.17 | 130.4 |
| 33,404 | 29.1 | 17.67 | 141.4 | 13.91 | 167.3 |
| 41,629 | 25.6 | 17.58 | 142.2 | 17.33 | 144.3 |
| 50,390 | 22.9 | 16.84 | 148.5 | 17.10 | 146.2 |
| 58,382 | 23.1 | 17.34 | 144.2 | 17.36 | 144.0 |
| 66,524 | 23.2 | 15.52 | 161.1 | 16.44 | 152.1 |
| 74,342 | 22.3 | 14.70 | 170.0 | 15.44 | 161.9 |
| 82,194 | 20.2 | 14.79 | 169.0 | 10.99 | 200.9 |
| 89,974 | 18.4 | 14.02 | 178.3 | 13.89 | 180.0 |
| 97,685 | 16.6 | 14.33 | 174.5 | 14.80 | 169.0 |
| 105,394 | 16.6 | 13.96 | 179.1 | 13.71 | 182.3 |
| 113,015 | 15.8 | 13.62 | 183.5 | 13.37 | 187.0 |
| 120,279 | 15.1 | 13.29 | 188.1 | 13.20 | 189.4 |
| 128,416 | 14.5 | 12.62 | 198.1 | 10.78 | 237.0 |

Greedy tokens per cycle were 2.5 at every depth from 16K on (the text is the same kind of file summary each turn); the 8K row's
2.2 is the text, not the depth. For reference the Codex fixture at 4K reads 20.2 greedy / 19.7 sampled with 2.45 tokens per cycle.

## What it says

- **Decode is flat to ~25K tokens, then grows about 0.6 ms per cycle per 1,000 tokens** (128 ms at 25K, 198 ms at 128K). With 2.5
  tokens per cycle that is 19.6 → 12.6 tok/s. Extrapolated, not measured: ~6 tok/s at 512K.
- **Prefill of new tokens is the steeper curve:** 74 tok/s at 4K (Codex fixture), 64 at 8K, 29 at 33K, 23 at 50-67K, 14.5 at 128K.
  Filling the 128K took 97 minutes in total. A fresh 512K prompt is a matter of hours on this machine; a session only gets there by
  appending, where each turn pays for its own new tokens at the rate of its depth.
- **The prompt cache and checkpoints hold at every depth:** the greedy re-request at each depth re-prefilled 4 tokens (0.4 s at 1K,
  0.7 s at 128K), so a harness's append-only history costs only its new tokens.
- **Memory:** the service cgroup grew from 277.5 to 292.6 GiB over the session (about 110 MB per 1,000 tokens of depth), against a
  380 GiB limit. The KV cache is materialised at load, so this is checkpoints and prompt-cache copies; whether it keeps growing linearly
  past 128K or saturates at the cache-ram cap has not been measured. Until it is, a 512K session is projected safe (~335 GiB) and a
  1M session is not.

## Where the growth is (per-op trace, same session, one socket's verify graph)

[tools/depth_trace.py](../tools/depth_trace.py) rebuilds the probe's history so the prompt cache answers it, arms the CPU op
profiler once decode has started, and [tools/opdiff.py](../tools/opdiff.py) compares two traces
([raw traces and the table](../engineering/2026-09-20/depth-curve/)). Verify graph: 105.2 ms at 8K, 237.6 ms at 128K, the same
7,239 nodes.

| op (eleven DSA layers unless noted) | 8K | 128K | growth |
|---|---:|---:|---:|
| `LIGHTNING_INDEXER` pool score: 3 query rows x 32 heads against 34,178 pooled keys per layer | 2.2 ms | 37.5 ms | +35.3 |
| fused pool kernel (dispatched at the `indexer_pool_members` GET_ROWS node): validates 2 KB of cache per pool against its record, copies the cached result | 1.3 | 34.6 | +33.3 |
| one layer (the last DSA layer) not fused at 128K: `CONT` x2, `ADD`, `SOFT_MAX` over the whole [4,128,34178] member set | 0 | 51.0 | +51.0 |
| `TOP_K` over 34,178 pools x 3 rows | 0.3 | 3.1 | +2.8 |
| mask add, attention, MoE cache effects | | | +9 |

Attention itself grows 1.4 ms. The slope is the indexer's bookkeeping: a scoring kernel running at per-row overhead (~240 GFLOP/s
on 15 cores for 0.84 GFLOP per layer), validation bandwidth for a cache that already holds the answer, and one layer dropping off
the fused path (the 14-node matcher rejects when the allocator lets the output overlap its inputs). None of it is inherent; the
plan to remove it is in the engineering notes of the next revision.
