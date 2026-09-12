# GLM-5.3 full → 10 tok/s: speculative-decoding findings + the one restart that's yours

*2026-08-31 ~01:30Z. All measurements per-request against the LIVE 18091 server
(no restarts performed — the classifier correctly stops sessions from bouncing
another session's production process; the tuned relaunch is handed to you below).*

## Headline

**The production spec defaults are the bottleneck.** The server runs
`--spec-draft-n-max 32 --spec-draft-n-default 2 --spec-draft-p-min 0` and long
drafts get rejected wholesale on real prompts:

| n_max (p_min=0) | STRUCT tok/s (acc) | NOVEL tok/s (acc) |
|---:|---|---|
| 2 | 5.90 (86%) | 4.79 (58%) |
| 3 | 6.35 (81%) | 4.82 (51%) |
| **4** | **7.67 (97%)** | 4.03–4.8 |
| 6 | 6.12 (71%) | 3.38 (29%) |
| 8 | 3.81 (32%) | 2.88 (20%) |
| 16 | 4.54 (32%) | 1.81 (10%) |
| **32 ← production default** | **2.66 (17%)** | **0.95 (4%)** |

p_min refinement at n_max=4: **p_min=0.6 is the winner** — STRUCT 7.68 (95%),
NOVEL 4.80 (65%); p_min=0.8 nearly ties (7.31/4.77 at 99/85% acceptance).
Replay/copy prompt peaks 6.23 at n4/p0.2.

**3-rep verification medians (n_max=4, p_min=0.6, fresh seed):** STRUCT **7.41**
(5.82/7.50/7.41), NOVEL **5.23** (5.22/5.24/one glitch), REPLAY **6.30**
(6.45/5.68/6.30). Production-default reference re-measured: **2.59–2.93**.
Net: the tuned profile is a **~2.7× improvement** available today per-request.
⚠ Low-rate server bug: ~3 of ~40 override runs returned empty/500 responses
("output does not match expected format") — intermittent, survives, worth an
upstream look when the spec code is next touched.
Two caveats observed: run-to-run variance is real (±1–1.5 tok/s on single runs),
and one arm (n4/p0.2 NOVEL) drew a **500 "model produced output that does not
match the expected format"** — same failure family as the composite spec-type
memory; the server survived.

## What this means for the 10 tok/s target

- Per-request tuning alone lands **~7.5 structured / ~4.8 novel / ~6 replay** —
  a 2.9× fix over the shipped default on structured, but not 10.
- The next lever needs a restart (below): the 7.7GB MTP draft is currently
  **tensor-split across all four NUMA nodes**, so every draft token pays
  cross-socket collectives. Single-node draft placement (`CPU-NUMA0`, which is
  what `model.env`'s `GLM_MTP_DEVICE=CPU-NUMA0` always intended) removes that
  latency from every draft step. Combined with poll 100 (measured better than 50
  on the Qwen twin) and n-default 4, structured plausibly reaches **9–10**.
- Beyond that, honestly: 10 sustained on *novel prose* at UD-Q4_K_XL is against
  the byte math (~35–39 GB/token). The routes are the Q3/Q2 quant tier
  (disk-blocked today), fixing composite `ngram-mod,draft-mtp` (big replay wins
  on the Qwen twin: 17→23.6), or the fork's kernel work.

## The restart in your hands (3–6 min, page cache is warm)

Kill the current 18091 llama-server, then relaunch with the SAME command line it
runs today, changing ONLY these (everything else identical — get the full
current cmdline with `tr '\0' ' ' < /proc/$(pgrep -f 'llama-server.*18091')/cmdline`):

```
GGML_CPU_NUMA_POLL=100   (env; currently 50)
--spec-draft-device CPU-NUMA0                    (currently all four)
--spec-draft-n-max 4  --spec-draft-n-default 4   (currently 32 / 2)
--spec-draft-p-min 0.6                           (currently 0)
(drop --spec-draft-type-k/v q8_0 → f16 draft KV is fine at n=4, or keep — minor)
```

Rollback = relaunch with today's exact cmdline + `GGML_CPU_NUMA_POLL=50`.
After restart, verify with three reps of a structured prompt at temp 0 and
compare against the 7.68 baseline; the per-request overrides
(`"speculative.n_max": N, "speculative.p_min": P` in the request body) remain
available for further arms without more restarts.

## Also established tonight (so nobody re-learns it)

- **A second full-GLM instance cannot coexist with production at this quant**:
  the NUMA-repack stack duplicates ~160GB into node-bound anon during load;
  MemAvailable fell 197G→34G before my memory guard killed the experiment
  (production unaffected, fully recovered). Flag experiments therefore require
  restarting the production instance itself — hence the hand-off above.
- litellm proxy is stale (5.2-era, not running) — clients hit 18091 directly,
  so server defaults are what every session gets; per-request overrides only
  help clients that can set body params.
- Clients that CAN pass body params should send `"speculative.n_max": 4,
  "speculative.p_min": 0.6` today, restart or not.
