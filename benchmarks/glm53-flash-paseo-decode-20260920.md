# GLM-5.3-Flash decode as an agent sees it: 12.0 to 17.6 tok/s, September 18-20

**Status:** 15.5-16.2 tok/s is what the production service delivers to a Codex agent inside Paseo today. A candidate stack measured
**17.61 tok/s** in the same loaded process on 2026-09-20 and is **not deployed**. The 18 tok/s target is **not reached**.

Machine: Lenovo SR950, 4x Xeon Gold 6242 (16 cores each, AVX-512 VNNI, no AMX), 24 DIMMs DDR4-2400, 755 GiB, no GPU.
Measured aggregate read ceiling 381.6 GB/s ([tools/membw.c](../tools/membw.c)). Model: GLM-5.3-Flash UD-Q4_K_XL (186 GiB), Q8_0 MTP
sidecar, draft depth 2, four CPU-NUMA devices in tensor-parallel, 15 workers per socket, 1,048,576-token context, two unified slots.

## How the number is measured

The rate that matters is the one an agent gets, so the probe is an actual Codex turn through the installed Paseo wrapper
([tools/paseo_codex_bench.py](../tools/paseo_codex_bench.py)): app-server, `thread/start` and `turn/start` with the lowercase model
name Paseo sends, one fixed no-tools prompt, about 4,000 input tokens of instructions and tool schemas, 257-270 generated tokens.
Decode tok/s is `tokens_predicted / tokens_predicted_seconds` from the server's `/metrics` delta, so prefill and client time are excluded.

Two things make comparisons valid:

- **Same process.** Separate loads of the same configuration differed by about 3% (15.5-15.8 on 09-19, 16.24 in a quiet 01:00 window).
  Every gain below compares modes inside ONE loaded server, switched through a memory-mapped control word while both slots are idle,
  in balanced order (ABBA, or ABCCBA twice).
- **Same tokens.** A loopback adapter sets `temperature=0, seed=42, cache_prompt=true`, and each pair is checked for identical
  generated text, draft, accepted-draft and verification counts. Where a change alters the text this is stated.

What the adapter hides: Codex sends **no sampling fields**, so real turns use the server defaults (the GGUF carries temp 1.0, top_p 0.95).
One pair of cold runs on the production service, minutes apart: 15.6 tok/s at 78% draft acceptance greedy, **14.8 tok/s at 70%** with the request exactly as sent.

## What moved it

| Change | Same-process result | Output | State |
| --- | --- | --- | --- |
| Codex model catalog had no lowercase alias, so Paseo's requests fell back to 20,751 characters of default instructions | input 7,739 -> 3,998 tokens; 9.6 -> 12.0 tok/s (cold, sampled, single observations) | n/a | applied 09-19 |
| Fused pooled-indexer compressor + flat recurrent-state copy | native greedy 12.65/14.62/14.17 -> 14.10/16.21/15.71 for the copy alone | byte-identical | deployed 09-19 |
| Eight-channel pool inner loop | 14.60 -> 15.15 (+3.8%); at 29,930 tokens 9.05 -> 10.26 (+13.4%) | byte-identical | deployed 09-19 |
| KV rollback scans the used prefix, not 1,048,576 cells | 15.24 -> 15.61 (+2.4%) | identical | deployed 09-19 |
| MTP catch-up with zero outputs stops after the KV write | 15.43 -> 15.82 (+2.5%) | identical | deployed 09-19 |
| Cell-split MQA attention + selection top-k | 16.24 -> 17.23 (+6.1%), 157 vs 158 of 196 drafts accepted | **changes**, toward float64 | measured 09-20, not deployed |
| + content-verified pooled-result cache | 17.23 -> 17.61 (+2.3%) | identical to the row above | measured 09-20, not deployed |

Window w1 (09-20, four runs per mode): production-equivalent 16.22 / 16.26 / 16.27 / 16.20; kernels 17.05 / 17.24 / 17.30 / 17.30;
kernels + cache 17.80 / 17.66 / 17.51 / 17.48. On ~200-token native prompts the same stack is within noise of production
(14.3/16.5/16.3 vs 14.8/16.6/16.4): the gain is a context term, and an agent never runs at 200 tokens of context.

Patches and their evidence: [patches/README-20260920-glm5next.md](../patches/README-20260920-glm5next.md). Records:
[engineering/2026-09-20](../engineering/2026-09-20/README.md).

## Where a cycle goes

One speculative cycle verifies three tokens and yields 2.57 on this fixture (2.62 in w1). Production profile at the 4K fixture, 09-19:

| Part | ms | Note |
| --- | ---: | --- |
| target verify graph | 137 | 7,152 nodes |
| MTP draft: catch-up + two passes | 18.2 | 3.3 ms of it is graph rebuild |
| outside graphs (sampling, rollback, server) | 6.0 | |
| **cycle** | **161** | 2.57 tokens -> 15.9 tok/s |

Worker-0 attribution inside the verify graph (117 ms of ops + 12 ms of barriers): routed experts 42.3, dense matmul 35.6 (32.5 of it Q8_0),
flash attention 9.5, pooled-key gather 3.8, cross-socket reduces 4.4, top-k 2.9, gated delta net 2.7, about 13 in small ops.

**The experts are finished.** 24 (token, expert) pairs x 42 layers x 3.8 MB per socket in 42 ms is ~96 GB/s per socket, i.e. the machine's
measured ceiling. Every expert's matrices are already split four ways across the sockets. No expert kernel can go faster on these DIMMs;
only fewer bytes can. The three KDA projections measure the same way. What is left is everything a GPU-hybrid engine hands to the GPU:
attention, the indexer, the hyper-connection small ops, the draft loop, host work. All gains in the table came from there.

The marginal verify token costs roughly 29 ms (137 ms for three tokens against about 78 ms for a single-token graph, from 12.8 tok/s with
speculation off on 09-11) and only ~14 ms of that is expert traffic.
That is why draft depth 3-4 loses here (-22% at depth 4) and why the non-expert per-token work is the target.

## What did not work

Kept because each one looked right first ([log](../engineering/2026-09-20/archive/sr950-strategy/GLM53-FLASH-CODEX-20260919.md),
[09-18 windows](../engineering/2026-09-20/archive/serving/fleet-0912-ctx/RESULTS-LOG.md)):

- MTP depth 4: -22%; acceptance falls 49/65/63% -> 30/38/39%.
- MoE/FFN fusion switches: output changes on all three prompts, no speed signal.
- Porting a wider expert kernel: cancelled by arithmetic before any code; the experts already run at bandwidth.
- Q4 token batching, Q4/Q5 clamp fusion, a batched Q8 kernel (13-34% faster alone, +0.8% in a four-socket synthetic cycle): no model gain.
- 16 workers per socket, zero-spin dispatch, pinning helper threads to the spare cores (-9%).
- A retained 128-thread OpenMP team made an optional barrier look 1.4x faster; without the artificial team it is 0.97x.
- Per-request `speculative.n_max` is silently ignored by this build; depth changes need a restart.
- Selection top-k loses to `std::partial_sort` above ~25,000 entries, so the new top-k is gated to short rows.

## Open

- 17.61 is not 18, and it is a test-window number. The candidate stack changes generated text (the stock attention kernel is ~1% off,
  see [the accumulation report](cpu-flash-attn-f16-accumulation.md)), so promoting it is an owner decision.
- A draft-loop restructure (catch-up folded into the first draft pass, direct draft-token pick) is written and has not run.
- No 30K-token measurement exists for the 09-20 kernels. Production measured 10.26 tok/s there on 09-19.
- The cached-versus-fresh output discrepancy recorded on 09-19 is unresolved.
