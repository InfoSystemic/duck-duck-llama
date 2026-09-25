# MiMo-V2.6-Pro-RL on four CPU sockets, September 21–23

This benchmark covers a 1.02-trillion-parameter mixture-of-experts model (518 GiB MXFP4/Q8_0 GGUF) on four Xeon Gold 6242 sockets, with no GPU. The machine's measured aggregate memory read is 381.6 GB/s. Every number comes from the [2026-09-23 snapshot](../engineering/2026-09-23/README.md); the model guide is [here](../docs/models/mimo-v26-pro.md).

## Method

- **Speed by workload.** [`draft-probe.py`](../engineering/2026-09-23/archive/serving/mimo-v26-pro/draft-probe.py) sends five prompts that span predictability: verbatim repetition, counting, a memorised list, code, and open prose. Decode rate depends on the text, so every speculative number below is quoted with its workload.
- **Numerical parity.** [`golden.py`](../engineering/2026-09-23/archive/serving/mimo-v26-pro/golden.py) compares three fixed greedy prompts token by token against a recorded reference, along with each token's top-1 log-probability. "Identical" means the same 48 tokens and log-probabilities equal to four decimal places (maximum |Δ| 0.0000). Kernel-level bit-identity is checked separately, by output hashes and max-difference checks in the kernel benchmarks.
- **Depth.** [`depth-probe.py`](../engineering/2026-09-23/archive/serving/mimo-v26-pro/depth-probe.py) extends one append-only session. It reports the prefill rate of the newly added tokens and the decode rate at each depth.
- **Shared machine.** Other sessions' jobs ran on the machine. A job on any one of the four tensor-parallel nodes slows every operation by about 30%. Numbers marked *contended* were taken under such load. Comparisons between builds use interleaved runs, or kernel benchmarks timed by the minimum over many iterations.

## From 1 to 27 tok/s

| Configuration | Decode tok/s | Evidence |
| --- | ---: | --- |
| Engine without tensor parallelism, no speculation | 1.03 | [plain](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/mimo-plain.json) |
| Four-way tensor parallel, no speculation (129.5 GiB per node) | 7.94 | [TP4](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/mimo-tp4-nospec.json) |
| MTP heads, draft depth 1 / 2 / 3 (acceptance 0.110 / 0.062 / 0.042) | 6.21 / 5.03 / 4.18 | [MTP detail](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/spec-detail-mtpfix-d1.txt) |
| DFlash, first working build, n_max 7 (verbatim / counting / code / list / prose) | 19.2 / 20.4 / 13.8 / 11.7 / 5.5 | [speculation notes](../engineering/2026-09-23/archive/serving/mimo-v26-pro/STATE-SPECULATION-20260922.md) |
| DFlash with `p_min` 0.5 (same workloads) | 19.2–20.4 / 13.4 / 11.7 / 8.8 | [final numbers](../engineering/2026-09-23/archive/serving/mimo-v26-pro/STATE-SPECULATION-20260922.md#final-numbers-and-the-q4_k-trade), [`p_min` sweep](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/dflash-pmin-sweep.md) |
| Build 0922b: grouped-query flash attention, multi-column GEMM, F32 projector | 22.3 / 25.3 / 15.7 / 12.7 / 8.6 | [deploy 0922b](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/deploy-0922b/) |
| Build 0922c: activation block sums computed once (output identical to 0922b) | **22.8 / 26.9 / 16.3 / 13.2 / 8.8** | [deploy 0922c](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/deploy-0922c/) |

Builds 0922d through 0922j reproduce 0922c's golden output exactly. They were deployed while other sessions' jobs held the machine, so they have no quiet-box workload numbers. Their compute speedups are in the kernel table below.

The spread across workloads is inherent to a block drafter. When the text is predictable, one verify pass yields 7.4 tokens. On open prose it yields about 2, and each rejected row still costs a row of expert bandwidth.

## Draft length and `p_min`

With the fixed-length drafter on open prose (`p_min` 0), longer drafts were slower:

| Draft length n | 7 | 6 | 5 | 4 | 3 | 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 6.28 | 6.74 | 6.63 | 7.07 | 7.17 | 8.04 |
| Tokens per cycle | 2.38 | 2.31 | 2.08 | 1.98 | 1.76 | 1.69 |
| ms per cycle | 374.8 | 339.0 | 309.6 | 275.9 | 242.6 | 207.2 |

A fixed length is the wrong control. `p_min` truncates the block at the first position whose top-candidate probability falls below the threshold. Both prompts below use n = 7:

| `p_min` | 0.0 | 0.3 | **0.5** | 0.7 | 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open prose: tok/s (acceptance) | 6.23 (0.20) | 7.73 (0.38) | **8.80 (0.73)** | 7.91 (0.73) | 7.69 (1.00) |
| Code: tok/s | 13.45 | 13.36 | **13.40** | 13.07 | 14.01 |

`p_min` 0.5 raises the worst case by 41%, and code is flat across the surface.

The optimum moves with the cost of a cycle. With Q4_K attention the cycle is cheaper, and n = 4 with `p_min` 0 wins outright; code reaches 15.34 tok/s against 14.05. After build 0922b made the verify pass cheaper, the setting was re-checked across all five workloads, interleaved to cancel drift. Shorter drafts lost on the geometric mean: n = 5 by 3.9% and n = 4 by 12.3%.

### The cycle model

Over draft lengths 2–7, the verify cycle fits `cycle_ms = 99.2 + 36.83 · f(rows)` to within 0.7% ([fit](../engineering/2026-09-23/archive/serving/mimo-v26-pro/fit-cycle.py)). The terms:

- `f(rows) = 384 · (1 − (1 − 8/384)^rows) / 8` is the expected number of distinct expert sets that `rows` tokens touch.
- **99.2 ms** is everything read once per cycle: attention, the router, the output head, and one drafter forward pass.
- **36.83 ms** is one expert set: 10.3 GiB at 300 GB/s, which is 79% of the measured wall.

At that duty cycle, faster serving code has little left to gain; nearly all of the remaining speedup would have to come from more accepted tokens per cycle.

## Depth: before and after the attention kernel

| Depth | Prefill tok/s, before → after | Decode tok/s, before → after | Draft acceptance, after | ms per cycle, after |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 35.5 → **62.2** | 7.68 → **18.45** | 0.98 | 315 |
| 16K | 31.4 → **54.6** | 5.29 → **8.56** | 0.55 | 288 |
| 32K | — → **43.8** | — → **9.25** | 0.65 | 292 |
| 64K | 21.0 → **33.1** | 1.98 → **6.36** | 0.48 | 364 |

Before the fix, decode fell 3.9× between 4K and 64K. The cause was not KV bytes: the full-attention layers add only about 1.3× to the per-token read at 64K. It was the attention kernel, which streamed the whole cache once per query row and head ([report](cpu-flash-attn-gqa-splitkv.md)).

After the fix, the verify cycle is nearly flat from 4K to 64K (288–364 ms). Decode at depth is now set by draft acceptance.

The 4K figure comes from a very predictable continuation (acceptance 0.98), so compare cycle times across builds rather than tok/s.

## Kernel builds, all output-identical from 0922c on

| Build | Change | Measured effect |
| --- | --- | --- |
| 0922b | Grouped-query split-KV flash attention; multi-column x16 GEMM | Attention 11× at 64K × 8 rows. Prompt matmuls went from one GEMV per token to 8 columns per weight load. |
| 0922c | Activation block sums once per matmul | Short-context workloads +2–6% (table above). Smoke-test prefill 60.5 → 69 tok/s. |
| 0922d | MXFP4 E8M0 scales converted with six AVX-512 operations | Exhaustively equal on all 256 codes. |
| 0922f | K tiles transposed 16×16 in registers; L1-blocked tile GEMMs; 64-row chunks for prompt batches | Attention at 64K × 8: 16.1 → 10.6–11.9 ms. Proxy prefill +8–14%. |
| 0922g | Even-split kernel calls; 512-row MoE-down tiles; SCALE threaded | MoE gate +16%, dense +4%, MoE down 1.21× (kernel benchmark). |
| 0922i | Paired 16-row groups (one register broadcast feeds two dot products); exact MoE weighted-sum fusion; 1024-token micro-batches | Single core: dense 1.26×, MoE gate 1.21×. MUL + ADD 7.5% → 2.5% of prefill. |
| 0922j | Vectorised row max in the attention softmax | Softmax share 19.9% → 10.7%. Attention 1.25–1.4×. Proxy prefill from 0922g to 0922j: 873–889 → 1,050–1,128 tok/s. |

On the full model, contended (load average 66), 0922j prefilled an 8,031-token prompt at 61.5 tok/s and decoded at 9.5 tok/s. The quiet-machine record for 0922b was 62 / 55 tok/s prefill at 4K / 16K.

## Quality gates that held and one that did not

- **Speculation did not change output.** Throughout the DFlash bring-up and the sweeps, the golden prompts produced the same tokens with and without the drafter, with log-probabilities equal to four decimals. A drafter proposes; the target decides.
- **The attention kernel changed output on purpose.** The old kernel summed V in FP16, with relative error 1e-3 at short context and 4e-2 at 64K; the new one reaches 1e-7. Against the earlier build, two golden prompts are token-identical and the third flips at a known near-tie at token 11. Draft acceptance, an independent check on the target, was equal or higher on every workload.
- **Q4_K attention failed.** It was faster: counting and verbatim 21.3–22.9 tok/s, code 15.3, prose 10.0. But one golden prompt diverged at token 11 of 48 (maximum |Δ log-probability| 0.181). It is not used.
- **Functional checks on every deploy:**
  - smoke test 7/7;
  - real-image vision 4/4, including exact OCR of a 1440×1024 screenshot;
  - audio 12/12 words;
  - tool calls 0/20 malformed at temperature 1.0.

  Evidence is in [results/](../engineering/2026-09-23/archive/serving/mimo-v26-pro/results/).
