# Model choice, storage, and conditional decode speeds

The latest Qwen speed target is **40+ generated tok/s at Q6**. [The current speed and headroom assessment](MODEL-SPEED-HEADROOM-20260910.md) records the observed maxima, the 25 ms/token budget, conditional bandwidth figures, and measured optimization limits. No 40 tok/s or all-model 250 GB/s result is established.

September 10 UTC steering: the user explicitly selected GLM-5.3-Flash
UD-Q4_K_XL. Download, verification, and runtime selection are complete; see [the switch report](FLASH-Q4-SWITCH-20260910.md). This supersedes
the tentative Q4 status and Q8-only selection rule below. Qwen remains
UD-Q6_K_XL. Its inventoried active target weight estimate falls from 6.964 to
6.359 GB per raw token under Q4_K_XL: an 8.69% byte reduction, corresponding
to 9.51% weight-only speed headroom at equal bandwidth. This does not establish
an MTP speed gain; the dense weights and other runtime costs remain.

Latest user correction: Full is not required to remain running or resident.
The assistant's earlier attribution of that restriction to the user was
incorrect. Evaluate each selected model against the whole server and load
models as needed. Latest Qwen requirement: a higher-precision Flash-Next
configuration exceeding 28 generated tok/s. The older Q2 measurements below
are historical baselines, not the newly requested configuration.

Follow-on execution: all six Qwen UD-Q6_K_XL shards (169.17 decimal GB) are
downloaded and hash-verified. Full has been stopped through its systemd user
service, and the Q6 runtime uses port 18095. After closing the authorized busy
Chrome instances, MTP4 measured 28.23 tok/s on code; its repeat was 27.92.
The selected wider Q8 batch build measured 28.46 on code and 21.53 on prose,
following an earlier 28.10/21.32 run. MTP3's best
prose result is 22.99. These are fresh, short prompts with a 4096-token context
setting, and do not establish a 28 tok/s floor across workloads. The selected
MTP4 run used about 137 GB/s of background-adjusted memory traffic, around
36% of 380 GB/s. The persistent launcher now selects Q6, with Q2's configuration
backed up. Prompt reuse is off by default to avoid an observed cached-extension
mismatch; default requests pass all four fresh-evaluation parity checks.
See [the Q6 trial report](QWEN-HIGH-QUANT-20260907.md).

September 7: the user wants near-lossless GLM-5.3-Flash quality, is considering
UD-Q4_K_XL for Flash, and asks whether Qwen3.8-Flash-Next makes Qwen3.8-27B
redundant. The original comparison and planning pass below was read-only.
The subsequent Q6 execution is recorded above. No model files were deleted.

## Qwen choice

Flash-Next is the stronger candidate for the main Qwen role. The publisher's
comparison reports SWE-bench Multilingual 81.0 versus 73.8, Toolathlon Verified
73.5 versus 67.1, and GPQA Diamond 91.7 versus 89.2 for Flash-Next versus 27B.
These are publisher model evaluations, not measurements of the installed GGUFs.
Source: [Qwen's comparison](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#benchmark-results).

The original Flash-Next UD-Q2_K_XL baseline measured 21.81/24.17 generated
tok/s in the September 6 prose/code samples with MTP2. The standard 27B Q8
configuration has historical general MTP results around 15-16.5 tok/s, with
slower novel-prose samples as well. These were different runs, prompts, context
settings, and runtimes; they are not a controlled head-to-head comparison.
The 27B Q4 composite-speculation profile reached 27.2-28.2 tok/s on a specialized
exact code replay. That replay result is not a general-generation rate.

Flash-Next's sparse model activates about 6B parameters per token; 27B is
dense. This gives Flash-Next substantially lower active-weight traffic and
more bandwidth headroom in the installed configurations. It does not prove
that low-bit Flash-Next preserves its published quality lead over local 27B Q8.
Keep the standard 27B Q8 artifact as a fallback until an appropriate local
quality comparison establishes that it is unnecessary.

## Storage observation

A read-only process inventory found only Full on port 18091 and Flash-Next
on 18095 loaded. No Qwen 27B inference process was found. Deleting the unloaded
27B files would free disk capacity rather than a live 27B model's inference RAM.

The inspected 27B directories contain ten GGUF files totaling 140467253248
allocated bytes across distinct inodes. The standard Q8 artifact occupies
31458004992 allocated bytes on the home filesystem, and its vision projector
occupies 927612928 bytes on the model filesystem. Keeping those two retains
about 32.39 decimal GB. The remaining 108081635328 bytes are other quantizations,
draft weights, and derivative-model/projector files. They are potential cleanup
candidates if their distinct behaviors are no longer needed; merely being
unloaded does not establish that every derivative is redundant.

This observation does not authorize deletion. The file inventory is relevant
to disk planning. Simultaneous Full plus high-precision Flash RAM residency
is not a user requirement.

## Speed projections for the three principal models

Assume one request stream and one active model with all 380 decimal GB/s
available. These are bandwidth-only projections, not achieved speeds or
proof that the complete graph can sustain the requested utilization.

| Model/configuration | Basis | At 85% | At 93% | At 100% |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next UD-Q2_K_XL | Measured MTP2 bytes per emitted token | 70.3-75.3 | 76.9-82.3 | 82.7-88.5 |
| Qwen3.8-Flash-Next UD-Q6_K_XL | Estimated weights per ordinary raw token | 46.4 | 50.7 | 54.6 |
| GLM-5.3-Flash UD-Q4_K_XL | Estimated packed weights per ordinary raw token | 22.7 | 24.8 | 26.7 |
| GLM-5.3 Full current Q4 artifact and load-time requantization | Measured MTP2 prose/reasoning bytes per emitted token | 13.0-15.6 | 14.2-17.1 | 15.2-18.4 |

All rates are tok/s. The MTP projections assume the observed traffic and
acceptance pattern survives optimization. They are not directly comparable
to Flash's raw weight-only projection, which excludes all speculation and
additional state/activation traffic. No Q4 Flash model run has been performed.

For a weight-only raw comparison, Qwen's existing layout estimate is 66.3
tok/s at 85%, 72.6 at 93%, and 78.0 at 100%. Full's measured zero-offered-draft
traffic, which still includes draft maintenance, instead implies 10.9,
11.9-12.0, and 12.8-12.9 tok/s at those bandwidth levels.

Full's favorable exact copy/edit cases with longer speculation have separate
93% projections of 29.4-33.5 tok/s; those are not its general reasoning rates.
Measured passing edit speeds were 13.59-15.70 tok/s. The fresh reasoning
samples measured 7.89-9.64 generated tok/s and exhausted their token budgets
inside reasoning, so they do not measure completed-answer latency.

## Q4 Flash inventory and quality limitation

Only bounded GGUF header ranges were fetched: 14672739 bytes in total across
six shards at publisher revision 621d456e93e926e4b52f85cff5f634358c1828f9.
Target tensor names and shapes match the installed IQ2 artifact, with 45
target layers and 8 selected experts out of 288. The file set is
199707321347 bytes. No full weight download was performed.

Canonical active weights are 14.117189624 GB per raw token; estimated current
packed layouts give 14.246164472 GB. About 8.965 GB is dense/shared/other and
5.281 GB is routed experts. Unlike the IQ2 file, the Q4 candidate preserves
many dense/shared matrices as Q8; no additional lower-precision load-time
conversion is included in this projection. Q8 x16 stores 544 bytes per 16
blocks, the same weight-byte count as ordinary Q8_0.

The user proposed Q4 as a candidate, not as a withdrawal of the near-lossless
quality requirement. The publisher's quantization table gives Q4_K_XL mean
KLD 0.049294 and a top-1 comparison of 92.22%; that statistic is not 92.22%
retained task-solving ability and does not establish near-lossless quality.
Q4 needs appropriate quality evidence before selection.
Source: [publisher analysis](https://unsloth.ai/docs/models/glm-5.3-flash#quantization-analysis).

## Local evidence

- [Q4 Flash header inventory](results/glm-flash-q4-bandwidth-inventory-0907.json).
- [Original Qwen Q2 measurements](results/qwen-even-split-existing-baseline-0906/result.json).
- [Qwen Q6 measurements and ongoing optimization](QWEN-HIGH-QUANT-20260907.md).
- [Current Full reasoning measurements](results/glm53-full-current-bandwidth-0906/result.json).
- [Full replay and other bandwidth results](MODEL-BANDWIDTH-TARGETS-20260905.md).
- [Historical 27B tuning summary](FLEET-TUNING-20260903.md).
- [27B Q4 replay measurements](FLEET-TUNING-20260903.md#19-20-toks-found--and-it-was-the-dense-model-not-the-flash-ones).
- [Near-lossless quality requirement and capacity check](GLM-FLASH-QUALITY-POLICY-20260907.md).
