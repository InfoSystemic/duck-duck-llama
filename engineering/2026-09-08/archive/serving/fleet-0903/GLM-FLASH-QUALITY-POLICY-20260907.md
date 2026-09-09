# GLM-5.3-Flash: preserve quality before optimizing speed

User correction, September 7: the user did not instruct us to keep the existing
Full service running. The assistant incorrectly attributed that restriction
to the user. It is withdrawn. Evaluate each selected model against the whole
server, with models loaded as needed; simultaneous residency beside Full is
not a quantization-selection requirement. Earlier notes claiming otherwise
are superseded by this correction.

User requirement, September 7, 2026: GLM-5.3-Flash is the near-lossless-quality
option. Qwen3.8-Flash-Next is the option when additional speed is needed.
Lowering GLM Flash's target-weight precision to meet a speed or bandwidth
number does not satisfy this requirement.

Later September 7 steering: the user proposed UD-Q4_K_XL as a candidate while
asking for speed ceilings. It is now included in the [model comparison and
header-based estimate](MODEL-CHOICES-20260907.md). This tentative proposal does
not establish that Q4 meets near-lossless quality or authorize deployment.

This policy supersedes the earlier IQ2 speed-first selection and the proposed
IQ3/IQ4 comparison as a way to choose the desired serving configuration.
Historical IQ2 measurements remain valid measurements of that artifact, but
they do not qualify it for the new quality requirement. The current runtime
checks establish correctness relative to the quantized reference, not quality
equivalence to the original released model.

## Candidate and reference

- Primary candidate: Q8_0, subject to reference-quality and runtime validation.
  Its name and bit width alone do not establish near-lossless task quality.
- Secondary candidate: UD-Q6_K_XL, only if evidence establishes that it meets
  the same quality requirement. Memory or speed pressure is not sufficient
  grounds to select it.
- Compare against the official released checkpoint, with checkpoint revision,
  conversion provenance, tokenizer, template, model graph, reasoning settings,
  and context recorded. The official configuration includes FP8 quantization
  with higher-precision exclusions. A BF16 GGUF conversion is not evidence
  of access to original pre-quantization training weights; verify provenance.
- Do not up-convert the existing IQ2 files and call the result higher quality.
  Obtain candidate weights from the appropriate higher-precision source.

The publisher's quantization table reports mean KLD of 0.450148 for IQ2_XXS,
0.283772 for IQ3_XXS, 0.049294 for Q4_K_XL, and 0.019007 for Q6_K_XL.
It does not list a Q8 result in that table. Its top-1 comparison is not a
measurement of coding, reasoning, or agentic success retention. None of those
numbers alone certifies near-lossless quality on the user's work.

## Quality acceptance and subsequent optimization

Use paired evaluation against the released-model reference on representative
coding, reasoning, tool-use, and long-context tasks, including completed
outputs and executable correctness checks where applicable. Include vision
evaluation if the deployment will serve multimodal requests. Record sample
size, effect size, uncertainty, and an explicit near-equivalence margin before
using an evaluation to qualify a candidate. A small smoke suite or a failure
to find a statistically significant difference is insufficient evidence.
No numeric task-quality margin has been agreed or claimed satisfied yet.

Logit divergence and perplexity are diagnostic checks alongside task results.
Keep high-precision attention/cache/state settings during initial comparison.
Validate the final combined configuration, including any cache changes.
Speculation is acceptable when target verification preserves the intended
target behavior; draft acceptance and higher token throughput are not quality
evidence by themselves.

Once a candidate meets the quality requirement, optimize NUMA placement,
load balance, exact weight layouts, kernels, scheduling, and verified
speculation. Continue measuring actual memory-controller traffic and useful
generation speed. Bandwidth utilization is subordinate to quality and must
not be improved merely by increasing unnecessary traffic.

The earlier 30-35 raw tok/s projections describe UD-IQ2_XXS only. Recalculate
traffic and speed for the selected higher-precision tensors and runtime;
do not carry those projections into this policy as a promised delivery rate.

## Capacity check, 2026-09-07 07:24:44 UTC

The following is a historical occupancy snapshot, not a requirement to keep
these models resident together. Read-only host inspection found 755.50 GiB
physical RAM, 458.24 GiB Full RSS,
80.03 GiB Qwen RSS, and 145.62 GiB MemAvailable. Even the optimistic estimate
of MemAvailable plus all Qwen RSS is only 225.65 GiB. It is not a guarantee of
allocatable memory or authorization to stop Qwen.

Publisher GGUF revision: `621d456e93e926e4b52f85cff5f634358c1828f9`.
The following are full text-weight file totals, not measured runtime RSS:

| Candidate/reference artifact | Shards | Bytes | Decimal GB |
| --- | ---: | ---: | ---: |
| Q8_0 | 8 | 340981966112 | 340.98 |
| UD-Q6_K_XL | 7 | 291833111712 | 291.83 |
| BF16 | 14 | 641641064192 | 641.64 |

The Q8 file set is about 317.56 GiB. Full plus the entire Q8 file set already
exceeds physical RAM before other workloads and runtime buffers. Files can
include weights that are not all resident at once, so this arithmetic is a
planning check, not a measured loading result. The current available-memory
check did not establish a viable fully resident Q8 or Q6 deployment beside
Full. That simultaneous-residency condition is not required. The full server's
755.50 GiB capacity is the selection budget, with runtime and host overhead
accounted for separately. Resolve loading and storage without reducing quality
to accommodate an unsupported requirement to keep Full resident.

The model filesystem had 104148377600 bytes available; the home filesystem
had 234110545920 bytes available. Neither alone has room for a new complete
Q8 or Q6 file set. A storage plan is needed before staging weights. No existing
model files were deleted or replaced.

No production runtime, model, launch configuration, or service state changed
during this capacity assessment. No model download, load, benchmark, or trial
was queued. The assistant's earlier statement that Full must remain running
was not a user instruction and must not be carried forward as one.

## Sources and evidence

- [Publisher quantization analysis](https://unsloth.ai/docs/models/glm-5.3-flash#quantization-analysis),
  read as Markdown on September 7. The tabulated values, rather than the
  inconsistent rounded prose elsewhere on that page, are used above.
- [Pinned GGUF files](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9).
- [Official model configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json).
- [Planning snapshot](results/glm-flash-quality-planning-0907.json).
- [Historical IQ2 runtime validation](GLM-FLASH-VALIDATED.md).
