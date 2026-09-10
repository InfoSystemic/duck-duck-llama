# Q6 single-activation experiment, September 10 UTC

The candidate preserves exact outputs but does not qualify for a model trial.
The 250 GB/s objective remains unmet for GLM Flash, GLM Full, and Qwen.
No model was loaded, no serving profile was promoted, and the separate Qwen
service was preserved throughout this experiment.

## Arithmetic and private build

The existing Q6 helper moves the integer zero-point correction before scaling.
This removes one vector integer multiplication per sub-block. Packed weight
storage, six-bit codes, scales, and FP32 accumulation order remain unchanged.
The earlier independent scalar/native proof passes 432 cases and 146,880
output comparisons. The new runtime uses only its single-activation branch;
expert tiles, routing, and MTP row grouping retain the corrected gather baseline.

The new single-core comparison qualifies four cases under the existing limit
of four background CPU cores. Candidate/native speed ratios are:

| Output rows | Cached matrix | 512 shuffled matrices |
| --- | ---: | ---: |
| 64 | 1.2315 | 1.0832 |
| 32 | 1.1903 | 1.1335 |

These are component rates, not model tok/s or measured IMC bandwidth. One
earlier attempt failed the background limit and is retained as invalid.
The two September 8 timing attempts never obtained a quiet window; they did
not measure a kernel regression.

The unchanged x86 object and composed baseline library rebuild byte for byte.
The first private compile failed because the export macro already contained
`extern`. The separate `0910b` version fixes the C linkage declaration and
inherits the verified baseline. Both versions remain archived.

Candidate CPU SHA-256:
`bf09cb90e62569cf111917c04b14fb1cf74fabda74fdd71d70a05d5514b46074`.
Its parent is corrected gather CPU
`12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7`.
The opt-in flag is `GGML_CPU_QWEN_Q6_PACKED_SINGLE=1`; its default is off.
Only `repack-x86.cpp.o` changes. Diagnostic counters are enabled separately
for correctness checks and disabled for timing.

## Actual graph checks and timing limit

All 210 expert graph cases pass complete output comparison, input and weight
preservation, and execution-counter checks. They cover 512 experts, ten and
eight routes, 1/4/15 workers, padded input, an extra gate consumer, and token
counts through 64. Both selected sockets produce the same outputs.

All ten four-NUMA arms also pass complete output comparison and numeric
references, including Q6 gate/up, unchanged Q8 down, shared dispatch cleanup,
one/three/five/64-token inputs, and an unfused five-token path.

Four of eight graph timing arms fail the unchanged background limit. The two
fully qualified cold pairs give geometric mean speed ratios of 0.9849 on
socket 0 and 1.0006 on socket 2. Their single-token ratios are 0.9605 and
0.9796. These use scattered synthetic route IDs; they are not a capture of
the model's routing distribution. Neither qualified pair meets the declared
2% graph gate, and a complete repeated timing comparison is not established.

The single-core gain therefore does not justify a model speed claim.
`run_qwen_q6_single_0910.py` is prepared but unexecuted; its eligibility check
rejects these results before loading a model. There are zero new model
bandwidth measurements from this experiment. The host audit confirms that
all owned components exited, private model ports and the lifecycle lock are
free, Full remains inactive, and the peer's selected preset is unchanged.

## Higher precision follow-up

The pinned Q8 file is 188.225 GB versus 169.165 GB for Q6 XL. Among 1,152
inventoried target tensors, it raises 94 expert gate/up tensors from Q6 to Q8
but also quantizes 168 currently FP32 HC-injection and SSM tensors to Q8.
It is therefore not a uniform precision upgrade.

An expert-only Q8 mixture could preserve the existing FP32 tensors. Its
estimated additional expert storage is 19.097 GB, with 0.373 GB more active
weight traffic per ordinary raw token. That mixture has not been built or
tested, and no quality or speed improvement is established. The static
inventory comparison does not verify payload identity or certify model quality.

## Evidence

- [Qualified single-core timings](results/qwen-q6-packed-single-component-qualified-0910/result.json)
- [Private build](results/qwen-q6-single-build-0910b/result.json)
- [Graph validation](results/qwen-q6-single-validation-0910/result.json)
- [Qualified-pair assessment](results/qwen-q6-single-assessment-0910.json)
- [Host cleanup audit](results/qwen-q6-single-host-audit-0910.json)
- [Q8 tensor precision assessment](results/qwen-q8-precision-assessment-0910.json)

The Flash Q8 handoff experiment remains prepared and unexecuted pending the
existing request to pause and restore the independently launched Qwen service.
This is not a requirement to keep Full resident.
