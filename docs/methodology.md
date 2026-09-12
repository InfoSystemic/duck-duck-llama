# How results are measured

The project distinguishes numerical correctness, model behavior, and speed. Passing one does not imply the others.

## Evidence levels

| Level | What it establishes | What it does not establish |
| --- | --- | --- |
| Source and build verification | Identified code reconstructs, compiles, links, or loads as recorded | Valid model execution |
| Component reference check | The inspected shapes, values, state transitions, or metadata match a defined reference | Whole-model quality or throughput |
| Model comparison | Recorded requests run on real weights with the stated output and timing checks | Broad task quality, all contexts, or general hardware scaling |
| Serving observation | A particular loaded process answers requests under recorded conditions | A persistent service guarantee or reproducibility under different host load |

The archive retains failed experiments and rejected promotions. A table entry is not automatically a recommended runtime.

## Timing

**Decode tok/s** uses the server's decode interval. Client wall time includes prompt processing, downloads, queueing, and other overhead. Generated tokens may include reasoning or hit a token cap before a final answer appears. Answer latency and output quality therefore need their own checks.

Speculative runs report emitted tokens alongside draft and acceptance counts. Prompt caches and learned ngram state can warm repeated prompts; the newer A/B/A controller uses fresh servers and rejects cached timing samples, short completions, changed requests, or mismatched greedy outputs.

The [fresh-server controller](../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md) runs baseline, candidate, and baseline again. Its single sample per prompt is a useful qualification gate, not a claim of statistical significance.

## Memory bandwidth

The later bandwidth experiments use Intel memory-controller counters over stable decode windows. The Flash audit covers 48 counters and checks window coverage and adjacent idle traffic. Adjusted GB/s subtracts the larger adjacent idle value from systemwide traffic. That improves attribution but still estimates model traffic on a shared machine.

For non-speculative decode, active weight bytes per token multiplied by decode rate can provide a traffic estimate. It is not a hardware counter. With speculative output, several emitted tokens can share target work, so that product must not be presented as measured DRAM bandwidth.

The earlier local/interleaved bandwidth probes and the later requested 380 GB/s comparison denominator are different quantities. The project does not claim that every reported percentage uses one universal, proven ceiling.

## Correctness and promotion

Kernel changes use exact comparison where the intended arithmetic allows it. Fixtures include quantization formats, strided layouts, tail dimensions, rejection paths, and request-state transitions. A faster microbenchmark must then survive a matched model comparison.

Output identity is strong regression evidence for the tested prompts. It is not a comprehensive quality evaluation or proof of equivalence to a publisher's accelerator implementation. Historical Full runs with divergent output and reasoning-cap exhaustion remain explicitly rejected.

## Publication

Source files and executable configuration are preserved with hashes. Evidence JSON omits unrelated host/process inventories, full environment dumps, and response text; oversized values may be summarized by count and hash. Model weights, compiled libraries, and raw performance traces are excluded. [Archive contract](../engineering/README.md#archive-contract).

Repository CI checks publication integrity and portable tests. Full-model and CPU-specific results are recorded separately and are not rerun by CI.
