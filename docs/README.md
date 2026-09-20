# Engineering guide

This guide is the curated entry point to Duck Duck Llama. Each case study connects a concrete problem to its implementation, measured result, and remaining limits.

## Engineering case studies

| Case study | Questions it answers |
| --- | --- |
| [NUMA execution and memory bandwidth](case-studies/numa-and-bandwidth.md) | Why can using more sockets make inference slower? How do worker placement, tensor layout, and collectives interact? |
| [Quantization, layout, and correctness](case-studies/quantization-and-layout.md) | Why is a quantization label insufficient? When does repacking help, and how do numerical checks catch an incorrect optimization? |
| [Speculative decoding and request state](case-studies/speculative-decoding.md) | When does MTP increase useful output? How are acceptance, rollback, cache state, and benchmark fairness checked? |
| [Native DeepSeek on CPU](case-studies/native-deepseek.md) | What does native FP8/FP4 execution require beyond loading weights? How are lookup, sparse attention, cache limits, and model integration validated? |

## Read by task

- **Assess the results:** [measurements](results.md), [methodology](methodology.md), and [model status](models/README.md).
- **Understand the implementation:** [architecture](architecture.md), the case studies above, and [source provenance](provenance.md).
- **Run or extend the work:** [reproduction](reproducing.md), [tools](../tools/README.md), [profiles](../profiles/README.md), and [contribution guidance](../CONTRIBUTING.md).
- **Find the original experiment:** [archive index](../engineering/README.md) and `python3 tools/search_archive.py QUERY`.
- **See what remains:** [roadmap](roadmap.md).

The [previous repository overview](history/legacy-overview-20260912.md) is preserved for historical context. Its chronological runtime updates are superseded by the model guides when discussing current validation status.
- [The case for CPU + multi-channel memory](case-for-cpu-multichannel.md) — capacity economics, the 4.07x NUMA scaling result, and where CPU honestly loses
- [What is worth taking upstream, and who has to do it](upstream-candidates.md) — five candidates checked against upstream HEAD, one already fixed there, and why submission is a human's job
- [Qwen3.8-Flash-Next: where the ceiling actually is](qwen4exp-ceiling-20260915.md) — the decode-vs-context curve, 35% bandwidth extraction, the indexer finding, and eight refuted claims including our own
