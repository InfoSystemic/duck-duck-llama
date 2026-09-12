# Native DeepSeek-V4.1 on CPU

This work goes beyond fitting a converted file into memory. It implements and checks the native mixed-precision operators, lookup state, attention behavior, storage path, and serving lifecycle needed to execute the released text model on CPUs.

## From components to a real model

The native path retains FP8/FP4 checkpoint storage and adds CPU kernels, quantization reuse, grouped expert projections, native hyper-connection work, and sparse attention. The fused configuration raised matched short-request decode from 1.097 to 1.781 tok/s; a later independent endpoint audit measured 1.800 tok/s. [CPU tuning and model evidence](../../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md).

That audit uses the fixed prompt `Hi.`, a cached ten-token response including EOS, and nine decode forward passes. It is a meaningful integration and regression result with real weights. It does not establish long-form quality, cold-cache performance, or general 1.8 tok/s serving throughput.

## Exact Engram lookup

The lookup work joins token history, hashing, row-shard dispatch, and native FP8/E8M0-to-BF16 decoding through a C API. Eight variants cover scalar/AVX-512 execution, division/reciprocal hashing, and one/four shards.

| Checked quantity | Result |
| --- | ---: |
| Exact BF16 output comparisons | 267,583,488 |
| Exact selected-row comparisons | 1,044,096 |
| Rejected calls preserving state/output | 112 |
| Actual checkpoint rows checked separately | 48 |

The large fixture uses sparse address reservations and populated test rows, not a fully resident 202.8 GB table. The actual-row check retrieves bounded checkpoint ranges separately. [Implementation, oracle, and independent audit](../../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md).

## A faster candidate can still lose promotion

The revised exact FP4 row16 kernel produced a 7.4% gain in its matched model trial. Its later HTTP comparison gained 3.3%, below the predeclared 5% serving threshold, so the previous implementation was restored. The repository keeps the candidate and the failed promotion evidence. [Timing and selection record](../../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md).

## Context and cache engineering

The extended-context candidate bounds initial prefill, follows the official single-token continuation path, resizes derived caches, and replaces a fatal monotonic Engram-row limit with bounded memory/persistence. Five fault-injection tests reproduced and then fixed interrupted-write quota errors, alongside four existing context tests. [Context candidate and recovery evidence](../../engineering/2026-09-12/archive/serving/fleet-0912/deepseek-context/README.md).

This is still a component-tested candidate. The native endpoint's audited context was 256 tokens; no validated 4K/16K full-model result is claimed. Novel tokens may require many small range fetches, so cold storage access remains a central usability problem.

## Separate llama.cpp port

The isolated JigSaw integration is pinned to `3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5`. Local fixes correct Linux prefetch alignment and a missing standard include. The original source fails the focused prefetch regression; the patch passes, and CPU benchmark/server builds succeed. Full-checkpoint inference in this port remains untested. [Port assessment and source evidence](../../engineering/2026-09-12/archive/serving/fleet-0912/upstream/DEEPSEEK-V41-UPSTREAM-20260912.md).
