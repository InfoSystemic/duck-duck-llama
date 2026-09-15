# The cross-socket all-reduce writes every result four times — three of them pull the cache line first

Tensor-parallel inference on a multi-socket CPU ends every split matrix multiply
with an all-reduce: each device owns a slice of the partial sums, reduces it, and
the summed result must end up in **all** device-local copies so the next operation
can read it locally. The reduction arithmetic is trivial. The stores are not.

## The problem

Reducing `n` devices over `ne` elements, each device handles `ne/n` of them and
writes the sum into all `n` copies:

```c
for (int64_t k = t_start; k < t_end; ++k) {
    const float sum = (p0[k] + p1[k]) + (p2[k] + p3[k]);
    p0[k] = sum; p1[k] = sum; p2[k] = sum; p3[k] = sum;
}
```

Three of every four stores land in another socket's memory. A normal store to a
line the core does not own issues a **read-for-ownership** first: the line is
pulled across the interconnect, modified, and later written back. The read is
pure waste — the whole line is about to be overwritten.

## The change

`_mm512_stream_ps` writes 64 bytes without acquiring the line, so the RFO
disappears on the three remote copies. The additions are unchanged and happen in
the same order, so results are **bit-identical** — which the parity check
confirms rather than assumes.

Requirements that make it correct rather than merely fast:

- `_mm_sfence()` before the exit barrier. Non-temporal stores are weakly ordered
  and the other devices must not observe the barrier before the data.
- All four pointers must share alignment mod 64, checked at runtime, with a
  scalar prologue to the first aligned index and a scalar tail. If they disagree,
  the original loop runs unchanged.
- The enclosing library is compiled **without** AVX-512 flags in a stock build.
  The first build of this change compiled the entire block out and produced a
  byte-identical object; the giveaway was the object size not moving. Add the ISA
  flag for that one file and verify `vmovntps` appears in the disassembly.

## Measured

Qwen3.8-Flash-Next (`UD-Q6_K_XL`, 158 GB, 512 experts / 10 active, 48 layers),
4-socket Xeon Gold 6242, 60 worker threads, 381 GB/s measured aggregate. Same
window, arms back to back, identical draft acceptance:

| | baseline | non-temporal | change |
| --- | ---: | ---: | ---: |
| speculative cycle, 3 prompts (ms) | 143.2 / 142.7 / 141.2 | 130.7 / 132.1 / 134.0 | **-8.7 / -7.4 / -5.1%** |
| decode (tok/s) | 23.40 / 22.60 / 25.64 | 25.63 / 24.40 / 27.02 | +5 to +10% |
| prefill at 2.25K (tok/s) | 119.0 | 134.1 | +12.7% |
| prefill at 6.75K (tok/s) | 107.6 | 119.0 | +10.6% |
| greedy-text parity | — | IDENTICAL | bit-exact |
| 10-question factual battery | — | 10/10 identical | |

GLM-5.3-Flash (`UD-Q4_K_XL`, 1M context, 2 slots), same kernel, same engine
family:

| | baseline | non-temporal |
| --- | ---: | ---: |
| decode (tok/s) | 11.87 / 13.84 / 13.76 | 12.14 / 14.20 / 13.99 |
| two concurrent streams, aggregate | 17.1 / 20.1 / 20.4 | 17.7 / 20.8 / 21.1 |
| greedy-text parity | — | IDENTICAL |

The second model gains far less — about 2% single-stream against 5 to 10% — which
is the useful part of the result. The size of the win tracks how much of a graph's
time sits in cross-socket reduction, and that is architecture-specific. Both were
promoted anyway, because a bit-identical change with a positive sign costs nothing.

## Why this is not upstream

Mainline llama.cpp does not do multi-socket tensor parallelism at all. The NUMA
work being discussed upstream **mirrors the whole model per node**, doubling memory
and removing the reduction rather than optimising it. This change only exists for
engines that split tensors across sockets and pay for the reduce.

---

## A failed hypothesis, kept here because the control is the lesson

A third patch was briefly included in this branch and has been removed. A Q4_K_M
build of the same architecture aborted at graph build:

```
cannot infer split for op=ADD srcs=[alpha-0 axis=10(MIRRORED), ssm_dt.bias axis=0(SPLIT)]
```

because that file quantises `ssm_alpha`/`ssm_beta` where the Q6_K build keeps them
F32, and the generic splitter's inferred axis depends on tensor **type**, not only
name. The proposed fix mirrored the whole small recurrent family so the rule would
be type-independent.

It does not work, and the arm that proved it was the control: the **existing,
working Q6_K model** run through the same library with the rule enabled failed at
the same new assert as the Q4 file (`src_ss[3].axis == SPLIT_AXIS_1`). So the rule
does not reconcile the inconsistency, it relocates it — something downstream
requires one of those tensors split on a specific axis.

Two arms would have shown a fix that was not one: rule-off fails at the original
assert, rule-on gets further. Only the third arm, the model already known to work,
distinguishes "repairs the new file" from "breaks the graph for everything". Any
split-rule change should be run against a model that currently loads, every time.
