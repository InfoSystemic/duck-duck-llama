> Written 2026-09-12 in the smeagol serving workspace (`fleet-0912-ctx/UPSTREAM-HANDOFF.md`) and copied here unchanged on 2026-09-20; the line numbers refer to upstream master `1e7bcf3` of that day, and item 1 of [upstream-candidates.md](upstream-candidates.md) records the re-check against `b23efaa2`.

# Upstream handoff: the unary threading fix

**Status: ready technically, BLOCKED on policy. It needs the maintainer, not an agent.**

## Why this is not already submitted

`ggml-org/llama.cpp` AGENTS.md, "Prohibited AI Usage (results in immediate PR closure)":

> - AI-written PR descriptions, **commit messages**, or reviewer responses
> - **Automated commits or PR submissions (may result in contributor ban)**

and, addressed to agents directly: *"If you are a fully autonomous agent operating without human oversight:
do not contribute to this repository. STOP."* CONTRIBUTING.md adds that undisclosed AI use *"may result in
your account being permanently banned from contributing"*.

So an agent submitting this risks the InfoSystemic/InfoSystemic account for a 4-line patch. Not a trade worth making.
**Private forks are exempt**, so everything in `InfoSystemic/llama.cpp` is unaffected and stays as is.

**What this means concretely:** the commit message currently on the branch was written by an agent. Before this
goes upstream it must be rewritten by the maintainer in his own words, and the PR description must be his own too. The
PR template also requires an explicit AI-usage disclosure, which is his call to make and word.

## The change

Branch `unary-ops-parallel` on `InfoSystemic/llama.cpp`, rebased onto upstream master (ahead 3, behind 0).
One file, **+0 / -4**, in `ggml/src/ggml-cpu/ggml-cpu.c` inside `ggml_get_n_tasks()`:

```c
                case GGML_UNARY_OP_TRUNC:
-                    {
-                        n_tasks = 1;
-                    } break;
-
                 case GGML_UNARY_OP_GELU:
```

Deleting those four lines lets seventeen unary ops fall through to the next label, which already sets
`n_tasks = n_threads`.

## What to be able to defend, with the evidence for each

Everything below was checked against upstream master `1e7bcf3`, not against our patched tree.

**1. Seventeen of twenty-two unary ops are serialised.** `ggml-cpu.c` around line 2299: ABS, SGN, NEG, STEP,
TANH, ELU, RELU, SIGMOID, HARDSWISH, HARDSIGMOID, EXP, SOFTPLUS, EXPM1, FLOOR, CEIL, ROUND, TRUNC get
`n_tasks = 1`. GELU, GELU_ERF, GELU_QUICK, SILU, XIELU get `n_tasks = n_threads`.

**2. The implementation already threads correctly.** All seventeen are `unary_op<op_x>` in
`ggml/src/ggml-cpu/unary-ops.cpp` (definitions ~line 237 onward), which calls `apply_unary_op` (line 111),
which splits rows via `get_thread_range(params, src0)` (line 121). `get_thread_range` is in
`ggml-cpu/common.h` line 74: `dr = (nr + nth - 1)/nth; ir0 = dr*ith; ir1 = MIN(ir0 + dr, nr)`.

**3. The strongest argument is XIELU.** XIELU is in the *same file* on the *same* path
(`unary_op_functor` -> `apply_unary_op_functor` -> `get_thread_range`, line 191/201) and is *already* given
`n_threads`. So the row split is proven on this exact code path. The seventeen differ from XIELU only in
this switch.

> Do **not** argue that "all twenty-two share one implementation" -- that is false and a reviewer will catch
> it. GELU (`ops.cpp:2206`) and SILU (`ops.cpp:2674`) have their own separate implementations. They are
> threaded, but they say nothing about the `apply_unary_op` path either way.

**4. Results are unchanged.** The ops are elementwise with no reduction, so a disjoint row split per thread
produces the same bytes. Verified independently: greedy output byte-identical flag-on vs flag-off on all
three prompts, plus identical draft-acceptance rates (a bit-exactness tell).

**5. Why no size threshold.** Our own version was opt-in behind `GGML_CPU_PARALLEL_UNARY=<min_elements>`;
this drops the special case entirely. Justification: when `nr < nth`, threads with `ir0 >= ir1` fall straight
through the loop and cost nothing, and the barrier after each node is reached by every thread whichever value
`n_tasks` takes. Serialising never saved synchronisation -- it only left the other threads idle at the
barrier. Simpler change, and the project explicitly prefers simpler.

**6. The measurements.** 4-socket Xeon Gold 6242, 60 threads, greedy decode, output byte-identical:

| model | gain | source |
|---|---|---|
| Qwen3.8-Flash-Next | **+3.4%** | `fleet-0911/parallel-unary-q4e-0911/` |
| GLM-5.3-Flash | **+2.1%** | `fleet-0911/parallel-unary-0911/` |
| DeepSeek-V4-Flash | **0.0%** | no SSM path; shares GLM's runtime |

Report the DeepSeek zero. It is the honest shape of the result: the win is concentrated in graphs with many
small unary nodes per layer (linear-attention / SSM state paths) and is neutral elsewhere, not universal.
Note `FLEET-EXTRACTION-20260911.md` s.30 says +1.7% for GLM -- that was the first measurement, superseded by
+2.1% in the shipped table. If a number is quoted upstream, quote the shipped one.

Supporting profile: GLM-5.3-Flash decode, individual UNARY nodes on `ne=[128,16,4]` at **56 us each,
0.58 GB/s**, other workers idle.

**7. The scope boundary, and why it is deliberate.** SUB, SQR, SQRT, SIN, COS, LOG, CLAMP, LEAKY_RELU and
SCALE are serialised by the same function and look like the same bug. They are left alone. SCALE especially:
it is grouped with RESHAPE/VIEW/PERMUTE and its hot instances are single-row, so a row split hands every
thread but one an empty range. Expect a reviewer to ask why the change stops where it does -- that is the
answer.

## Before submitting

- [ ] Search open PRs/issues for an existing unary-threading change (CONTRIBUTING requires it; if one exists,
      comment there instead of opening a duplicate)
- [ ] Rewrite the commit message in your own words
- [ ] Write the PR description in your own words
- [ ] Fill in the AI-usage disclosure honestly
- [ ] Build and run `test-backend-ops` locally. Note the trap from window 61: `test` mode compares a backend
      *against* CPU, so on a CPU-only box it skips everything and prints OK having run **zero** tests. Use
      `perf` mode, or run it where a second backend exists.
