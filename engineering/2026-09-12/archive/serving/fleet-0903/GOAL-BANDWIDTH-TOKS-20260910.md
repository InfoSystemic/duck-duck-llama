# Goal: 60%+ of ~380 GB/s and 20+ tok/s on all four models — 2026-09-10

Goal as stated: **DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next, GLM-5.3-Flash and GLM-5.3 (Full)
each at >= 60% of ~380 GB/s (= 228 GB/s) and >= 20 generated tok/s.**

## 0. Scoreboard (2026-09-10)

| # | line | target | measured | pass |
|---|---|---|---|:--:|
| 1 | GLM-5.3-Flash Q4 tok/s | >= 20 | 16.6 prose / 17.3 code (512-tok, counters) | **no** |
| 2 | GLM-5.3-Flash Q4 bandwidth | >= 228 GB/s | 196.8 / 198.8 (52%) | **no** |
| 3 | Qwen3.8-Flash-Next Q6 tok/s | >= 20 | 24.0-32.2 | **YES** |
| 4 | Qwen3.8-Flash-Next Q6 bandwidth | >= 228 GB/s | 138.6-152.6 (38%) | **no** (see s.1 — wrong target for this model) |
| 5 | GLM-5.3 Full Q4 tok/s | >= 20 | 7.9-9.6 | **no** (arithmetically out of reach, s.1) |
| 6 | GLM-5.3 Full Q4 bandwidth | >= 228 GB/s | 196.7-199.0 (52%) | **no** |
| 7 | DeepSeek-V4.1-Flash tok/s | >= 20 | 1.78 (PyTorch reference) | **no** (needs an engine port, s.7) |
| 8 | DeepSeek-V4.1-Flash bandwidth | >= 228 GB/s | not measurable in the Torch path | **no** |

1 of 8 lines passes today. Lines 4, 5, 6 and 8 are not reachable by tuning and the reasons
are arithmetic, not effort — read section 1 before spending more on them. Line 1 is the
only one within striking distance (~+15%), and it carries line 2 with it.

## 1. The two criteria are partly in conflict — read this first

Bandwidth and speed are linked by one identity:

```
GB/s  =  (GB of DRAM traffic per generated token)  x  (tok/s)
```

So "228 GB/s **and** 20 tok/s" is not two independent targets. It requires the model to
move **>= 11.4 GB of DRAM traffic per generated token**. A model that generates a token
more cheaply than that *cannot* satisfy both at once, no matter how fast it runs.

| model + config | tok/s | GB/s | % of 380 | GB per generated token | tok/s needed for 60% | GB/s implied at 20 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| GLM-5.3-Flash Q4, MTP2 | 16.6 | 198.7 | 52% | **11.96** | 19.1 | 239 (63%) |
| GLM-5.3 Full Q4, MTP2 | 8.8 | 197.8 | 52% | 22.55 | 10.1 | **451 — exceeds the machine** |
| Qwen3.8-Flash-Next Q6, MTP4 | 28.1 | 145.6 | 38% | 5.18 | **44.0** | 104 (27%) |
| DeepSeek-V4.1-Flash (PyTorch) | 1.78 | n/a | n/a | n/a | n/a | n/a |

Three consequences, and they matter more than any tuning result below:

- **GLM-5.3-Flash Q4 is the one model where both criteria land together.** At 11.96 GB per
  generated token, simply reaching 20 tok/s puts it at ~239 GB/s = 63% of capacity. Both
  criteria are met by the same work. This is the model to push.
- **Qwen3.8-Flash-Next already passes the speed bar (24-32 tok/s) and fails the bandwidth
  bar only because its speculation works well.** MTP4 makes a generated token cost 5.18 GB
  instead of ~9.8 GB raw. To reach 60% utilisation it would have to hit **44 tok/s**, or
  give up speculation and get slower. Here a *lower* GB/s number is the better engine.
- **GLM-5.3 Full cannot meet both on this hardware.** 20 tok/s at its current 22.55 GB per
  generated token needs 451 GB/s against a ~380 GB/s ceiling. It needs its per-token bytes
  roughly halved (quantisation or much higher speculation acceptance) before 20 tok/s is
  even arithmetically available.

**Recommendation: treat tok/s as the goal and bandwidth utilisation as a diagnostic.**
Every optimisation that removes redundant traffic raises tok/s and *lowers* the utilisation
percentage. Optimising for the 60% number directly rewards waste.

## 2. Measured today (2026-09-10)

GLM-5.3-Flash UD-Q4_K_XL, port 18131, TP4, 15 threads/socket, Q8 MTP draft, n_max=2.
512-token prose/code samples, temperature 0, seed 42, cache off, system-wide IMC counters.

| config | prose tok/s | code tok/s | prose GB/s | code GB/s | % of 380 |
|---|---:|---:|---:|---:|---:|
| selected baseline (draft on 4 sockets) | 16.30 | 16.61 | 193.4 | 198.7 | 51-52% |
| draft on 1 socket | 14.73 | 14.55 | 175.3 | 175.9 | 46% |

Short-form gate (256 tokens, 4 prompts) before/after the knob stack in section 5:
prose 17.14 -> 17.79, code 17.20 -> 18.23, math 19.04 -> 19.77, geo 17.22 -> 18.96.

## 3. Where GLM-5.3-Flash's memory traffic actually goes

Solved from the server's own counters plus a full GGUF tensor inventory of the Q4
checkpoint (1,412 tensors across 6 shards), classified against the `GLM5NEXT` branch of
`llama_meta_device_get_split_state` in `src/llama-model.cpp`:

- inventoried DRAM traffic per target forward pass: **16.01 GB**
- of which the **mirror tax is only 1.57 GB/token (10%)** — `attn_q_a`, `attn_kv_a_mqa`,
  the indexer tensors, `hc_attn_fn`/`hc_ffn_fn`, `ssm_f_a`/`ssm_g_a` and the F32 router
  match no split pattern and fall through to `SPLIT_AXIS_MIRRORED`, so all four sockets
  re-stream them. Worth fixing, but it is not the main cost.
- The budget closes to within 1% as:

```
per round (1 verify pass of n+1 tokens + 2 draft passes) = 28.6 GB, yielding 2.42 tokens
  dense / attention / mirrored, read once per round      10.74 GB  -> 4.44 GB per token
  routed experts, read once PER TOKEN IN THE BATCH       15.80 GB  -> 6.53 GB per token
  2 x MTP draft passes                                    2.10 GB  -> 0.87 GB per token
                                                                     -----------------
                                                                     11.84 GB per token
                                        (measured: 11.78 GB per token)
```

**The dominant cost is the routed experts, and it does not amortise.** The dense half of a
forward pass is paid once per *round* and speculation spreads it over 2.42 tokens, but a
verify batch of n+1 tokens routes to up to 8(n+1) distinct experts, so expert traffic
scales with tokens. Experts alone (6.53 GB/token) cap this model at ~30 tok/s at
199 GB/s even if everything else were free.

## 4. Negative results (each of these was never tested before today)

- **Draft placement on one socket is worse, not better.** 14.7 vs 16.3 tok/s and 175 vs
  193 GB/s. All 278 previously recorded runs used the same 4-socket draft device, and the
  standing note "never shard a small MTP draft across sockets" does **not** hold for this
  build. Total bytes moved were unchanged (6,094 vs 6,030 GB), which also proves the draft
  was never mirroring-bound — it is latency/compute-bound and benefits from 4 sockets.
- **Per-request `speculative.n_max` / `p_min` overrides are silently ignored** by this
  server build. `n_max` = 1, 2 and 4 all returned identical `draft_n` (154) and identical
  acceptance (113). Any past or future sweep done through request parameters measures
  nothing. Draft depth can only be changed by restarting with `--spec-draft-n-max`.
- **Host contention is not currently the limiter.** A contended run (load ~20, an unpinned
  pytest/Chrome suite) measured 16.30/16.61 against 16.90/16.63 for the recorded quiet
  runs. `isolate-background.sh` still confines docker/containerd/netdata correctly;
  Paseo, Thunderbird, Chrome and netdata's `apps.plugin` remain unconfined on 0-127, and
  `host-setup/quiesce-desktop.sh` (new, reversible) will confine them when it matters.

## 4b. Two more negative results on the draft (both new)

- **A Q4 draft head is 36% smaller and buys nothing.** Extracted `blk.45` + `output` +
  `token_embd` from the UD-Q4_K_XL checkpoint with
  `serving/glm53-flash/extract-glm5next-mtp-gguf.py` -> 5.94 GB vs the Q8 draft's 9.26 GB
  (`/dev/shm/GLM-5.3-Flash-MTP-Q4_K_XL-621d456e93e9.gguf`, run it under
  `uv run --script`; the system python has no numpy). Output was **byte-identical** and
  acceptance equal-or-better (code 157/196 vs 154/201), but decode was 17.51/18.08 vs
  17.79/18.23 -- i.e. no gain, ~1% slower.
- Combined with the single-socket result, this settles it: **the MTP draft pass is
  latency/op-bound, not bandwidth-bound.** Cutting its bytes by 36% and moving it between
  1 and 4 sockets both changed decode by <=1% in the wrong direction. The draft costs
  ~25% of round wall-time for ~7% of the bytes, and that cost is dispatch and dependency
  latency on a 1-layer graph, not streaming.

## 5. What moved the number

Four knobs the binary supports but nothing had ever set, applied together:

```
GGML_CPU_ROUTER_F16=1                    # F32 router -> F16; the router is mirrored, 0.81 GB/token
GGML_CPU_MOE_WEIGHTED_SUM_FUSION=1
GGML_CPU_MOE_DOWN_WEIGHTED_SUM_FUSION=1
GGML_CPU_Q4_K_REPACK=1                   # VNNI repack for the Q4_K gate/up experts
```

**+3.8% to +10.1%** on the 256-token gate; **+1.5% (prose) / +3.8% (code)** on the
512-token counter runs. Output is *not* byte-identical to the baseline (speculation is
exactness-preserving at temperature 0, so this is a real numeric change — `ROUTER_F16` is
an explicit precision reduction and the MoE fusions reorder accumulation). Semantic
correctness was verified by hand: `17*23` -> `391`, capital of France -> `Paris`, and the
code reasoning stays on-topic and high quality.

Dropping `ROUTER_F16` (the only precision-reducing knob) and keeping the other three still
gives most of it: prose 17.70 / code 17.00 / math 19.74 / geo 18.80.

**Caveat, stated plainly: this gain is close to the noise floor.** Two captures of the
*same* validated baseline read prose 17.14 and 17.57 (+2.5%) on the same host. The knob
stack needs interleaved repeats before it should be promoted; a single A/B at this
magnitude does not establish it.

### Disposition

The server on 18131 was **restored to its exact pre-session configuration** — argv and env
diffed byte-identical against `results/goal-0910-restore/{argv.txt,env.txt}`, and the four
gate prompts reproduce the baseline output hashes exactly. The knob stack is *not*
promoted: every variant changes numeric output, `GLM-FLASH-QUALITY-POLICY-20260907.md`
sets a near-lossless bar, and ~3% does not justify overriding it without a decision. Both
candidate env files are kept (`env-knobs.txt`, `env-knobs-noprec.txt`) and either can be
activated with one command.

Env file: `results/goal-0910-restore/env-knobs.txt`. Relaunch with
`./relaunch_flash_draft_device_0910.sh <draft-device> [env-file]`; the exact pre-existing
argv and env are captured in `results/goal-0910-restore/` for verbatim restore.

## 6. Honest per-model position

| model | now | 20 tok/s reachable? | what it would take |
|---|---|---|---|
| **GLM-5.3-Flash Q4** | 16.3-18.2 | **yes, closest** | ~+15% more. Levers left: split the 1.57 GB/token mirror tax (`attn_q_a`, `attn_kv_a_mqa`, indexer, `hc_*`, `ssm_f_a/g_a`); attention requant Q8_0->Q6_K (~11% of traffic, but it is a quality decision against the near-lossless policy); expert-dedup across the verify batch |
| **Qwen3.8-Flash-Next Q6** | 24-32 | **already met** | speed bar passed. 60% utilisation needs 44 tok/s and is the wrong target for this model |
| **GLM-5.3 Full Q4** | 7.9-9.6 | **no, not at 22.55 GB/token** | needs per-token bytes roughly halved before 20 tok/s is arithmetically possible |
| **DeepSeek-V4.1-Flash** | 1.78 | **not by tuning** | it is served by a PyTorch reference implementation, not a CPU inference engine. See below |

## 7. DeepSeek-V4.1-Flash is a different kind of problem

It is not slow because of NUMA placement, kernels or quantisation. It runs on the
**official PyTorch reference implementation** (`results/deepseek-v41-intake-0910/official/
inference/model.py`) behind an OpenAI-shaped shim on port 18170, with hand-written native
GEMM bridges. Prior sessions took it from 1.10 to 1.78 tok/s (+62%) with native FP4/FP8
GEMM, activation-quantisation reuse, grouped MoE and scripted sparse attention. That is
real work, and it is still **11x short of 20 tok/s**.

The architecture has no llama.cpp support: engram embedding tables (2 layers x ~384M
embeddings), hyper-connections with 20 Sinkhorn iterations, DSpark draft blocks, a
lightning-indexer sparse attention with candidate blocks, FP8 activations with FP4
experts, and `o_groups`/`o_lora` output factorisation. At ~8-10 GB/token it *should*
reach 38-47 tok/s at this machine's bandwidth, so the target is not unreasonable — but
the route is a GGUF conversion plus an architecture port into the fork, which is a
multi-session engine project of the same size as the `glm5next` port, not a tuning task.
Continuing to micro-optimise the Torch path cannot close an 11x gap.

## 8. Reproduce

```bash
cd ~/InfoSystemic/AI-Server/serving/fleet-0903
python3 measure-model-bandwidth.py <label> --port 18131 --pid <pid> \
        --alias glm-flash-goal --drafts 2 --tokens 512 --allowed-idle-pids ''
```
Results: `results/glmflash-q4-contended-0910b`, `results/glmflash-q4-draft1sock-0910`,
`results/glmflash-q4-knobs-v1-0910`. Correctness gate: `results/goal-0910-knobs/`.
