# Q8 batched kernel result

Status: built and exact-output validated; not deployed. Production remains on MTP cache-only catch-up, bounded KV rollback and wider pooling. There is no new measured model tok/s result from this experiment.

The candidate uses two independent integer dot-product chains and calculates activation sums/scales inline for 16-output Q8 batches of two, three or four tokens. Each final integer correction, float scale multiplication and sequential block FMA is preserved. Other dispatch shapes retain their original path. It defaults off behind GGML_CPU_Q8_BATCH_FAST. The exact deployed CPU library was rebuilt byte for byte before replacing only the frozen x86 repack object.

Candidate SHA256: e37362cdc7d2afd0a93f34b8b1099ec3b10bae99f09458f56acf1c355e129857.

## Exact gates

- Direct kernels: 18,448 signed-byte sum cases and 1,296 matrix cases, 1,782,648 compared float values. Outputs match both a scalar canonical calculation and the actual production library. Production, candidate off and candidate on save the same 7,130,592 bytes.
- Compiled graph validation: 242 cases in each of eight arms (production/off/on/alternating mode at one and fifteen workers). All 7,311,792 saved bytes per arm match the corresponding production worker configuration. The alternating arms each switch 968 times. Enabled invocation counters prove engagement.
- Four-NUMA graphs: all repeated target/draft outputs are exact across off/on modes, including a distinct-weight 32-layer workload. An excluded diagnostic run confirms 147,456 NR=3 candidate invocations in its enabled arm and zero in the disabled arm. Performance runs disable that counter.

## Performance evidence

Isolated three-token, 16-output kernels have about 1.13-1.34x speedups across the sampled reduction sizes and hot/rotating weight conditions. That gain mostly disappears in the four-socket synthetic graph.

| Synthetic workload | Cycle median speedup | Target median speedup | Aggregate cycle speedup |
| --- | ---: | ---: | ---: |
| Initial eight separate processes, four weight sets | 1.0161x | Not separated | 1.0258x / 1.0403x by block |
| Same process, four weight sets | 1.0050x | 1.0260x | 1.0142x |
| Same process, 32 distinct weight sets | 1.0082x | 1.0145x | 1.0225x |

The latter tests use ABBA then BAAB mode order, 61 measured cycles per arm, three-token target graphs and two one-token draft calls per cycle, 15 workers on each of four NUMA nodes, and the actual production inference environment. No artificial retained OpenMP team is added. The 32-set cycle aggregate block speedups are 1.0058x and 1.0395x. The unchanged draft phase also varies, so the aggregate difference cannot all be attributed to this kernel. These graphs omit model attention, experts, KV state and real speculative acceptance; they are not decode-throughput measurements.

Decision: retain the candidate and its exact gate, but do not deploy or claim a model gain from this small, variable synthetic effect. It is not evidence for the missing 8-10% needed on the standard short Codex fixture. A later full-model A/B may still find a small benefit; it would need isolated request accounting and automatic restoration.

The production PID, command, inference environment and mapped library hashes remain unchanged through every gate. No model outage occurred. Raw reports and logs are preserved in the candidate directory; result-summary.json pins their hashes.
