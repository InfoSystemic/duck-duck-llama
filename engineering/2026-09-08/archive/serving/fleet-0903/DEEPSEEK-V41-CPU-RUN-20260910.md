# DeepSeek V4.1 Flash: real CPU execution and local endpoint

The released text model is running on the SR950 at `http://127.0.0.1:18170/v1`, model ID `DeepSeek-V4.1-Flash`. The endpoint returns both JSON chat completions and streaming SSE. The complete verified greeting is “Hello! How can I help you today?” followed by EOS. [Independent run audit](results/deepseek-v41-real-run-audit-0910.json).

This is an initial CPU implementation of the publisher's graph, using 16 physical cores on one socket. The endpoint supports text, greedy sampling, one active request, a 256-token combined context, and up to 128 output tokens. Vision and DSpark are disabled. This is not a whole-server performance result or broad quality evaluation.

| Observation | Prefill | Decode | Downloads during measured request |
| --- | ---: | ---: | ---: |
| Initial native CPU path, warm complete greeting | 8.126 s | 0.482 tok/s | 0 |
| Retained endpoint with direct FP8 expansion, warm complete greeting | 6.137 s | 0.606 tok/s | 0 |

Each warm response contains ten generated tokens including EOS: one produced by prefill and nine by decode. These are short bring-up observations on different execution paths, not an isolated controlled kernel-speed comparison. No IMC bandwidth measurement was taken. The initial cold prefill took 364.556 seconds, including 14.213 GB of downloads. Subsequent cold steps also fetched experts and Engram rows. Cold download time must not be presented as RAM bandwidth or steady-state model speed. [First real run](results/deepseek-v41-checkpoint-run-0910b/generation.json), [retained endpoint greeting](results/deepseek-v41-server-greeting-0910/result.json).

All 93,338 declared text parameter shapes match the pinned released checkpoint at `fb2764a5cf321eaa5070ca8f9e892818f477c16d`. The graph retains all 40 layers, dimension 5,120, 384 routed experts per layer, six selected experts per token, the published Engram/hash layout, shared compressed attention, hierarchical indexer, FP4 KV rounding, and mHC equations. The initial cold and warm runs produce identical token IDs and identical FP32 logits hashes at every step. No synthetic parameter is used for real generation. The small synthetic graph fixtures remain separately labeled. [Full-shape check](results/deepseek-v41-checkpoint-run-0910b/shape-check.json).

The native checkpoint occupies 510.297 GB, while persistent storage had only about 23 GB free. The loader avoids full residency by caching native expert tensors and fetching the actual selected Engram rows. It does not substitute missing experts or zero-fill missing rows. The current cache contains 10,560 tensors and 38.842 GB, plus the previously retrieved 0.315 GB of Engram projection weights. The 90 GiB cache lives in `/dev/shm/deepseek-v41-native-fb2764-0910` and is lost on reboot. The endpoint preserves fetched native Engram rows across process restarts within that cache. New prompts can still incur substantial network latency.

Tensor downloads use the pinned public safetensors headers, exact HTTP 206 Content-Range and byte-count checks, per-range hashes, and complete downloaded-tensor SHA-256 records. The store checks cached tensor hashes before first reuse. Whole-shard LFS hashes have not been verified because only selected ranges are present. BF16-to-FP32 head/compressor promotion and FP8-to-BF16 grouped attention weights follow the official reference's conversions; routed FP4 and other FP8 weights are not requantized. [Loader](deepseek_v41_checkpoint_0910.py), [coalesced transport](deepseek_v41_checkpoint_transport_0910b.py).

The first real-weight attempt failed on an HTTP download error before a token was produced. Its exact cause was not captured. The successful resume reused its cache, coalesced adjacent tensors into ranges of at most 32 MiB, reused HTTP connections, spaced requests, and added bounded retries/backoff. It completed 4,195 requests with no retries. The failed result remains archived. [Failed attempt](results/deepseek-v41-checkpoint-run-0910/result.json), [successful transport](results/deepseek-v41-checkpoint-run-0910b/transport.json).

Native FP8/FP4 matrix kernels match the independent CPU reduction tree on 82,032 tested values across three worker counts. Direct FP8 expansion matches all 254 finite codes, preserves signed zero, recognizes both NaN codes, and leaves the 40-layer synthetic graph's logits unchanged. The standalone Engram projection/gate component is also archived; its gate has three BF16 boundary differences on actual-weight fixtures. The retained model uses the publisher's PyTorch gate equations rather than that standalone C++ gate. [Kernel checks](results/deepseek-v41-native-gemm-0910b/kernel-check.json), [format check](results/deepseek-v41-native-gemm-0910b/decode-check.json), [standalone projection/gate checks](results/deepseek-v41-engram-project-0910b/correctness.json).

The endpoint's cache evicts only journaled expert files in its owned directory. Common weights and symlinks are protected; interrupted eviction recovery and 24 streaming-tokenizer prefixes were checked. The retained process owns the lifecycle lock so older model-swap/benchmark controllers cannot unknowingly overlap it. Existing Flash Q4 PID 3437776 was preserved. Its residency is not a requirement for subsequent DeepSeek tuning. [Cache checks](results/deepseek-v41-server-0910/cache-check.json), [endpoint validation](results/deepseek-v41-server-0910/result.json), [selected endpoint](deepseek-v41-selected.json).

```bash
curl http://127.0.0.1:18170/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Hi."}],"temperature":0,"max_tokens":32,"stream":true}'
```

The next performance work is retaining hot expert parameter bindings, profiling the CPU graph, distributing work and weight pages across sockets, and replacing remaining costly Python/PyTorch operations. Long-context operation, vision, DSpark, task-quality comparisons, full resident storage, and measured bandwidth utilization remain open. Earlier claims that full-model integration was pending are superseded by this report; component-only results remain component-only evidence.

This refresh also preserves the completed Flash Q4 quiet/raw observations and shared-dispatch lifecycle fixture prepared before the DeepSeek goal took priority. No Flash shared-dispatch model speed gain is claimed, and the shared-dispatch candidate was not selected.

The two generated Flash shared-dispatch diagnostic patches use zero context and relative headers; apply them with `git apply --unidiff-zero`. Exact reconstruction of the unchanged candidate source is [verified separately](results/flash-q4-shared-dispatch-publication-0910.json). No executed build input or runtime was changed by that publication formatting.
