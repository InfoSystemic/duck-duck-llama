# Architecture and operating constraints

Duck Duck Llama contains two related implementation paths, tested on the same CPU server.

## Machine

| Resource | Recorded host |
| --- | --- |
| Processor | 4 × Intel Xeon Gold 6242 |
| Execution resources | 64 physical cores, 128 logical CPUs |
| Memory | Approximately 755 GiB usable RAM across four NUMA nodes |
| Inference accelerator | CPU only |
| Operating environment | Linux, C++/OpenMP, Python, llama.cpp/GGML |

Capacity and placement are separate constraints. An allocation bound to one NUMA node can fail while other nodes still have available memory. RAM-backed model staging also consumes physical capacity after the process that populated it exits.

## GLM and Qwen: GGUF through llama.cpp

The llama.cpp work adds or integrates per-node CPU devices, model-specific tensor distribution, persistent worker pools, collectives, quantized layouts, fused operations, and speculative-serving behavior. Different model architectures require different split rules; a working backend alone does not establish that an arbitrary tensor split is numerically valid.

The source bundles capture separate integration lines. Their base commits and reconstructed tree IDs are in the [latest source inventory](../engineering/2026-09-12/source-bundles.json). The [reproduction guide](reproducing.md) explains how complete patches and private library overlays relate.

## DeepSeek: native execution and a separate engine port

The native CPU path uses the publisher's model structure with native FP8/FP4 operator implementations, Python orchestration, sparse attention, and bounded tensor/Engram stores. It fetches missing checkpoint ranges and retains a limited working set. Short warm requests exercise a different storage regime from novel prompts.

The isolated JigSaw llama.cpp port is a separate integration effort. Linux prefetch and build fixes have been verified, but a successful build does not establish full-checkpoint inference or equivalence to the native CPU path. [DeepSeek guide](models/deepseek-v41.md).

## Evidence path

An experiment should connect five things: the source revision, the actual loaded libraries, the runtime settings, the input/output record, and the measurement window. The newer controllers verify process start identity and socket ownership, require the advertised model identity, and preserve live processes when an observation deadline expires.

Publication adds source hashes, filtered evidence, and patch reconstruction. Those checks establish that an artifact was faithfully exported. They do not turn a component test into a model-quality result.
